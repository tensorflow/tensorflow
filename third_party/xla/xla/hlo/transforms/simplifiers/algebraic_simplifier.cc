/* Copyright 2017 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/hlo/transforms/simplifiers/algebraic_simplifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/hlo/evaluator/hlo_evaluator.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_sharding_util.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/overflow_util.h"
#include "xla/permutation_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gather_scatter_utils.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/host_offload_utils.h"
#include "xla/service/memory_annotations.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

namespace m = match;

using primitive_util::NativeTypeOf;

// Unwraps broadcasts hunting for a constant.  If we find one, checks if the
// constant contains only the given value.
bool IsAll(const HloInstruction* op, int8_t value) {
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAll(op->operand(0), value);
    case HloOpcode::kConstant:
      return op->literal().IsAll(value);
    default:
      return false;
  }
}

// Unwraps broadcasts hunting for a constant.  If we find one, checks if the
// constant contains only the given value.
bool IsAllFloat(const HloInstruction* op, float value) {
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAllFloat(op->operand(0), value);
    case HloOpcode::kConstant:
      return op->literal().IsAllFloat(value);
    default:
      return false;
  }
}

bool IsAll(const HloInstruction* op, const Literal& scalar) {
  CHECK(ShapeUtil::IsScalar(scalar.shape()));
  switch (op->opcode()) {
    case HloOpcode::kBroadcast:
      return IsAll(op->operand(0), scalar);
    case HloOpcode::kConstant:
      return op->literal().IsAll(scalar);
    default:
      return false;
  }
}

bool IsAnyOperandComplex(const HloInstruction* hlo) {
  for (auto operand : hlo->operands()) {
    if (ShapeUtil::ElementIsComplex(operand->shape())) {
      return true;
    }
  }
  return false;
}

bool IsPositive(const HloInstruction* hlo,
                const AlgebraicSimplifierOptions& options) {
  // Utility only handles real types.
  if (IsAnyOperandComplex(hlo)) {
    return false;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kGetTupleElement: {
      const HloInstruction* gte_operand = hlo->operand(0);
      switch (gte_operand->opcode()) {
        case HloOpcode::kCustomCall: {
          const auto& target = gte_operand->custom_call_target();
          return target ==
                     options.get_cudnn_batchnorm_forward_training_metadata() &&
                 hlo->tuple_index() == 2;
        }
        default:
          return false;
      }
    }
    case HloOpcode::kPower:
    case HloOpcode::kAbs:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSqrt:
      return IsPositive(hlo->operand(0), options);

    case HloOpcode::kMultiply: {
      return hlo->operand(0) == hlo->operand(1) &&
             IsPositive(hlo->operand(0), options);
    }
    default:
      return false;
  }
}

static bool IsScalarConstant(const HloInstruction* hlo) {
  return hlo->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsEffectiveScalar(hlo->shape());
}

std::optional<double> GetConstantValue(const HloInstruction* inst) {
  if (!IsScalarConstant(inst)) {
    return std::nullopt;
  }
  return primitive_util::PrimitiveTypeSwitch<std::optional<double>>(
      [&](auto primitive_type_constant) -> std::optional<double> {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return static_cast<double>(
              inst->literal().GetFirstElement<NativeT>());
        } else if constexpr (primitive_util::IsIntegralType(
                                 primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return static_cast<int64_t>(
              inst->literal().GetFirstElement<NativeT>());
        }
        return std::nullopt;
      },
      inst->shape().element_type());
}

static bool IsScalarConstantZero(const HloInstruction* hlo) {
  if (!IsScalarConstant(hlo)) {
    return false;
  }
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return hlo->literal().GetFirstElement<NativeT>() ==
                 static_cast<NativeT>(0);
        }
        return false;
      },
      hlo->shape().element_type());
}

static bool IsScalarConstantNegInf(const HloInstruction* hlo) {
  if (!IsScalarConstant(hlo)) {
    return false;
  }
  if (primitive_util::IsComplexType(hlo->shape().element_type())) {
    return false;
  }
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          if constexpr (std::numeric_limits<NativeT>::has_infinity) {
            return hlo->literal().GetFirstElement<NativeT>() ==
                   -std::numeric_limits<NativeT>::infinity();
          }
          return hlo->literal().GetFirstElement<NativeT>() ==
                 std::numeric_limits<NativeT>::lowest();
        }
        return false;
      },
      hlo->shape().element_type());
}

static bool IsScalarConstantInf(const HloInstruction* hlo) {
  if (!IsScalarConstant(hlo)) {
    return false;
  }
  if (primitive_util::IsComplexType(hlo->shape().element_type())) {
    return false;
  }
  return primitive_util::PrimitiveTypeSwitch<bool>(
      [&](auto primitive_type_constant) -> bool {
        if constexpr (primitive_util::IsArrayType(primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          if constexpr (std::numeric_limits<NativeT>::has_infinity) {
            return hlo->literal().GetFirstElement<NativeT>() ==
                   std::numeric_limits<NativeT>::infinity();
          }
          return hlo->literal().GetFirstElement<NativeT>() ==
                 std::numeric_limits<NativeT>::max();
        }
        return false;
      },
      hlo->shape().element_type());
}

// Checks whether `op` is a floating-point constant or broadcast of a constant
// of the form +/- 2^k for some integer k positive, negative, or zero.  Such
// values are interesting because multiplying by a power of 2 just moves the
// exponent.
bool IsAllFpConstantPowerOf2(const HloInstruction* op) {
  // Unwrap the broadcast if necessary.
  const HloInstruction* c;
  if (!Match(op, m::ConstantEffectiveScalar(&c)) &&
      !Match(op, m::Broadcast(m::Constant(&c).WithShape(
                     m::Shape().IsEffectiveScalar())))) {
    return false;
  }
  auto val = GetConstantValue(c);
  if (!val) {
    return false;
  }

  int exp;
  double mantissa = std::frexp(*val, &exp);
  // frexp returns a value in the range (-1, -0.5] U [0.5, 1).  A return value
  // of +/-0.5 therefore indicates that the floating point value is a power of
  // 2.
  return mantissa == 0.5 || mantissa == -0.5;
}

// Returns whether the given transpose produces a result which is bit-wise
// identical to its operand and thus may be replaced with a bitcast.
bool TransposeIsBitcast(const HloInstruction* transpose) {
  CHECK_EQ(HloOpcode::kTranspose, transpose->opcode());
  const HloInstruction* operand = transpose->operand(0);
  return ShapeUtil::TransposeIsBitcast(operand->shape(), transpose->shape(),
                                       transpose->dimensions());
}

// Recursive helper for method below.
HloInstruction* BitcastingOperandOfReshapeOrCopyChainHelper(
    HloInstruction* instr, HloInstruction* operand,
    const AlgebraicSimplifierOptions& options) {
  // Can't replace chain of copies and reshapes with bitcasts if the compiler
  // used a memory layout which isn't compatible.
  if (options.ReshapeIsBitcast(operand->shape(), instr->shape())) {
    return operand;
  }

  // If the operand is a copy or reshape try to see if the operand's operand
  // would produce a bitcast with initial instruction.
  if (HloOpcode::kReshape == operand->opcode() ||
      HloOpcode::kCopy == operand->opcode()) {
    return BitcastingOperandOfReshapeOrCopyChainHelper(
        instr, operand->mutable_operand(0), options);
  }
  return nullptr;
}

// Returns an operand of a chain of reshapes and copies that is bit-wise
// identical to first reshape or copy in the chain.
HloInstruction* BitcastingOperandOfReshapeOrCopyChain(
    HloInstruction* instr, const AlgebraicSimplifierOptions& options) {
  if (!options.is_layout_sensitive()) {
    return nullptr;
  }
  CHECK(HloOpcode::kReshape == instr->opcode() ||
        HloOpcode::kCopy == instr->opcode());
  return BitcastingOperandOfReshapeOrCopyChainHelper(
      instr, instr->mutable_operand(0), options);
}

// Returns bool to determine whether a pair of converts can be eliminated.
bool IsConvertPairNoOp(const HloInstruction* convert) {
  //    [operand_convert]         [convert]
  // (src)->convert-(intermediate)->convert-(dest)
  const HloInstruction* operand_convert = convert->operand(0);
  if (operand_convert->opcode() != HloOpcode::kConvert) {
    return false;
  }
  const PrimitiveType src_type =
      operand_convert->operand(0)->shape().element_type();
  const PrimitiveType intermediate_type =
      operand_convert->shape().element_type();

  return src_type == convert->shape().element_type() &&
         primitive_util::CastPreservesValues(src_type, intermediate_type);
}

PrecisionConfig SwapOperandsInDotPrecisionConfig(PrecisionConfig config) {
  CHECK_EQ(config.operand_precision_size(), 2);
  std::swap(config.mutable_operand_precision()->at(0),
            config.mutable_operand_precision()->at(1));
  return config;
}

// Validate whether tiling and padding assignments in the bitcasted shapes
// will make the two shapes non-equivalent.
bool ValidateTilingOfBitcast(
    const Shape& bitcast_shape, const Shape& op_shape,
    const std::vector<std::vector<int64_t>>& operand_map) {
  if (op_shape.layout().tiles().empty() ||
      bitcast_shape.layout().tiles().empty()) {
    return true;
  }
  VLOG(2) << "op shape:" << op_shape.ToString(true) << "\n";
  VLOG(2) << "bitcast shape:" << bitcast_shape.ToString(true) << "\n";
  VLOG(2) << "operand_map size:" << operand_map.size() << "\n";
  auto op_tile = op_shape.layout().tiles(0);
  auto bitcast_tile = bitcast_shape.layout().tiles(0);
  int64_t num_of_tiled_dims = op_tile.dimensions().size(),
          tiled_dim_idx = num_of_tiled_dims - 1;
  if (bitcast_tile.dimensions().size() != num_of_tiled_dims) {
    return false;
  }
  for (auto op_dim : op_shape.layout().minor_to_major()) {
    VLOG(3) << "op_dim = " << op_dim << "\n";
    VLOG(3) << "tiled_dim_idx = " << tiled_dim_idx << "\n";
    VLOG(3) << "tiled_dim_size = " << op_tile.dimension(tiled_dim_idx) << ":"
            << bitcast_tile.dimension(tiled_dim_idx) << "\n";
    if (op_tile.dimensions()[tiled_dim_idx] !=
        bitcast_tile.dimensions()[tiled_dim_idx]) {
      VLOG(2) << "Abort b/c tiled dimension " << op_dim
              << " has different tiling sizes before and after bitcast.\n";
      return false;
    }
    if (operand_map.size() <= op_dim || operand_map[op_dim].empty()) {
      if (op_tile.dimensions()[tiled_dim_idx] != 1) {
        VLOG(2) << "Abort b/c tiled dimension " << op_dim << " has size 1.\n";
        return false;
      }
    } else if (bitcast_shape.dimensions_size() <= operand_map[op_dim][0]) {
      VLOG(2) << "Abort because the bitcasted dimensions are not aligned!\n";
      return false;
    } else if (bitcast_shape.dimensions(operand_map[op_dim][0]) <
               op_shape.dimensions(op_dim)) {
      if (operand_map[op_dim].size() == 1) {
        VLOG(2) << "Abort b/c a dimension (possibly padded) is shrank to a "
                   "smaller size.\n";
        return false;
      }
      if (tiled_dim_idx > 0) {
        VLOG(2) << "Abort b/c a non-major tiled dimension is split.\n";
        return false;
      }
      if (bitcast_shape.dimensions(operand_map[op_dim][0]) %
                  op_tile.dimensions()[tiled_dim_idx] !=
              0 ||
          op_shape.dimensions(op_dim) %
                  bitcast_shape.dimensions(operand_map[op_dim][0]) !=
              0) {
        VLOG(2) << "Abort b/c tiled dimension " << op_dim
                << " has been split in bitcasted layout\n";
        return false;
      }
    } else if (bitcast_shape.dimensions(operand_map[op_dim][0]) >
               op_shape.dimensions(op_dim)) {
      if (tiled_dim_idx > 0) {
        VLOG(2) << "Abort b/c a non-major tiled dimension is combined.\n";
        return false;
      }
      if (bitcast_shape.dimensions(operand_map[op_dim][0]) %
                  op_shape.dimensions(op_dim) !=
              0 ||
          op_shape.dimensions(op_dim) % op_tile.dimensions()[tiled_dim_idx] !=
              0) {
        VLOG(2) << "Abort b/c tiled dimension " << op_dim
                << " has been combined in bitcasted layout\n";
        return false;
      }
    }
    if (--tiled_dim_idx < 0) {
      break;
    }
  }
  return true;
}

// Constructs the maps that take dims of A and dims of B to dims of AB, mapping
// to -1 for dimensions not present in AB. For an example, consider we are
// computing a dot whose operands have shapes [m,n,p] and [n,q]. Assuming we
// contract over n, this produces an array with shape [m,p,q]. This function
// will return vectors map_a_ab = {0, -1, 1} and map_b_ab = {-1, 2}
std::pair<std::vector<int64_t>, std::vector<int64_t>> ConstructToDotMaps(
    DotDimensionNumbers dnums, const Shape& a_shape, const Shape& b_shape) {
  std::vector<int64_t> map_a_ab(a_shape.dimensions_size(), -1),
      map_b_ab(b_shape.dimensions_size(), -1);
  int64_t ab_index = 0;
  // Extract a and b contraction dimensions from dnums
  auto a_batch_dims = dnums.lhs_batch_dimensions();
  auto b_batch_dims = dnums.rhs_batch_dimensions();
  const auto& a_contracting_dims = dnums.lhs_contracting_dimensions();
  const auto& b_contracting_dims = dnums.rhs_contracting_dimensions();
  // First add the batch dimensions
  for (int64_t i = 0; i < a_batch_dims.size(); i++) {
    map_a_ab[a_batch_dims[i]] = ab_index;
    map_b_ab[b_batch_dims[i]] = ab_index;
    ab_index++;
  }
  // Then add the free dimensions from a
  for (int64_t a_index = 0; a_index < a_shape.dimensions_size(); a_index++) {
    if (!absl::c_linear_search(a_contracting_dims, a_index) &&
        !absl::c_linear_search(a_batch_dims, a_index)) {
      map_a_ab[a_index] = ab_index;
      ab_index++;
    }
  }
  // Finally add the free dimensions from b
  for (int64_t b_index = 0; b_index < b_shape.dimensions_size(); b_index++) {
    if (!absl::c_linear_search(b_contracting_dims, b_index) &&
        !absl::c_linear_search(b_batch_dims, b_index)) {
      map_b_ab[b_index] = ab_index;
      ab_index++;
    }
  }
  return {map_a_ab, map_b_ab};
}

// Constructs the maps that take dims of AB to dims of A and dims of B mapping
// to -1 for dimensions not present in A/B. For an example, consider we are
// computing a dot whose operands have shapes [m,n,p] and [n,q]. Assuming we
// contract over n, this produces an array with shape [m,p,q]. This function
// will return vectors map_ab_a = {0,2,-1} and map_ab_b = {-1,-1,1}
std::pair<std::vector<int64_t>, std::vector<int64_t>> ConstructFromDotMaps(
    const HloInstruction* dot, const Shape& a_shape, const Shape& b_shape) {
  // Reserve space for new maps
  std::vector<int64_t> map_ab_a(dot->shape().dimensions_size(), -1),
      map_ab_b(dot->shape().dimensions_size(), -1);
  // Construct the maps going in the opposite direction
  std::vector<int64_t> map_a_ab, map_b_ab;
  std::tie(map_a_ab, map_b_ab) =
      ConstructToDotMaps(dot->dot_dimension_numbers(), a_shape, b_shape);
  // Construct these maps by inverting those above
  int64_t a_index = 0;
  for (auto ab_index : map_a_ab) {
    if (ab_index != -1) {
      map_ab_a[ab_index] = a_index;
    }
    a_index++;
  }
  int64_t b_index = 0;
  for (auto ab_index : map_b_ab) {
    if (ab_index != -1) {
      map_ab_b[ab_index] = b_index;
    }
    b_index++;
  }
  return {map_ab_a, map_ab_b};
}

bool DotHasOnlyBatchAndContractingOnOneOperand(
    int64_t lhs_rank, int64_t rhs_rank, const DotDimensionNumbers dnums) {
  return (dnums.lhs_batch_dimensions_size() +
              dnums.lhs_contracting_dimensions_size() ==
          lhs_rank) ||
         (dnums.rhs_contracting_dimensions_size() +
              dnums.rhs_batch_dimensions_size() ==
          rhs_rank);
}

// Estimates the number of flops a reduce requires
int64_t GetReduceFlops(const HloInstruction* reduce) {
  int64_t reduce_product = 1;
  for (int64_t dim : reduce->dimensions()) {
    reduce_product *= reduce->operand(0)->shape().dimensions(dim);
  }
  // Reduce along a dimension of size n requires n-1 reductions
  return ShapeUtil::ElementsIn(reduce->shape()) * (reduce_product - 1);
}

}  // namespace

bool AlgebraicSimplifierVisitor::IsNonNegative(
    const HloInstruction* hlo, const AlgebraicSimplifierOptions& options) {
  // Utility only handles real types.
  if (IsAnyOperandComplex(hlo)) {
    return false;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kMultiply: {
      return hlo->operand(0) == hlo->operand(1);
    }
    case HloOpcode::kAbs:
    case HloOpcode::kExp:
    case HloOpcode::kIota: {
      return true;
    }
    case HloOpcode::kBroadcast: {
      return IsNonNegative(hlo->operand(0), options);
    }
    case HloOpcode::kConstant: {
      if (std::optional<double> value = GetConstantValue(hlo)) {
        // return false for -0.0, -Inf, NaNs and negative values
        return !std::signbit(*value) && !std::isnan(*value);
      }
      return false;
    }
    case HloOpcode::kMinimum: {
      return IsNonNegative(hlo->operand(0), options) &&
             IsNonNegative(hlo->operand(1), options);
    }
    case HloOpcode::kMaximum: {
      return IsNonNegative(hlo->operand(0), options) ||
             IsNonNegative(hlo->operand(1), options);
    }
    case HloOpcode::kPower: {
      return IsNonNegative(hlo->operand(0), options);
    }
    case HloOpcode::kSelect: {
      return IsNonNegative(hlo->operand(1), options) &&
             IsNonNegative(hlo->operand(2), options);
    }
    default:
      return IsPositive(hlo, options);
  }
}

void AlgebraicSimplifierVisitor::ResetState(HloComputation* computation) {
  ResetVisitStates();
  computation_ = computation;
}

bool AlgebraicSimplifierVisitor::Run(HloComputation* computation,
                                     const AlgebraicSimplifierOptions& options,
                                     AlgebraicSimplifier* simplifier) {
  ResetState(computation);
  TF_CHECK_OK(computation->Accept(this));
  return changed();
}

bool AlgebraicSimplifierVisitor::SameShape(const HloInstruction* lhs,
                                           const HloInstruction* rhs) const {
  return SameShape(lhs->shape(), rhs->shape());
}

bool AlgebraicSimplifierVisitor::SameShape(const Shape& lhs,
                                           const Shape& rhs) const {
  if (options_.is_layout_sensitive()) {
    return ShapeUtil::Equal(lhs, rhs);
  } else {
    return ShapeUtil::Compatible(lhs, rhs);
  }
}

namespace {

bool IsOpCodeMultiplyCommutative(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kMultiply:
    case HloOpcode::kTranspose:
    case HloOpcode::kReshape:
    case HloOpcode::kSelect:
      return true;
    default:
      return false;
  }
}

std::unique_ptr<HloInstruction> MakeScalarInstruction(HloInstruction* target,
                                                      float multiplier) {
  return primitive_util::PrimitiveTypeSwitch<std::unique_ptr<HloInstruction>>(
      [&](auto primitive_type_constant) -> std::unique_ptr<HloInstruction> {
        if constexpr (primitive_util::IsFloatingPointType(
                          primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<NativeT>(static_cast<NativeT>(multiplier)));
        } else if constexpr (primitive_util::IsIntegralType(
                                 primitive_type_constant)) {
          using NativeT = NativeTypeOf<primitive_type_constant>;
          return HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<NativeT>(static_cast<NativeT>(multiplier)));
        }
        LOG(FATAL) << "Unsupported data type: "
                   << target->shape().element_type();
      },
      target->shape().element_type());
}

}  // namespace

absl::Status AlgebraicSimplifierVisitor::ScalarMultiplyReduction(
    HloInstruction* dot) {
  // We only process bfloat16 and float32 for now.
  if (dot->shape().element_type() != BF16 &&
      dot->shape().element_type() != F32) {
    return absl::OkStatus();
  }

  auto lhs = dot->mutable_operand(0);
  auto rhs = dot->mutable_operand(1);

  const int64_t dot_size = ShapeUtil::ElementsIn(dot->shape());
  const int64_t lhs_size = ShapeUtil::ElementsIn(lhs->shape());
  const int64_t rhs_size = ShapeUtil::ElementsIn(rhs->shape());

  HloInstruction* target = nullptr;
  // (current node, user, operand_index)
  std::vector<std::tuple<HloInstruction*, HloInstruction*, int64_t>> operands;
  std::vector<HloInstruction*> users;

  // Find which side of dot has the smallest size:
  // operand 0, operand 1, or output.
  if (dot_size <= std::min(lhs_size, rhs_size)) {
    target = dot;
    if (dot_size < lhs_size) {
      operands.emplace_back(lhs, dot, 0);
    }
    if (dot_size < rhs_size) {
      operands.emplace_back(rhs, dot, 1);
    }
  } else if (lhs_size <= rhs_size) {
    target = lhs;
    if (lhs_size < rhs_size) {
      operands.emplace_back(rhs, dot, 1);
    }
    if (lhs_size < dot_size && dot->user_count() == 1) {
      users.push_back(dot->users().front());
    }
  } else {
    target = rhs;
    if (rhs_size < lhs_size) {
      operands.emplace_back(lhs, dot, 0);
    }
    if (rhs_size < dot_size && dot->user_count() == 1) {
      users.push_back(dot->users().front());
    }
  }

  std::vector<float> values;

  // DFS to find scalar multiply ops from the operands.
  while (!operands.empty()) {
    HloInstruction* inst;
    HloInstruction* user;
    int64_t index;
    std::tie(inst, user, index) = operands.back();
    operands.pop_back();

    // Skip the op types that are not commutative with multiply.
    if (!IsOpCodeMultiplyCommutative(inst->opcode())) {
      continue;
    }

    HloInstruction* operand;
    HloInstruction* multiplier;
    // Pattern match a scalar multiply.
    if (Match(inst, m::MultiplyAnyOrder(
                        m::Op(&operand),
                        m::Broadcast(m::ConstantScalar(&multiplier))))) {
      CHECK_LT(index, user->operand_count());
      CHECK_EQ(inst, user->operands()[index]);

      // When found a scalar multiply, save its scalar value.
      values.push_back(*GetConstantValue(multiplier));
      // And remove the scalar multiply op.
      TF_RETURN_IF_ERROR(user->ReplaceOperandWith(index, operand));
      inst = operand;
    }

    // Push the operands of inst.
    int64_t i = 0;
    for (auto* operand : inst->operands()) {
      operands.emplace_back(operand, inst, i++);
    }
  }

  // DFS to find scalar multiply ops from the users.
  while (!users.empty()) {
    auto inst = users.back();
    users.pop_back();

    if (!IsOpCodeMultiplyCommutative(inst->opcode())) {
      continue;
    }

    HloInstruction* operand;
    HloInstruction* multiplier;
    if (Match(inst, m::MultiplyAnyOrder(
                        m::Op(&operand),
                        m::Broadcast(m::ConstantScalar(&multiplier))))) {
      values.push_back(*GetConstantValue(multiplier));

      TF_RETURN_IF_ERROR(inst->ReplaceAllUsesWith(operand));
      inst = operand;
    }

    // Process the instructions with only one user.
    // Otherwise moving scalar multiply to the operands changes the values of
    // other users.
    if (inst->user_count() == 1) {
      users.push_back(inst->users().front());
    }
  }

  if (values.empty()) {
    return absl::OkStatus();
  }

  MarkAsChanged();

  // Combine all constant multipliers.
  float multiplier = 1.0;
  for (const float v : values) {
    multiplier *= v;
  }

  // Create a new const scalar multiply instruction.
  HloInstruction* new_const_inst;
  new_const_inst =
      target->AddInstruction(MakeScalarInstruction(target, multiplier));

  // Broadcast the scalar multiplier.
  HloInstruction* new_broadcast = target->AddInstruction(
      HloInstruction::CreateBroadcast(target->shape(), new_const_inst, {}));
  // Create a new scalar multiply instruction.
  HloInstruction* new_multiply =
      target->AddInstruction(HloInstruction::CreateBinary(
          target->shape(), HloOpcode::kMultiply, target, new_broadcast));
  CHECK_EQ(new_multiply->shape(), target->shape());

  // Update the dependency with the rest of the instructions.
  if (target == lhs) {
    return dot->ReplaceOperandWith(0, new_multiply);
  } else if (target == rhs) {
    return dot->ReplaceOperandWith(1, new_multiply);
  } else {
    CHECK_EQ(target, dot);
    return dot->ReplaceAllUsesWith(new_multiply);
  }
}

void AlgebraicSimplifierVisitor::ReplaceWithBitcast(HloInstruction* instruction,
                                                    HloInstruction* operand) {
  CHECK_EQ(1, instruction->operand_count());
  if (operand == nullptr) {
    operand = instruction->mutable_operand(0);
  }
  CHECK_EQ(ShapeUtil::ElementsIn(instruction->shape()),
           ShapeUtil::ElementsIn(operand->shape()));
  CHECK_EQ(ShapeUtil::ByteSizeOf(instruction->shape()),
           ShapeUtil::ByteSizeOf(operand->shape()));

  auto bitcast = instruction->AddInstruction(
      HloInstruction::CreateBitcast(instruction->shape(), operand));
  TF_CHECK_OK(ReplaceInstruction(instruction, bitcast));
}

// Replace the old instruction with the new one if they are compatible, i.e.,
// 1. they have same shape
// 2. the replacement will not cause loss of sharding
bool AlgebraicSimplifierVisitor::ReplaceInstructionIfCompatible(
    HloInstruction* old_instruction, HloInstruction* new_instruction) {
  // It's tricky for the simplifier to determine whether
  // it should remove the op when control deps are present. I.e.
  // control deps might be added to preserve a certain order.
  // It's better to not process in that case.
  if (!old_instruction->control_predecessors().empty()) {
    VLOG(3) << old_instruction->ToString()
            << " has control predecessors, skipping.";
    return false;
  }

  if (!SameShape(old_instruction, new_instruction)) {
    return false;
  }
  return ReplaceInstruction(old_instruction, new_instruction,
                            /*preserve_sharding=*/true)
      .value();
}

bool AlgebraicSimplifierVisitor::ReplaceInstructionIfCompatible(
    HloInstruction* old_instruction,
    absl::Span<HloInstruction* const> new_instructions) {
  // It's tricky for the simplifier to determine whether
  // it should remove the op when control deps are present. I.e.
  // control deps might be added to preserve a certain order.
  // It's better to not process in that case.
  if (!old_instruction->control_predecessors().empty()) {
    VLOG(3) << old_instruction->ToString()
            << " has control predecessors, skipping.";
    return false;
  }

  if (new_instructions.size() == 1) {
    return ReplaceInstructionIfCompatible(old_instruction, new_instructions[0]);
  }
  CHECK(!new_instructions.empty());
  if (!old_instruction->shape().IsTuple() ||
      old_instruction->shape().tuple_shapes_size() != new_instructions.size()) {
    return false;
  }
  for (int i = 0, n = new_instructions.size(); i < n; ++i) {
    if (!SameShape(old_instruction->shape().tuple_shapes(i),
                   new_instructions[i]->shape())) {
      return false;
    }
  }
  return ReplaceInstruction(old_instruction, MaybeMakeTuple(new_instructions),
                            /*preserve_sharding=*/true)
      .value();
}

absl::Status AlgebraicSimplifierVisitor::HandleAbs(HloInstruction* abs) {
  HloInstruction* abs_operand = abs->mutable_operand(0);
  VLOG(10) << "trying transform [Abs(A) => A] " << abs->ToString()
           << " Abs operand is: " << abs_operand->ToString();
  if (IsNonNegative(abs->operand(0), options_)) {
    return ReplaceInstruction(abs, abs_operand);
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(add, m::Add(m::Op(&lhs), m::Op(&rhs))));

  // A + 0 => A
  VLOG(10) << "trying transform [A + 0 => A]: " << add->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(add, lhs)) {
    return absl::OkStatus();
  }
  // 0 + A => A
  VLOG(10) << "trying transform [0 + A => A]: " << add->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfCompatible(add, rhs)) {
    return absl::OkStatus();
  }

  // Canonicalization: Put constants on the right.  This makes the reassociation
  // rules below simpler.
  VLOG(10) << "trying transform [Const + A => A + Const]";
  if (Match(add, m::Add(m::Constant(), m::NonConstant()))) {
    return ReplaceWithNewInstruction(
        add,
        HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, rhs, lhs));
  }

  // Reassociate to allow constant folding.
  //
  // Note: This is not general.  For example, we won't reassociate
  //
  //   (A + C1) + (B + C2) =>  A + B + (C1 + C2).
  //
  VLOG(10) << "trying transform [(A + C1) + C2 => A + (C1 + C2)]";
  HloInstruction *a, *c1, *c2;
  if (Match(add, m::Add(m::Add(m::NonConstant(&a), m::Constant(&c1)),
                        m::Constant(&c2))) ||
      Match(add, m::Add(m::Add(m::NonConstant(&a),
                               m::Broadcast(m::ConstantScalar(&c1))),
                        m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(auto* sum_of_constants,
                        MakeBinaryHlo(HloOpcode::kAdd, c1, c2));
    if (ShapeUtil::IsScalar(sum_of_constants->shape()) &&
        !ShapeUtil::IsScalar(add->shape())) {
      sum_of_constants = add->AddInstruction(
          HloInstruction::CreateBroadcast(add->shape(), sum_of_constants, {}));
    }
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kAdd, a,
                                          sum_of_constants));
  }

  VLOG(10) << "trying transform [(C1 - A) + C2 => (C1 + C2) - A]";
  if (Match(add, m::Add(m::Subtract(m::Constant(&c1), m::NonConstant(&a)),
                        m::Constant(&c2))) ||
      Match(add, m::Add(m::Subtract(m::Broadcast(m::ConstantScalar(&c1)),
                                    m::NonConstant(&a)),
                        m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(HloInstruction * sum_of_constants,
                        MakeBinaryHlo(HloOpcode::kAdd, c1, c2));
    if (ShapeUtil::IsScalar(sum_of_constants->shape()) &&
        !ShapeUtil::IsScalar(add->shape())) {
      sum_of_constants = add->AddInstruction(
          HloInstruction::CreateBroadcast(add->shape(), sum_of_constants, {}));
    }
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(add->shape(), HloOpcode::kSubtract,
                                          sum_of_constants, a));
  }

  // Convert add with fullshape into add with partial shape when a
  // portion of add is effective:
  //             zero (fullshape)   rhs (partialshape)
  // .           |                  |
  // . lhs .    dynamic_update_slice (fullshape)
  // . |         |
  // Add (fullshape)
  //
  // to:
  //              lhs
  //              |
  //             dynamic_slice (partialshape)   rhs (partialshape)
  // .           |                      |
  // . lhs .    add (partial_shape)+----+
  // . |         |
  // dynamic_update_slice (fullshape)
  //
  // This is pattern is discovered in control flow V2 gradient update.
  if (Match(add,
            m::AddAnyOrder(
                m::Op(&lhs),
                m::Op(&rhs)
                    .WithOpcode(HloOpcode::kDynamicUpdateSlice)
                    .WithOperand(
                        0, m::Broadcast(m::ConstantEffectiveScalar(0)))))) {
    const Shape& partial_shape = rhs->operand(1)->shape();
    auto sliced_lhs = lhs->AddInstruction(HloInstruction::CreateDynamicSlice(
        partial_shape, lhs, absl::MakeSpan(rhs->operands()).subspan(2),
        partial_shape.dimensions()));

    auto add_partial = rhs->AddInstruction(
        HloInstruction::CreateBinary(rhs->operand(1)->shape(), HloOpcode::kAdd,
                                     sliced_lhs, rhs->mutable_operand(1)));

    auto dynamic_update_slice_full = HloInstruction::CreateDynamicUpdateSlice(
        lhs->shape(), lhs, add_partial,
        absl::MakeSpan(rhs->operands()).subspan(2));

    return ReplaceWithNewInstruction(add, std::move(dynamic_update_slice_full));
  }

  // A*C + B*C => (A+B)*C
  //
  //  - If A, B, and C are integers, do this unconditionally. Proof of
  //    correctness: https://rise4fun.com/Alive/u9X.
  //
  //  - If A, B, and C are floating point, do this if C is a scalar constant or
  //    broadcast of scalar constant and is equal to +/- 2^k for some (possibly
  //    negative) integer k.
  //
  //    Multiplying by a power of 2 just moves the exponent, so our answer is
  //    exact modulo rounding of intermediate results so long as
  //
  //     - none of the three products has an exponent which underflows (so the
  //       result is 0 or denormal), and
  //     - none of the three products overflows to inf.
  //
  //    Proof: See algebraic_simplifier_proof_distributive_property.py.
  //
  //    We deem these differences in rounding, underflow, and overflow
  //    acceptable in the ML context.
  //
  //    Furthermore, if `enable_floats_are_real` is true, the simplification is
  //    done nonetheless. This might cause numerical differences even if there
  //    is no underflow or overflow.
  HloInstruction *b, *c;
  if (((Match(lhs, m::Multiply(m::Op(&a), m::Op(&c))) &&
        Match(rhs, m::MultiplyAnyOrder(m::Op().Is(c), m::Op(&b)))) ||
       (Match(lhs, m::Multiply(m::Op(&c), m::Op(&a))) &&
        Match(rhs, m::MultiplyAnyOrder(m::Op().Is(c), m::Op(&b))))) &&
      // Make sure we would decrease the number of multiplies.
      (lhs->user_count() == 1 && rhs->user_count() == 1) &&
      (ShapeUtil::ElementIsIntegral(add->shape()) ||
       options_.enable_floats_are_real() || IsAllFpConstantPowerOf2(c))) {
    return ReplaceWithNewInstruction(
        add, HloInstruction::CreateBinary(
                 add->shape(), HloOpcode::kMultiply,
                 lhs->AddInstruction(HloInstruction::CreateBinary(
                     add->shape(), HloOpcode::kAdd, a, b)),
                 c));
  }

  if (options_.is_layout_sensitive()) {
    return absl::OkStatus();
  }

  HloInstruction* lhs_scatter_operand = nullptr;
  HloInstruction* rhs_scatter_operand = nullptr;
  HloInstruction* lhs_scatter_update = nullptr;
  HloInstruction* rhs_scatter_update = nullptr;
  HloInstruction* lhs_scatter_index = nullptr;
  HloInstruction* rhs_scatter_index = nullptr;
  bool lhs_scatter = Match(lhs, m::Scatter(m::Op(&lhs_scatter_operand),
                                           m::Op(&lhs_scatter_index),
                                           m::Op(&lhs_scatter_update))
                                    .WithOneUse()) &&
                     Match(lhs->to_apply()->root_instruction(),
                           m::Add(m::Parameter(), m::Parameter()));
  bool rhs_scatter = Match(rhs, m::Scatter(m::Op(&rhs_scatter_operand),
                                           m::Op(&rhs_scatter_index),
                                           m::Op(&rhs_scatter_update))
                                    .WithOneUse()) &&
                     Match(rhs->to_apply()->root_instruction(),
                           m::Add(m::Parameter(), m::Parameter()));
  if (rhs_scatter && lhs_scatter) {
    const auto& lhs_dnums = lhs->scatter_dimension_numbers();
    const auto& rhs_dnums = rhs->scatter_dimension_numbers();
    std::optional<int64_t> index_concat_dimension;
    std::optional<int64_t> update_concat_dimension;
    // Don't try to combine scatters of different ranks.
    if (lhs_scatter_index->shape().dimensions_size() !=
        rhs_scatter_index->shape().dimensions_size()) {
      return absl::OkStatus();
    }

    int64_t first_index_dim = lhs_scatter_index->shape().dimensions_size();
    int64_t first_update_dim = lhs_scatter_update->shape().dimensions_size();
    // Find a dimension where it is possible to concatenate the indices and
    // updates. This is the first and only non-equal dimension or the first
    // equally sized dimension.
    for (int64_t d = lhs_scatter_index->shape().dimensions_size() - 1,
                 update_dim = lhs_scatter_update->shape().dimensions_size() - 1;
         d >= 0; --d) {
      if (d == lhs_dnums.index_vector_dim()) {
        continue;
      }
      // Skip the dimensions that are in the update window before we subtract 1
      // from `update_dim` for the next iteration.
      while (
          absl::c_linear_search(lhs_dnums.update_window_dims(), update_dim)) {
        --update_dim;
      }
      if (absl::c_linear_search(lhs_dnums.scatter_indices_batching_dims(), d)) {
        // Corresponding batch dimensions in updates, scatter_indices and inputs
        // have the same sizes. So we can't concatenate a batch dim in updates
        // and scatter_indices without changing inputs. Instead, we ensure the
        // two scatter instructions have the same batch dimensions to support
        // the transformation.
        if (lhs_scatter_index->shape().dimensions(d) !=
            rhs_scatter_index->shape().dimensions(d)) {
          // This shouldn't be reachable as we currently only combine two
          // scatter instructions feeding into the same add straightforwardly,
          // which should have the same result shapes.
          return absl::OkStatus();
        }
        update_dim--;
        continue;
      }
      if (lhs_scatter_index->shape().dimensions(d) ==
          rhs_scatter_index->shape().dimensions(d)) {
        first_index_dim = d;
        first_update_dim = update_dim--;
        continue;
      }
      // More than one dimension of unequal size was found, bail out.
      if (index_concat_dimension) {
        return absl::OkStatus();
      }
      index_concat_dimension = d;
      update_concat_dimension = update_dim--;
    }
    if (!index_concat_dimension) {
      index_concat_dimension = first_index_dim;
      update_concat_dimension = first_update_dim;
    }

    // A scalar scatter will require additional reshapes of the index and
    // update.
    if (*index_concat_dimension ==
        lhs_scatter_index->shape().dimensions_size()) {
      return absl::OkStatus();
    }
    const bool update_concat_is_cheap =
        ShapeUtil::ElementsIn(rhs_scatter_update->shape()) +
            ShapeUtil::ElementsIn(lhs_scatter_update->shape()) <
        ShapeUtil::ElementsIn(lhs->shape());
    if (!update_concat_is_cheap) {
      return absl::OkStatus();
    }
    const bool same_dimension_numbers =
        lhs_dnums.index_vector_dim() == rhs_dnums.index_vector_dim() &&
        absl::c_equal(lhs_dnums.scatter_dims_to_operand_dims(),
                      rhs_dnums.scatter_dims_to_operand_dims()) &&
        absl::c_equal(lhs_dnums.inserted_window_dims(),
                      rhs_dnums.inserted_window_dims()) &&
        absl::c_equal(lhs_dnums.update_window_dims(),
                      rhs_dnums.update_window_dims()) &&
        absl::c_equal(lhs_dnums.scatter_indices_batching_dims(),
                      rhs_dnums.scatter_indices_batching_dims()) &&
        absl::c_equal(lhs_dnums.input_batching_dims(),
                      rhs_dnums.input_batching_dims());
    const bool index_concat_is_safe =
        !lhs->unique_indices() && !rhs->unique_indices() &&
        !DynCast<HloScatterInstruction>(lhs)->indices_are_sorted() &&
        !DynCast<HloScatterInstruction>(rhs)->indices_are_sorted();

    Shape lhs_update_window = ShapeUtil::FilterDimensions(
        [&](int64_t dim) {
          return absl::c_linear_search(lhs_dnums.update_window_dims(), dim);
        },
        lhs_scatter_update->shape());
    Shape rhs_update_window = ShapeUtil::FilterDimensions(
        [&](int64_t dim) {
          return absl::c_linear_search(rhs_dnums.update_window_dims(), dim);
        },
        rhs_scatter_update->shape());
    // Concatenate the indices and updates
    if (index_concat_is_safe && same_dimension_numbers &&
        index_concat_dimension &&
        lhs_scatter_index->shape().element_type() ==
            rhs_scatter_index->shape().element_type() &&
        ShapeUtil::SameDimensions(lhs_update_window, rhs_update_window)) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                          MakeBinaryHlo(HloOpcode::kAdd, lhs_scatter_operand,
                                        rhs_scatter_operand));
      TF_ASSIGN_OR_RETURN(HloInstruction * new_index,
                          MakeConcatHlo({lhs_scatter_index, rhs_scatter_index},
                                        *index_concat_dimension));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_update,
          MakeConcatHlo({lhs_scatter_update, rhs_scatter_update},
                        *update_concat_dimension));
      return ReplaceWithNewInstruction(
          add, HloInstruction::CreateScatter(
                   add->shape(), new_operand, new_index, new_update,
                   lhs->to_apply(), lhs_dnums, false, false));
    }
    TF_ASSIGN_OR_RETURN(HloInstruction * new_operand,
                        MakeBinaryHlo(HloOpcode::kAdd, lhs_scatter_operand,
                                      rhs_scatter_operand));
    TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(0, new_operand));
    TF_RETURN_IF_ERROR(lhs->ReplaceOperandWith(0, rhs));
    return ReplaceInstruction(add, lhs);
  } else if (rhs_scatter) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        MakeBinaryHlo(HloOpcode::kAdd, lhs, rhs_scatter_operand));
    TF_RETURN_IF_ERROR(rhs->ReplaceOperandWith(0, new_operand));
    return ReplaceInstruction(add, rhs);
  } else if (lhs_scatter) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_operand,
        MakeBinaryHlo(HloOpcode::kAdd, lhs_scatter_operand, rhs));
    TF_RETURN_IF_ERROR(lhs->ReplaceOperandWith(0, new_operand));
    return ReplaceInstruction(add, lhs);
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::TrySimplifyTautologicalCompare(
    HloInstruction* conjunction) {
  HloInstruction *lhs, *rhs;
  if (!Match(conjunction, m::And(m::Op(&lhs), m::Op(&rhs)))) {
    return false;
  }
  struct LessThanCompareInfo {  // (LT var constant)
    HloInstruction* var;
    int64_t constant;
  };

  auto get_compare_info =
      [&](HloInstruction* cmp) -> std::optional<LessThanCompareInfo> {
    HloInstruction *lhs, *rhs;
    auto scalar_shape_matcher =
        m::Shape().IsEffectiveScalar().WithElementType(PrimitiveType::S32);
    if (Match(cmp, m::Compare(m::Op(&lhs),
                              m::Constant(&rhs).WithShape(scalar_shape_matcher))
                       .WithComparisonDirection(ComparisonDirection::kLt))) {
      return {LessThanCompareInfo{lhs, *rhs->literal().GetFirstInteger()}};
    } else if (Match(
                   cmp,
                   m::Compare(m::Constant(&lhs).WithShape(scalar_shape_matcher),
                              m::Op(&rhs))
                       .WithComparisonDirection(ComparisonDirection::kGt))) {
      return {LessThanCompareInfo{rhs, *lhs->literal().GetFirstInteger()}};
    }
    return std::nullopt;
  };

  std::optional<LessThanCompareInfo> lhs_info = get_compare_info(lhs);
  std::optional<LessThanCompareInfo> rhs_info = get_compare_info(rhs);
  if (lhs_info && rhs_info && lhs_info->var == rhs_info->var) {
    int64_t new_bound = std::min(lhs_info->constant, rhs_info->constant);
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        conjunction,
        HloInstruction::CreateCompare(lhs->shape(), lhs_info->var,
                                      MakeScalarLike(lhs_info->var, new_bound),
                                      ComparisonDirection::kLt)));
    return true;
  }
  return false;
}

absl::Status AlgebraicSimplifierVisitor::HandleAllGather(
    HloInstruction* all_gather) {
  if (all_gather->shape().IsArray() &&
      Match(all_gather->mutable_operand(0),
            m::Broadcast(m::ConstantScalar()))) {
    return ReplaceWithNewInstruction(
        all_gather,
        all_gather->mutable_operand(0)->CloneWithNewShape(all_gather->shape()));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleAllToAll(
    HloInstruction* all_to_all) {
  if (all_to_all->shape().IsArray() &&
      Match(all_to_all->mutable_operand(0),
            m::Broadcast(m::ConstantScalar()))) {
    return ReplaceInstruction(all_to_all, all_to_all->mutable_operand(0));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleAnd(
    HloInstruction* logical_and) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_and, m::And(m::Op(&lhs), m::Op(&rhs))));
  // Simplify logical and
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A && True => A
    VLOG(10) << "trying transform [A && True => A]: "
             << logical_and->ToString();
    if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(logical_and, lhs)) {
      return absl::OkStatus();
    }
    // True && A => A
    VLOG(10) << "trying transform [True && A => A]: "
             << logical_and->ToString();
    if (IsAll(lhs, 1) && ReplaceInstructionIfCompatible(logical_and, rhs)) {
      return absl::OkStatus();
    }
  }

  // A && False => False or A & 0 => 0
  VLOG(10) << "trying transform [A && False => False]: "
           << logical_and->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(logical_and, rhs)) {
    return absl::OkStatus();
  }

  // False && A => False or A & 0 => 0
  VLOG(10) << "trying transform [False && A => False]: "
           << logical_and->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfCompatible(logical_and, lhs)) {
    return absl::OkStatus();
  }

  // Simplify tautological conjunctions.
  TF_ASSIGN_OR_RETURN(bool found_tautological_compare,
                      TrySimplifyTautologicalCompare(logical_and));
  if (found_tautological_compare) {
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleBitcast(
    HloInstruction* bitcast) {
  // It's tricky for the simplifier to determine whether
  // it should remove the op when control deps are present. I.e.
  // control deps might be added to preserve a certain order.
  // It's better to not process in that case.
  if (!bitcast->control_predecessors().empty()) {
    VLOG(3) << bitcast->ToString() << " has control predecessors, skipping.";
    return absl::OkStatus();
  }
  // If a bitcast feeds a bitcast, make it a single bitcast.
  // Make sure the whole chain of bitcasts is optimized.
  if (bitcast->operand(0)->opcode() == HloOpcode::kBitcast) {
    TF_RETURN_IF_ERROR(HandleBitcast(bitcast->mutable_operand(0)));
  }
  HloInstruction* op;
  if (Match(bitcast, m::Bitcast(m::Bitcast(m::Op(&op))))) {
    auto new_bitcast = HloInstruction::CreateBitcast(bitcast->shape(), op);
    HloInstruction* new_bitcast_ptr = new_bitcast.get();
    TF_RETURN_IF_ERROR(
        ReplaceWithNewInstruction(bitcast, std::move(new_bitcast)));
    bitcast = new_bitcast_ptr;
  }

  HloInstruction* new_bitcast = bitcast->mutable_operand(0);
  // Below this point avoid bitcast optimizations with mismatched data types.
  if (!ShapeUtil::SameElementType(bitcast->shape(), new_bitcast->shape())) {
    return absl::OkStatus();
  }

  // All bitcasts can be eliminated (assuming layout constraints are satisfied).
  if (ReplaceInstructionIfCompatible(bitcast, new_bitcast)) {
    bitcast = new_bitcast;
  }

  // Check whether we can potentially simplify the bitcast into a broadcast
  // operand.
  if (bitcast->opcode() == HloOpcode::kBitcast &&
      bitcast->operand(0)->opcode() == HloOpcode::kBroadcast) {
    // Make sure the bitcast and the broadcast have the same tiling.
    bool enable_broadcast = bitcast->operand(0)->shape().layout().tiles() ==
                            bitcast->shape().layout().tiles();
    if (enable_broadcast) {
      // DeduceTransposeDimensionsForBitcast() checks whether the bitcast is a
      // transpose and returns the dimensions attribute if it is.
      auto dimensions = ShapeUtil::DeduceTransposeDimensionsForBitcast(
          bitcast->operand(0)->shape(), bitcast->shape());
      if (dimensions.has_value()) {
        return SimplifyTransposeOfBroadcast(bitcast, dimensions.value());
      }
    }
  }

  return absl::OkStatus();
}

// Compute a pair of maps for a bitcast operation, specifically between its
// result logical dimensions and the original logical dimensions of the operand.
// The maps are computed by matching the physical layout dimensions
// (minor-to-major) of the operands and the bitcasted result. Overall they
// record how the different logical dimensions of the operand may be combined or
// split in the resulting shape and in which orders they are combined/split. The
// function returns  std::nullopt if unsuccessful (e.g., such a logical
// dimension mapping cannot be constructed due to cases like bitcasting {4,4} to
// {2,8}.
std::optional<std::vector<std::vector<int64_t>>>
AlgebraicSimplifierVisitor::ComputeBitcastDimMap(const Shape& bitcast_shape,
                                                 const Shape& operand_shape) {
  std::vector<std::vector<int64_t>> operand_dim_map(
      operand_shape.dimensions_size());
  int64_t bitcast_rank = bitcast_shape.dimensions_size();
  int64_t operand_rank = operand_shape.dimensions_size();
  int64_t cur_bitcast_size = 1, cur_operand_size = 1;
  int64_t operand_pos = -1, operand_dim = -1;
  for (int64_t bitcast_pos = 0; bitcast_pos < bitcast_rank; ++bitcast_pos) {
    int64_t bitcast_dim = bitcast_shape.layout().minor_to_major(bitcast_pos);
    if (operand_pos >= operand_rank) {
      if (bitcast_shape.dimensions(bitcast_dim) != 1) {
        VLOG(3) << "Abort b/c bitcasted size is bigger than operand size.\n";
        return std::nullopt;
      }
      continue;
    }
    CHECK_LT(bitcast_dim, bitcast_shape.dimensions_size());
    int64_t bitcast_dim_size = bitcast_shape.dimensions()[bitcast_dim];
    auto prev_bitcast_size = cur_bitcast_size;
    cur_bitcast_size *= bitcast_dim_size;
    VLOG(2) << "bitcast pos = " << bitcast_pos << "\n";
    VLOG(2) << "bitcast size = " << cur_bitcast_size << "\n";
    if (cur_operand_size < cur_bitcast_size &&
        prev_bitcast_size < cur_operand_size) {
      // Here we are bitcasting (m1,n1) to (m2,n2), with m1 > m2 and m2 * n2
      // < m1, so (m1,n1) is re-partitioned instead of split or combined.
      VLOG(3) << "Abort b/c re-partitioning a group of dimensions is not "
                 "supported. \n";
      return std::nullopt;
    }
    while (operand_pos < operand_rank) {
      if (operand_pos < 0 || cur_operand_size < cur_bitcast_size) {
        VLOG(2) << "operand size < bitcase size\n";
        operand_pos++;
        if (operand_pos >= operand_rank) {
          VLOG(2)
              << "Abort due to size inconsistency: bitcasted size > operand "
                 "size.\n";
          return std::nullopt;
        }
        operand_dim = operand_shape.layout().minor_to_major(operand_pos);
        int64_t op_dim_size = operand_shape.dimensions()[operand_dim];
        cur_operand_size *= op_dim_size;
        VLOG(3) << "operand size = " << cur_operand_size << "\n";
        if (cur_operand_size > cur_bitcast_size &&
            op_dim_size < bitcast_dim_size && operand_pos > 0) {
          // Here we are bitcasting (m1,n1) to (m2,n2), with n1 < n2 and m1 * n1
          // > m2, so (m1,n1) is re-partitioned instead of split or combined.
          VLOG(3) << "Abort b/c re-partitioning a group of dimensions is not "
                     "supported. \n";
          return std::nullopt;
        }
      }
      CHECK_GE(operand_dim, 0);
      if (operand_shape.dimensions(operand_dim) > 1) {
        CHECK_LT(operand_dim, operand_dim_map.size());
        operand_dim_map[operand_dim].push_back(bitcast_dim);
        VLOG(3) << "operand dim_map[operand_dim] add " << bitcast_dim << " at "
                << operand_dim << "\n";
      }
      if (cur_operand_size >= cur_bitcast_size) {
        VLOG(3) << cur_operand_size << ">=" << cur_bitcast_size << "\n";
        CHECK_GE(operand_dim, 0);
        // If operand_dim is a degenerate one, move on to the next dimension.
        if (operand_shape.dimensions()[operand_dim] == 1) {
          operand_pos++;
        }
        break;
      }
    }
  }
  return operand_dim_map;
}

std::optional<Shape> AlgebraicSimplifierVisitor::ReshapeLayoutDimensions(
    const Shape& original_shape, const Shape& result_shape,
    const std::vector<std::vector<int64_t>>& original_map,
    const std::vector<std::vector<int64_t>>& result_map) {
  auto original_dimensions = original_shape.layout().minor_to_major();
  Shape new_shape = result_shape;
  auto* reshaped_dimensions =
      new_shape.mutable_layout()->mutable_minor_to_major();
  int64_t bitcast_pos = -1;
  for (int64_t op_pos = 0; op_pos < original_dimensions.size(); ++op_pos) {
    int64_t op_dim = original_dimensions[op_pos];
    VLOG(3) << "op_pos = " << op_pos << "\n";
    VLOG(3) << "op_dim = " << op_dim << "\n";
    if (original_map.size() <= op_dim) {
      VLOG(3) << "Skip due to original_map has too few dimensions.\n";
      continue;
    }
    auto bit_dims = original_map[op_dim];
    for (int64_t bitcast_dim : bit_dims) {
      VLOG(3) << "Add new reshaped dimension:" << bitcast_dim << "\n";
      if (bitcast_pos < 0 ||
          (*reshaped_dimensions)[bitcast_pos] != bitcast_dim) {
        bitcast_pos++;
        // If bitcast_pos has been over incremented, the new bitcast would
        // have to combine non-contiguous dimensions in op. Abort.
        if (bitcast_pos >= reshaped_dimensions->size()) {
          VLOG(3) << "bitcast pos is over incremented:" << bitcast_pos << "\n";
          return std::nullopt;
        }
        if (result_shape.dimensions(bitcast_dim) == 1) {
          // Postpone all degenerated dimensions (those with size 1) to the end.
          continue;
        }
        (*reshaped_dimensions)[bitcast_pos] = bitcast_dim;
      }
      auto op_dims = result_map[bitcast_dim];
      if (op_dims.size() > 1 && op_pos > 0) {
        // Check that op dimensions that are combined into bitcast_dim are not
        // non-contiguous or reordered to be different from how they appear in
        // result_map.
        int64_t op_dim_prev = original_dimensions[op_pos - 1];
        // If the current dimension is not the first being combined into
        // bitcast_dim, or is not contiguous with the previous dimension, abort.
        if (op_dims[0] != op_dim &&
            (original_map[op_dim_prev].empty() ||
             original_map[op_dim_prev][0] != bitcast_dim)) {
          VLOG(2) << "Abort b/c op dimensions that are combined into "
                     "bitcast_dim are not contiguous in the result. \n ";
          return std::nullopt;
        }
        // Now perform the dimension re-ordering check in the bitcast.
        for (int i = 0; i < op_dims.size(); ++i) {
          if (op_dims[i] == op_dim_prev) {
            if (i == op_dims.size() - 1 || op_dims[i + 1] != op_dim) {
              VLOG(2) << "Abort b/c op dimensions that are combined into "
                         "bitcast_dim are reordered in the new bitcast. \n ";
              return std::nullopt;
            }
          }
        }
      }
    }
  }
  for (int i = 0; i < result_shape.dimensions_size(); ++i) {
    if (result_shape.dimensions(i) == 1) {
      bitcast_pos++;
      // Since there is a possibility of over-incrementing bitcast_pos
      // we need such a check here also before accessing the vector.
      // Overincrementing is possible when the result's dimension is
      // smaller than the original dimension.
      if (bitcast_pos >= reshaped_dimensions->size()) {
        VLOG(3) << "bitcast pos is over incremented:" << bitcast_pos << "\n";
        return std::nullopt;
      }
      (*reshaped_dimensions)[bitcast_pos] = i;
    }
  }
  CHECK_EQ(bitcast_pos + 1, result_shape.dimensions_size());
  return new_shape;
}

std::vector<std::vector<int64_t>>
AlgebraicSimplifierVisitor::InvertBitcastDimMap(
    const Shape& original_shape, const Shape& bitcast_shape,
    const std::vector<std::vector<int64_t>>& original_map) {
  std::vector<std::vector<int64_t>> result_map(bitcast_shape.dimensions_size());
  // Invert the operand map into result map.
  for (auto i = 0; i < original_shape.dimensions_size(); ++i) {
    auto j = original_shape.layout().minor_to_major(i);
    VLOG(3) << "traversing minor to major (" << i << ")=" << j;
    for (auto k : original_map[j]) {
      VLOG(3) << "setting result_map[" << k << "] = " << j << "\n";
      result_map[k].push_back(j);
    }
  }
  return result_map;
}

bool AlgebraicSimplifierVisitor::SwapCopyBitcastCopy(
    HloInstruction* root_copy) {
  if (root_copy->opcode() != HloOpcode::kCopy) {
    return false;
  }
  if (host_offload_utils::IsSynchronousCopyFromOrToHost(root_copy)) {
    return false;
  }
  HloInstruction* bitcast = root_copy->mutable_operand(0);
  if (bitcast->opcode() != HloOpcode::kBitcast) {
    return false;
  }
  // All bitcasts above can be collapsed.
  HloInstruction* copy = bitcast->mutable_operand(0);
  while (copy->opcode() == HloOpcode::kBitcast) {
    copy = copy->mutable_operand(0);
  }
  if (copy->opcode() != HloOpcode::kCopy) {
    return false;
  }
  if (host_offload_utils::IsSynchronousCopyFromOrToHost(copy)) {
    return false;
  }
  VLOG(2) << "Processing " << copy->ToString() << "\n"
          << bitcast->ToString() << "\n"
          << root_copy->ToString() << "\n";
  HloInstruction* op = copy->mutable_operand(0);
  // Compute a pair of maps between op dimensions and bitcast dimensions.
  auto dim_map = ComputeBitcastDimMap(bitcast->shape(), copy->shape());
  if (!dim_map.has_value()) {
    VLOG(3) << "Failed to compute bitcast map.";
    return false;
  }
  std::vector<std::vector<int64_t>> operand_map = dim_map.value();
  if (!ValidateTilingOfBitcast(bitcast->shape(), copy->shape(), operand_map)) {
    VLOG(2) << "Abort because bitcast changes tiling assignment.\n";
    return false;
  }
  std::vector<std::vector<int64_t>> result_map =
      InvertBitcastDimMap(copy->shape(), bitcast->shape(), operand_map);
  if (ValidateTilingOfBitcast(bitcast->shape(), op->shape(), operand_map)) {
    auto new_shape = ReshapeLayoutDimensions(op->shape(), bitcast->shape(),
                                             operand_map, result_map);
    if (!new_shape.has_value() || !IsValidLayout(new_shape.value())) {
      return false;
    }
    auto repl = HloInstruction::CreateUnary(
        root_copy->shape(), HloOpcode::kCopy,
        bitcast->AddInstruction(
            bitcast->CloneWithNewOperands(new_shape.value(), {op})));
    VLOG(2) << "Replace with " << repl->operand(0)->ToString() << "\n"
            << repl->ToString() << "\n";
    TF_CHECK_OK(ReplaceWithNewInstruction(root_copy, std::move(repl)));
    return true;
  }

  if (ValidateTilingOfBitcast(copy->shape(), root_copy->shape(), result_map)) {
    auto new_shape = ReshapeLayoutDimensions(root_copy->shape(), copy->shape(),
                                             result_map, operand_map);
    if (!new_shape.has_value() || !IsValidLayout(new_shape.value())) {
      return false;
    }
    auto repl = HloInstruction::CreateUnary(
        root_copy->shape(), HloOpcode::kBitcast,
        bitcast->AddInstruction(
            root_copy->CloneWithNewOperands(new_shape.value(), {op})));
    VLOG(2) << "Replace with " << repl->operand(0)->ToString() << "\n"
            << repl->ToString() << "\n";
    TF_CHECK_OK(ReplaceWithNewInstruction(root_copy, std::move(repl)));
    return true;
  }
  return false;
}

absl::Status AlgebraicSimplifierVisitor::HandleBitcastConvert(
    HloInstruction* bitcast) {
  auto operand = bitcast->mutable_operand(0);

  // In a chain of BitcastConverts, only keep the last one.
  if (HloOpcode::kBitcastConvert == operand->opcode()) {
    return ReplaceWithNewInstruction(
        bitcast, HloInstruction::CreateBitcastConvert(
                     bitcast->shape(), operand->mutable_operand(0)));
  }

  TF_ASSIGN_OR_RETURN(bool replaced,
                      TrySimplifyTautologicalBitcastConvert(bitcast));
  if (replaced) {
    return absl::OkStatus();
  }
  // Eliminate bitcast converts between same shape.
  ReplaceInstructionIfCompatible(bitcast, bitcast->mutable_operand(0));
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleCopy(HloInstruction* copy) {
  if (SwapCopyBitcastCopy(copy)) {
    return absl::OkStatus();
  }
  // If a copy feeds a copy, make it a single copy.
  HloInstruction* op;
  if (Match(copy, m::Copy(m::Copy(m::Op(&op))))) {
    if (ShapeUtil::Equal(op->shape(), copy->shape())) {
      return ReplaceInstruction(copy, op);
    }
    return ReplaceWithNewInstruction(
        copy, HloInstruction::CreateUnary(copy->shape(), HloOpcode::kCopy, op));
  }
  // All copies can be eliminated (assuming layout constraints are satisfied).
  if ((!copy->has_sharding() ||
       copy->GetModule()->entry_computation()->root_instruction() != copy) &&
      ReplaceInstructionIfCompatible(copy, copy->mutable_operand(0))) {
    return absl::OkStatus();
  }

  const bool copy_is_to_different_memory_space =
      options_.is_layout_sensitive() && copy->shape().has_layout() &&
      copy->operand(0)->shape().has_layout() &&
      copy->shape().layout().memory_space() !=
          copy->operand(0)->shape().layout().memory_space();
  if (!copy_is_to_different_memory_space) {
    // Do not replace a copy between different memory spaces with a bitcast.
    HloInstruction* bitcast_operand =
        BitcastingOperandOfReshapeOrCopyChain(copy, options_);
    if (bitcast_operand != nullptr) {
      ReplaceWithBitcast(copy, bitcast_operand);
      return absl::OkStatus();
    }
  }

  // Replace Copy(Reshape()) with Reshape() if the Reshape is a logical bitcast.
  if (copy->operand(0)->opcode() == HloOpcode::kReshape &&
      copy->operand(0)->user_count() == 1 &&
      ShapeUtil::ReshapeIsBitcast(copy->operand(0)->shape(), copy->shape())) {
    return ReplaceWithNewInstruction(
        copy,
        copy->operand(0)->CloneWithNewOperands(
            copy->shape(), {copy->mutable_operand(0)->mutable_operand(0)}));
  }

  if (options_.is_layout_sensitive()) {
    // Try to reorder reshape-copy to copy-reshape.
    HloInstruction* copy_before = nullptr;
    HloInstruction* reshape = nullptr;
    if (Match(copy, m::Copy(m::Reshape(&reshape, m::Op()).WithOneUser()))) {
      Match(reshape, m::Reshape(m::Copy(&copy_before, m::Op()).WithOneUser()));

      HloInstruction* reshape_operand = reshape->mutable_operand(0);
      bool reshape_is_hardware_bitcast =
          options_.ReshapeIsBitcast(reshape_operand->shape(), reshape->shape());

      if (auto aligned_shape = ShapeUtil::AlignLayouts(
              copy->shape(), reshape_operand->shape())) {
        // We now have the option to do copy-reshape instead of
        // reshape-copy.
        Shape new_copy_shape = std::move(*aligned_shape);
        simplifier_->UpdateLayout(&new_copy_shape);
        bool new_reshape_is_hardware_bitcast =
            options_.ReshapeIsBitcast(new_copy_shape, copy->shape());

        bool should_rewrite = false;
        if (!reshape_is_hardware_bitcast) {
          if (new_reshape_is_hardware_bitcast) {
            // Can turn a reshape into a bitcast.
            should_rewrite = true;
          } else if (copy_before != nullptr) {
            // Neither reshapes are hardware bitcast.
            // Still can put two copies next to each other for a merge.
            should_rewrite = true;
          }
        } else if (new_reshape_is_hardware_bitcast) {
          if (copy_before != nullptr) {
            // Both reshapes are hardware bitcast.
            // Still can put two copies next to each other for a merge.
            should_rewrite = true;
          }
        }

        if (should_rewrite) {
          // Can now cut down the number of ops. Make sure the memory usage
          // does not increase too much.
          int64_t total_shape_size_before_rewrite = 0;
          if (copy_before != nullptr) {
            total_shape_size_before_rewrite +=
                ShapeUtil::ArraySize(copy_before->shape());
          }
          if (!reshape_is_hardware_bitcast) {
            total_shape_size_before_rewrite +=
                ShapeUtil::ArraySize(reshape->shape());
          }
          total_shape_size_before_rewrite +=
              ShapeUtil::ArraySize(copy->shape());

          int64_t total_shape_size_after_rewrite = 0;
          total_shape_size_after_rewrite +=
              ShapeUtil::ArraySize(new_copy_shape);
          if (!new_reshape_is_hardware_bitcast) {
            total_shape_size_after_rewrite +=
                ShapeUtil::ArraySize(copy->shape());
          }

          if (total_shape_size_after_rewrite >
              10 * total_shape_size_before_rewrite / 9) {
            should_rewrite = false;
          }
        }

        if (should_rewrite) {
          // The two copies become no-op.
          bool can_remove_copy =
              (copy_before != nullptr) &&
              Shape::Equal().IgnoreMemorySpaceInLayout()(
                  new_copy_shape, copy_before->operand(0)->shape());
          HloInstruction* new_copy =
              can_remove_copy
                  ? copy_before->mutable_operand(0)
                  : copy->AddInstruction(HloInstruction::CreateUnary(
                        new_copy_shape, HloOpcode::kCopy, reshape_operand));
          auto new_reshape = copy->AddInstruction(
              new_reshape_is_hardware_bitcast
                  ? HloInstruction::CreateBitcast(copy->shape(), new_copy)
                  : HloInstruction::CreateReshape(copy->shape(), new_copy));
          VLOG(5) << "Replace reshape-copy with copy-reshape: "
                  << reshape->ToString() << ", " << copy->ToString() << " => "
                  << new_copy->ToString() << ", " << new_reshape->ToString();
          if (copy_before != nullptr) {
            VLOG(5) << "Copy-before: " << copy_before->ToString();
          }
          return ReplaceInstruction(copy, new_reshape);
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleConcatenate(
    HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  if (operands.size() == 1) {
    // Unary concatenates are useless.
    ReplaceInstructionIfCompatible(concatenate, operands[0]);
    return absl::OkStatus();
  }
  // Filter out and remove empty operands.
  std::vector<HloInstruction*> nonempty_operands;
  for (HloInstruction* operand : operands) {
    if (!ShapeUtil::IsZeroElementArray(operand->shape())) {
      nonempty_operands.push_back(operand);
    }
  }
  if (nonempty_operands.size() < operands.size()) {
    HloInstruction* replacement;
    if (nonempty_operands.empty()) {
      replacement = operands[0];
    } else if (nonempty_operands.size() == 1) {
      replacement = nonempty_operands[0];
    } else {
      replacement =
          concatenate->AddInstruction(concatenate->CloneWithNewOperands(
              concatenate->shape(), nonempty_operands));
    }
    VLOG(10) << "trying to replace " << concatenate->ToString() << " with "
             << replacement->ToString();
    ReplaceInstructionIfCompatible(concatenate, replacement);
    return absl::OkStatus();
  }

  if (options_.is_layout_sensitive()) {
    return absl::OkStatus();
  }

  // concat(x, concat(y, z)) -> concat(x, y, z).  We only do this in
  // layout-insensitive mode because some backends may have (late,
  // layout-sensitive) passes that break up ops with many operands into smaller
  // pieces.  This would undo that.
  absl::InlinedVector<HloInstruction*, 8> unnested_concat_operands;
  for (HloInstruction* operand : operands) {
    if (operand->opcode() == HloOpcode::kConcatenate &&
        operand->concatenate_dimension() ==
            concatenate->concatenate_dimension()) {
      for (HloInstruction* instr : operand->operands()) {
        unnested_concat_operands.push_back(instr);
      }
    } else {
      unnested_concat_operands.push_back(operand);
    }
  }
  if (unnested_concat_operands.size() != concatenate->operand_count()) {
    return ReplaceWithNewInstruction(
        concatenate, HloInstruction::CreateConcatenate(
                         concatenate->shape(), unnested_concat_operands,
                         concatenate->concatenate_dimension()));
  }

  // Check if we can merge "adjacent" slice operands which take slices from the
  // same other op. For simplicity we only merge unstrided slices.
  int64_t concatenate_dimension = concatenate->concatenate_dimension();
  std::vector<HloInstruction*> new_operands;
  int64_t i = 0;
  while (i < operands.size()) {
    if (operands[i]->opcode() != HloOpcode::kSlice ||
        !hlo_instruction_utils::IsUnstridedSlice(operands[i])) {
      new_operands.push_back(operands[i]);
      ++i;
      continue;
    }
    int64_t slice_end = operands[i]->slice_limits(concatenate_dimension);
    HloInstruction* slice_operand = operands[i]->mutable_operand(0);
    int64_t j = i + 1;
    while (j < operands.size()) {
      if (operands[j]->opcode() != HloOpcode::kSlice ||
          !hlo_instruction_utils::IsUnstridedSlice(operands[j]) ||
          operands[j]->operand(0) != slice_operand ||
          operands[j]->slice_starts(concatenate_dimension) != slice_end) {
        break;
      }
      // Check that all the slice_start values are the same in all other
      // dimensions. This implies that the slice_limit values are also the same,
      // because operands of concatenate need to have the same shape, and we
      // already checked that the slices are unstrided.
      bool same_other_starts = true;
      for (int64_t k = 0; k < operands[j]->slice_starts().size(); ++k) {
        if (k == concatenate_dimension) {
          continue;
        }
        if (operands[i]->slice_starts(k) != operands[j]->slice_starts(k)) {
          same_other_starts = false;
          break;
        }
      }
      if (!same_other_starts) {
        break;
      }
      slice_end = operands[j]->slice_limits(concatenate_dimension);
      ++j;
    }
    if (j - i > 1) {
      Shape new_slice_shape = operands[i]->shape();
      new_slice_shape.set_dimensions(
          concatenate_dimension,
          slice_end - operands[i]->slice_starts(concatenate_dimension));
      simplifier_->UpdateLayout(&new_slice_shape);
      auto new_limit_indices = operands[i]->slice_limits();
      new_limit_indices[concatenate_dimension] = slice_end;
      auto new_slice_op =
          operands[i]->AddInstruction(HloInstruction::CreateSlice(
              new_slice_shape, slice_operand,
              /*start_indices=*/operands[i]->slice_starts(),
              /*limit_indices=*/new_limit_indices,
              /*strides=*/operands[i]->slice_strides()));
      new_operands.push_back(new_slice_op);
    } else {
      new_operands.push_back(operands[i]);
    }
    i = j;
  }
  if (new_operands.size() < operands.size()) {
    auto replacement = concatenate->AddInstruction(
        concatenate->CloneWithNewOperands(concatenate->shape(), new_operands));
    ReplaceInstructionIfCompatible(concatenate, replacement);
    return absl::OkStatus();
  }

  if (operands.size() == 2) {
    // A binary concat with a broadcasted scalar as an operand can be converted
    // into a pad which is simpler to fold into other operations.
    bool is_effective_low_pad = Match(
        operands[0], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    bool is_effective_high_pad = Match(
        operands[1], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    if (!is_effective_low_pad && !is_effective_high_pad) {
      return absl::OkStatus();
    }
    PaddingConfig padding_config;
    for (int64_t dim = 0; dim < operands[0]->shape().dimensions_size(); ++dim) {
      auto padding_config_dim = padding_config.add_dimensions();
      padding_config_dim->set_edge_padding_high(0);
      padding_config_dim->set_edge_padding_low(0);
      padding_config_dim->set_interior_padding(0);
      if (dim == concatenate_dimension) {
        if (is_effective_low_pad) {
          padding_config_dim->set_edge_padding_low(
              operands[0]->shape().dimensions(dim));
        } else {
          padding_config_dim->set_edge_padding_high(
              operands[1]->shape().dimensions(dim));
        }
      }
    }
    int64_t operand_to_pad = is_effective_low_pad ? 1 : 0;
    int64_t pad_value_operand = is_effective_low_pad ? 0 : 1;
    HloInstruction* pad = concatenate->AddInstruction(HloInstruction::CreatePad(
        concatenate->shape(), operands[operand_to_pad],
        operands[pad_value_operand]->mutable_operand(0), padding_config));
    return ReplaceInstruction(concatenate, pad);
  }

  if (absl::c_count(operands, operands[0]) == operands.size() &&
      operands[0]->shape().dimensions(concatenate_dimension) == 1) {
    Shape new_shape = operands[0]->shape();
    DimensionVector broadcast_dims;
    for (int64_t i = 0; i < new_shape.dimensions_size(); ++i) {
      if (i == concatenate_dimension) {
        continue;
      }
      broadcast_dims.push_back(i);
    }
    new_shape.DeleteDimension(concatenate_dimension);
    return ReplaceInstruction(
        concatenate,
        MakeBroadcastHlo(MakeReshapeHlo(new_shape, operands[0]).value(),
                         broadcast_dims, concatenate->shape()));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool>
AlgebraicSimplifierVisitor::TrySimplifyTautologicalBitcastConvert(
    HloInstruction* bitcast) {
  CHECK_EQ(bitcast->opcode(), HloOpcode::kBitcastConvert);
  PrimitiveType outer_to = bitcast->shape().element_type();
  HloInstruction* concat = bitcast->mutable_operand(0);
  if (concat->opcode() != HloOpcode::kConcatenate) {
    return false;
  }
  std::vector<HloInstruction*> outer_inputs;
  std::vector<HloInstruction*> to_remove_bitcasts;
  for (int i = 0; i < concat->operand_count(); i++) {
    HloInstruction* in = concat->mutable_operand(i);
    if (in->opcode() != HloOpcode::kBitcastConvert ||
        in->operand(0)->shape().element_type() != outer_to) {
      return false;
    }
    outer_inputs.push_back(in->mutable_operand(0));
    to_remove_bitcasts.push_back(in);
  }

  const int64_t concat_dim = concat->concatenate_dimension();
  TF_ASSIGN_OR_RETURN(HloInstruction * new_concat,
                      MakeConcatHlo(outer_inputs, concat_dim));
  TF_RETURN_IF_ERROR(ReplaceInstruction(bitcast, new_concat));

  return true;
}

absl::Status
AlgebraicSimplifierVisitor::TryRemoveUpcastAndDowncastSurroundingBinaryOp(
    HloInstruction* convert_instruction) {
  HloInstruction* arg_1 = nullptr;
  HloInstruction* arg_2 = nullptr;
  HloInstruction* bin_op_instr = nullptr;
  HloInstruction* final_convert_instr = nullptr;

  // also catch constants. For an example, look at
  // cudnn_fused_conv_rewriter.cc's IsLosslesslyConvertibleTo().
  auto arg_1_pattern = m::Convert(m::Op(&arg_1)).WithOneUser();
  auto arg_2_pattern = m::Convert(m::Op(&arg_2)).WithOneUser();

  auto is_unsigned_int_pred = [](const HloInstruction* instr) {
    // Only unsigned integer division/remainder is safe. Signed integer division
    // can result in undefined behavior. For example, in S8 consider -128/-1.
    return primitive_util::IsUnsignedIntegralType(
        instr->shape().element_type());
  };

  auto bin_op_pattern =
      m::Convert(&final_convert_instr,
                 m::AnyOf<HloInstruction>(
                     m::Add(&bin_op_instr, arg_1_pattern, arg_2_pattern),
                     m::Subtract(&bin_op_instr, arg_1_pattern, arg_2_pattern),
                     m::Multiply(&bin_op_instr, arg_1_pattern, arg_2_pattern),
                     m::Divide(&bin_op_instr, arg_1_pattern, arg_2_pattern)
                         .WithPredicate(is_unsigned_int_pred),
                     m::Remainder(&bin_op_instr, arg_1_pattern, arg_2_pattern)
                         .WithPredicate(is_unsigned_int_pred))
                     .WithOneUser());

  if (!Match(convert_instruction, bin_op_pattern)) {
    return absl::OkStatus();
  }

  const PrimitiveType arg_1_type = arg_1->shape().element_type();
  const PrimitiveType arg_2_type = arg_2->shape().element_type();
  const PrimitiveType final_type = final_convert_instr->shape().element_type();

  if (arg_1_type != final_type || arg_2_type != final_type) {
    // Only match when the series of instructions ends with the same types that
    // it started with.
    return absl::OkStatus();
  }

  const PrimitiveType bin_op_type = bin_op_instr->shape().element_type();
  if (!primitive_util::IsIntegralType(final_type) ||
      !primitive_util::IsIntegralType(bin_op_type) ||
      primitive_util::IsSubByteNonPredType(final_type) ||
      primitive_util::IsSubByteNonPredType(bin_op_type) ||
      (primitive_util::IsSignedIntegralType(final_type) !=
       primitive_util::IsSignedIntegralType(bin_op_type)) ||
      (primitive_util::IsUnsignedIntegralType(final_type) !=
       primitive_util::IsUnsignedIntegralType(bin_op_type))) {
    // So far, only the safety of this transformation with same signedness
    // non-4-bit integer types has been verified.
    return absl::OkStatus();
  }

  // Ensure that bin_op_type can represent everything that final_type can. This
  // is ensuring that the pattern is matching the case when we upcast, perform
  // the op, and then downcast.
  if (!primitive_util::CastPreservesValues(final_type, bin_op_type)) {
    return absl::OkStatus();
  }

  // Change the type of the binary op to the smaller type.
  HloComputation* computation = convert_instruction->parent();
  HloInstruction* new_bin_op =
      computation->AddInstruction(bin_op_instr->CloneWithNewOperands(
          ShapeUtil::ChangeElementType(bin_op_instr->shape(), final_type),
          {arg_1, arg_2}));
  TF_RETURN_IF_ERROR(ReplaceInstruction(final_convert_instr, new_bin_op));
  return absl::OkStatus();
}

static HloInstruction* BuildTupleConstant(HloComputation* computation,
                                          const LiteralSlice& literal,
                                          AlgebraicSimplifier* simplifier) {
  if (literal.shape().IsTuple()) {
    std::vector<HloInstruction*> elems;
    elems.reserve(ShapeUtil::TupleElementCount(literal.shape()));
    for (int i = 0; i < ShapeUtil::TupleElementCount(literal.shape()); ++i) {
      elems.push_back(BuildTupleConstant(
          computation, LiteralSlice(literal, {i}), simplifier));
    }
    return computation->AddInstruction(HloInstruction::CreateTuple(elems));
  } else {
    return computation->AddInstruction(
        simplifier->CreateConstantWithLayoutUpdated(literal.Clone()));
  }
}

absl::Status AlgebraicSimplifierVisitor::HandleConstant(
    HloInstruction* constant) {
  // Tuple constants aren't directly supported by any backend. Expand them into
  // explicit Tuple instructions.
  if (constant->shape().IsTuple()) {
    return ReplaceInstruction(
        constant,
        BuildTupleConstant(computation_, constant->literal(), simplifier_));
  }

  if (constant->shape().element_type() == TOKEN) {
    return absl::OkStatus();
  }

  // If a literal is all the same element replace it with a scalar broadcast.
  if (ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsAllFirst()) {
    Literal unique_scalar(
        LiteralUtil::GetFirstScalarLiteral(constant->literal()));
    HloInstruction* scalar = constant->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(std::move(unique_scalar)));
    HloInstruction* broadcast = constant->AddInstruction(
        HloInstruction::CreateBroadcast(constant->shape(), scalar, {}));
    return ReplaceInstruction(constant, broadcast);
  }

  // If a literal is an increasing sequence from zero, replace it with an iota.
  if (ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsR1Iota()) {
    return ReplaceWithNewInstruction(
        constant, HloInstruction::CreateIota(constant->shape(), 0));
  }

  if (std::optional<int64_t> stride = constant->literal().IsR1StridedIota()) {
    // Replace the constant with iota * stride.
    HloInstruction* stride_hlo = MakeScalarLike(constant, *stride);
    HloInstruction* iota = constant->AddInstruction(
        HloInstruction::CreateIota(constant->shape(), 0));
    return ReplaceWithNewInstruction(
        constant,
        HloInstruction::CreateBinary(constant->shape(), HloOpcode::kMultiply,
                                     iota, stride_hlo));
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleSubtract(HloInstruction* sub) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))));
  // A - A => 0
  if (options_.enable_fast_math() ||
      ShapeUtil::ElementIsIntegral(sub->shape())) {
    if (lhs == rhs) {
      return ReplaceInstruction(sub, MakeScalarLike(sub, 0));
    }
  }
  // A - 0 => A
  VLOG(10) << "trying transform [A - 0 => A]: " << sub->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(sub, lhs)) {
    return absl::OkStatus();
  }

  // Canonicalize subtraction of a constant to addition.
  VLOG(10) << "trying transform [A - Const => A + (-Const)]";
  if (Match(sub, m::Subtract(m::NonConstant(&lhs), m::Constant(&rhs))) ||
      Match(sub, m::Subtract(m::NonConstant(&lhs),
                             m::Broadcast(m::Constant(&rhs))))) {
    HloInstruction* negative_const = rhs->AddInstruction(
        HloInstruction::CreateUnary(rhs->shape(), HloOpcode::kNegate, rhs));
    if (const HloInstruction* broadcast =
            DynCast<HloBroadcastInstruction>(sub->operand(1))) {
      negative_const = rhs->AddInstruction(HloInstruction::CreateBroadcast(
          broadcast->shape(), negative_const, broadcast->dimensions()));
    }
    return ReplaceWithNewInstruction(
        sub, HloInstruction::CreateBinary(sub->shape(), HloOpcode::kAdd, lhs,
                                          negative_const));
  }

  // A - A => 0 for integer A.
  VLOG(10) << "trying transform [A - A => 0] for integer A.";
  if (lhs == rhs && ShapeUtil::ElementIsIntegral(sub->shape())) {
    return ReplaceInstruction(sub, MakeScalarLike(sub, 0));
  }

  return absl::OkStatus();
}
namespace {
template <typename T>
absl::Status InvertConstant(const HloInstruction& constant, Literal* result) {
  return result->Populate<T>([&](absl::Span<const int64_t> indices) {
    return T{1.0} / constant.literal().Get<T>(indices);
  });
}

template <typename T>
std::unique_ptr<HloInstruction> TryDivideToShift(
    HloInstruction* divide, HloComputation* computation,
    AlgebraicSimplifier* simplifier) {
  HloInstruction *a, *b, *c;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));

  if (ShapeUtil::ElementIsIntegral(divide->shape()) &&
      !Match(b, m::ConstantEffectiveScalar(&c)) &&
      !Match(b, m::Broadcast(m::ConstantEffectiveScalar(&c)))) {
    return nullptr;
  }

  if (ShapeUtil::ElementIsSigned(divide->shape())) {
    int64_t b_value = static_cast<int64_t>(c->literal().GetFirstElement<T>());
    if (b_value > 0 && absl::has_single_bit(static_cast<uint64_t>(b_value))) {
      // Handle negative dividends by negating the result of the division.
      HloInstruction* zero_like_a = MakeScalarLike(a, 0);

      Shape changed_shape = ShapeUtil::ChangeElementType(a->shape(), PRED);
      simplifier->UpdateLayout(&changed_shape);
      auto* dividend_is_negative =
          divide->AddInstruction(HloInstruction::CreateCompare(
              changed_shape, a, zero_like_a, ComparisonDirection::kLt));

      auto* negated_dividend = divide->AddInstruction(
          HloInstruction::CreateUnary(a->shape(), HloOpcode::kNegate, a));

      auto* abs_dividend = divide->AddInstruction(HloInstruction::CreateTernary(
          a->shape(), HloOpcode::kSelect, dividend_is_negative,
          negated_dividend, a));

      auto* quotient = divide->AddInstruction(HloInstruction::CreateBinary(
          divide->shape(), HloOpcode::kShiftRightLogical, abs_dividend,
          MakeScalarLike(abs_dividend, Log2Floor<uint64_t>(b_value))));

      auto* neqated_quotient =
          divide->AddInstruction(HloInstruction::CreateUnary(
              quotient->shape(), HloOpcode::kNegate, quotient));

      return HloInstruction::CreateTernary(divide->shape(), HloOpcode::kSelect,
                                           dividend_is_negative,
                                           neqated_quotient, quotient);
    }
  } else {
    uint64_t b_value = static_cast<uint64_t>(c->literal().GetFirstElement<T>());
    if (absl::has_single_bit(b_value)) {
      return HloInstruction::CreateBinary(
          divide->shape(), HloOpcode::kShiftRightLogical, a,
          MakeScalarLike(a, Log2Floor(b_value)));
    }
  }

  return nullptr;
}
}  // namespace

absl::Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  HloInstruction *a, *b, *c, *d;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (IsAll(b, 1) && ReplaceInstructionIfCompatible(divide, a)) {
    return absl::OkStatus();
  }

  // A / B => A >> log2(B) if B is a power of 2.
  if (std::unique_ptr<HloInstruction> shift =
          primitive_util::PrimitiveTypeSwitch<std::unique_ptr<HloInstruction>>(
              [&](auto kType) -> std::unique_ptr<HloInstruction> {
                if constexpr (primitive_util::IsIntegralType(kType)) {
                  using NativeT = primitive_util::NativeTypeOf<kType>;
                  return TryDivideToShift<NativeT>(divide, computation_,
                                                   simplifier_);
                }
                return nullptr;
              },
              divide->shape().element_type())) {
    return ReplaceWithNewInstruction(divide, std::move(shift));
  }

  Shape* shape;
  // exp(A)/exp(B) => exp(A-B)
  if (Match(divide, m::Divide(m::Exp(m::Op(&a)), m::Exp(m::Op(&b)))
                        .WithShape(m::Shape(&shape)))) {
    VLOG(10) << "transform [exp(A)/exp(B) => exp(A-B)]: " << divide->ToString();
    HloInstruction* subtract = divide->AddInstruction(
        HloInstruction::CreateBinary(*shape, HloOpcode::kSubtract, a, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateUnary(*shape, HloOpcode::kExp, subtract));
  }

  // A/exp(B) => A*exp(-B)
  if (Match(divide, m::Divide(m::Op(&a), m::Exp(m::Op(&b))))) {
    VLOG(10) << "transform [A/exp(B) => A*exp(-B)]: " << divide->ToString();
    HloInstruction* negate = divide->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kNegate, b));
    HloInstruction* new_exp = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kExp, negate));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(divide->shape(),
                                             HloOpcode::kMultiply, a, new_exp));
  }

  // A/pow(B,C) => A*pow(B,-C)
  if (Match(divide, m::Divide(m::Op(&a), m::Power(m::Op(&b), m::Op(&c))))) {
    VLOG(10) << "transform [A/pow(B,C) => A*pow(B,-C)]: " << divide->ToString();
    // The output shape of the created negate operator should be the same as the
    // input.
    const Shape& negate_shape = c->shape();
    HloInstruction* negate = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateUnary(negate_shape, HloOpcode::kNegate, c));
    // And the power operator should retain the output shape of the old one.
    const Shape& new_power_shape = b->shape();
    HloInstruction* new_power =
        divide->mutable_operand(1)->AddInstruction(HloInstruction::CreateBinary(
            new_power_shape, HloOpcode::kPower, b, negate));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(
                    divide->shape(), HloOpcode::kMultiply, a, new_power));
  }

  // A/sqrt(B) => A*rsqrt(B).
  if (Match(divide, m::Divide(m::Op(&a), m::Sqrt(m::Op(&b)).WithOneUse()))) {
    auto* rsqrt = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kRsqrt, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(rsqrt->shape(),
                                             HloOpcode::kMultiply, a, rsqrt));
  }

  // A/rsqrt(B) => A*sqrt(B).
  if (Match(divide, m::Divide(m::Op(&a), m::Rsqrt(m::Op(&b)).WithOneUse()))) {
    auto* sqrt = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kSqrt, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(sqrt->shape(),
                                             HloOpcode::kMultiply, a, sqrt));
  }

  // Simplifying integral division would produce unexpected results.
  if (ShapeUtil::ElementIsIntegral(divide->shape())) {
    return absl::OkStatus();
  }

  // A / Const => A * (1 / Const)
  //
  // (Backends can do this transformation, but generally only if the constant is
  // a scalar.)
  if (Match(divide, m::Divide(m::NonConstant(&a), m::Op(&b))) &&
      (Match(b, m::Constant(&c)) || Match(b, m::Broadcast(m::Constant(&c))))) {
    Shape result_shape = c->literal().shape();
    Literal new_literal(result_shape);
    return primitive_util::PrimitiveTypeSwitch<absl::Status>(
        [&](auto primitive_type_constant) -> absl::Status {
          if constexpr (primitive_util::IsFloatingPointType(
                            primitive_type_constant) ||
                        primitive_util::IsComplexType(
                            primitive_type_constant)) {
            using NativeT = NativeTypeOf<primitive_type_constant>;
            TF_RETURN_IF_ERROR(InvertConstant<NativeT>(*c, &new_literal));

            auto inverse =
                c->AddInstruction(simplifier_->CreateConstantWithLayoutUpdated(
                    new_literal.Clone()));
            if (b != c) {
              inverse = b->AddInstruction(HloInstruction::CreateBroadcast(
                  b->shape(), inverse, b->dimensions()));
            }
            TF_ASSIGN_OR_RETURN(
                auto new_divide,
                MakeBinaryHlo(HloOpcode::kMultiply, a, inverse));
            return ReplaceInstruction(divide, new_divide);
          }
          return absl::OkStatus();
        },
        result_shape.element_type());
  }

  // (A / B) / (C / D)  =>  (A / B)*(D / C) => (A * D) / (B * C)
  if (Match(divide, m::Divide(m::Divide(m::Op(&a), m::Op(&b)),
                              m::Divide(m::Op(&c), m::Op(&d))))) {
    TF_ASSIGN_OR_RETURN(auto a_times_d,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, d));
    TF_ASSIGN_OR_RETURN(auto b_times_c,
                        MakeBinaryHlo(HloOpcode::kMultiply, b, c));
    TF_ASSIGN_OR_RETURN(auto new_divide, MakeBinaryHlo(HloOpcode::kDivide,
                                                       a_times_d, b_times_c));

    return ReplaceInstruction(divide, new_divide);
  }

  // (A / B) / C => A / (B * C)
  if (Match(divide, m::Divide(m::Divide(m::Op(&a), m::Op(&b)), m::Op(&c)))) {
    TF_ASSIGN_OR_RETURN(auto b_times_c,
                        MakeBinaryHlo(HloOpcode::kMultiply, b, c));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kDivide, a, b_times_c));
    return ReplaceInstruction(divide, new_divide);
  }

  // A / (B / C) => (A*C) / B
  if (Match(divide, m::Divide(m::Op(&a), m::Divide(m::Op(&b), m::Op(&c))))) {
    TF_ASSIGN_OR_RETURN(auto a_times_c,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, c));
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kDivide, a_times_c, b));
    return ReplaceInstruction(divide, new_divide);
  }

  // If X is a convert from pred, then
  // X / broadcast(Y) => broadcast(1/Y) * X
  if (Match(divide,
            m::Divide(
                m::Convert(&a,
                           m::Op().WithShape(m::Shape().WithElementType(PRED))),
                m::Broadcast(m::Op(&b).WithShape(m::Shape().IsScalar()))))) {
    TF_ASSIGN_OR_RETURN(
        auto recip, MakeBinaryHlo(HloOpcode::kDivide, MakeScalarLike(b, 1), b));
    auto recip_bcast = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateBroadcast(divide->shape(), recip, {}));
    TF_ASSIGN_OR_RETURN(auto mul,
                        MakeBinaryHlo(HloOpcode::kMultiply, recip_bcast, a));
    return ReplaceInstruction(divide, mul);
  }

  return absl::OkStatus();
}

absl::StatusOr<bool>
AlgebraicSimplifierVisitor::RemoveDegenerateDimensionFromDot(
    HloDotInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  int64_t num_degenerate_lhs_dims = 0;
  std::vector<int64_t> lhs_dimension_map(lhs_shape.dimensions_size(), -1);
  for (int64_t i = 0; i < lhs_shape.dimensions_size(); ++i) {
    if (lhs_shape.dimensions(i) == 1) {
      ++num_degenerate_lhs_dims;
    } else {
      lhs_dimension_map[i] = i - num_degenerate_lhs_dims;
    }
  }

  const Shape& rhs_shape = dot->operand(1)->shape();
  int64_t num_degenerate_rhs_dims = 0;
  std::vector<int64_t> rhs_dimension_map(rhs_shape.dimensions_size(), -1);
  for (int64_t i = 0; i < rhs_shape.dimensions_size(); ++i) {
    if (rhs_shape.dimensions(i) == 1) {
      ++num_degenerate_rhs_dims;
    } else {
      rhs_dimension_map[i] = i - num_degenerate_rhs_dims;
    }
  }
  if (num_degenerate_lhs_dims == 0 && num_degenerate_rhs_dims == 0) {
    return false;
  }
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  DotDimensionNumbers new_dnums;
  for (int64_t dim : dnums.lhs_batch_dimensions()) {
    int64_t new_dim = lhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_lhs_batch_dimensions(new_dim);
    }
  }
  for (int64_t dim : dnums.lhs_contracting_dimensions()) {
    int64_t new_dim = lhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_lhs_contracting_dimensions(new_dim);
    }
  }

  for (int64_t dim : dnums.rhs_batch_dimensions()) {
    int64_t new_dim = rhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_rhs_batch_dimensions(new_dim);
    }
  }
  for (int64_t dim : dnums.rhs_contracting_dimensions()) {
    int64_t new_dim = rhs_dimension_map[dim];
    if (new_dim != -1) {
      new_dnums.add_rhs_contracting_dimensions(new_dim);
    }
  }

  std::vector<SparsityDescriptor> sparsity(dot->sparsity().begin(),
                                           dot->sparsity().end());
  std::vector<HloInstruction*> sparse_meta(sparsity.size());
  for (int i = 0; i < sparsity.size(); ++i) {
    // Update sparse dimension number in the descriptor.
    SparsityDescriptor& descriptor = sparsity[i];
    const std::vector<int64_t>& dimension_map =
        descriptor.index() == 0 ? lhs_dimension_map : rhs_dimension_map;
    CHECK_LT(static_cast<size_t>(descriptor.dimension()), dimension_map.size());
    int preceding_dims_elided = absl::c_count_if(
        absl::MakeSpan(dimension_map.data(), descriptor.dimension()),
        [&](int64_t dim) { return dim == -1; });
    descriptor.set_dimension(descriptor.dimension() - preceding_dims_elided);

    // Reshape sparsity metadata operand, if affected.
    HloInstruction* meta =
        dot->mutable_operand(HloDotInstruction::kOperands + i);
    Shape new_shape = ShapeUtil::DropDegenerateDimensions(meta->shape());
    if (!ShapeUtil::Equal(new_shape, meta->shape())) {
      TF_ASSIGN_OR_RETURN(meta, MakeReshapeHlo(new_shape, meta));
    }
    sparse_meta[i] = meta;
  }

  HloInstruction* new_lhs =
      num_degenerate_lhs_dims > 0
          ? dot->parent()->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::DropDegenerateDimensions(lhs_shape),
                dot->mutable_operand(0)))
          : dot->mutable_operand(0);
  HloInstruction* new_rhs =
      num_degenerate_rhs_dims > 0
          ? dot->parent()->AddInstruction(HloInstruction::CreateReshape(
                ShapeUtil::DropDegenerateDimensions(rhs_shape),
                dot->mutable_operand(1)))
          : dot->mutable_operand(1);
  TF_ASSIGN_OR_RETURN(
      auto new_dot,
      MakeDotHlo(new_lhs, new_rhs, new_dnums, dot->precision_config(),
                 dot->shape().element_type(), sparsity, sparse_meta));
  dot->SetupDerivedInstruction(new_dot);

  if (ShapeUtil::Compatible(dot->shape(), new_dot->shape())) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot, new_dot));
  } else {
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), new_dot)));
  }
  return true;
}

// transpose(broadcast(x)) -> broadcast(x), if the transpose leaves the relative
// order of the dimensions of `x` unchanged.
//
// To understand the permutations logic here, consider a simple case.
//
//  bcast = f32[1,2,3,4] broadcast(f32[2,4] x), dimensions={1,3}
//  trans = f32[2,3,1,4] transpose(f32[1,2,3,4] bcast), dimensions={1,2,0,3}
//
// We want to transform this into
//
//  bcast' = f32[2,3,1,4] broadcast(f32[2,4] x), dimensions={0,3}
absl::Status AlgebraicSimplifierVisitor::SimplifyTransposeOfBroadcast(
    HloInstruction* transpose, absl::Span<const int64_t> dimensions) {
  HloInstruction* broadcast = transpose->mutable_operand(0);
  if (broadcast->opcode() != HloOpcode::kBroadcast ||
      !absl::c_is_sorted(broadcast->dimensions())) {
    return absl::OkStatus();
  }

  // The algorithm to compute bcast'.dimensions() is:
  //
  //  * Let p' be the inverse of trans.dimensions(); in the example, {2,0,1,3}.
  //  * bcast'.dimensions() is [p'[dim] for dim in bcast.dimensions()].  In the
  //    example, p'[1] = 0, meaning that broadcast dim 1 (size 2) ends up at
  //    index 0 after the transpose.
  //
  // We also need to check that bcast'.dimensions() is "sorted the same" as
  // bcast.dimensions() -- otherwise, we're simply moving the transpose into the
  // broadcast op.  For now we cowardly refuse to consider broadcasts except
  // where their dimensions() are sorted, so we need only check that
  // bcast'.dimensions() is sorted.
  //
  // No one-user requirement on the transpose because having two different
  // broadcasts of x should be cheap -- certainly cheaper than using the
  // fully-materialized broadcasted+transposed value.

  auto inv_perm = InversePermutation(dimensions);
  absl::InlinedVector<int64_t, 8> new_bcast_dims;
  for (int64_t dim : broadcast->dimensions()) {
    new_bcast_dims.push_back(inv_perm[dim]);
  }
  if (!absl::c_is_sorted(new_bcast_dims)) {
    return absl::OkStatus();
  }
  // We don't want to create broadcasts that create implicit transposes. Check
  // whether the relative order of the layout of the broadcasted dimensions is
  // the same as the broadcast operand layout.
  if (options_.is_layout_sensitive()) {
    std::vector<int64_t> perm1(new_bcast_dims.size());
    absl::c_iota(perm1, 0);
    std::vector<int64_t> perm2 = perm1;
    Layout operand_layout = broadcast->mutable_operand(0)->shape().layout();
    absl::c_sort(perm1, [&](int a, int b) {
      return operand_layout.minor_to_major(a) <
             operand_layout.minor_to_major(b);
    });
    Layout transpose_layout = transpose->shape().layout();
    // Extract the part of the layout that corresponds to the broadcasted
    // dimensions.
    std::vector<int64_t> extracted_layout;
    extracted_layout.reserve(new_bcast_dims.size());
    for (int64_t dim : transpose_layout.minor_to_major()) {
      if (absl::c_binary_search(new_bcast_dims, dim)) {
        extracted_layout.push_back(dim);
      }
    }
    absl::c_sort(perm2, [&](int a, int b) {
      return extracted_layout[a] < extracted_layout[b];
    });
    if (perm1 != perm2) {
      return absl::OkStatus();
    }
  }
  return ReplaceInstruction(
      transpose, MakeBroadcastHlo(broadcast->mutable_operand(0), new_bcast_dims,
                                  transpose->shape()));
}

absl::StatusOr<bool>
AlgebraicSimplifierVisitor::RemoveTransposesFromDotOperands(
    HloDotInstruction* dot) {
  const int64_t rank = dot->shape().dimensions_size();
  const auto& dnums = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);

  // lhs and rhs must apply the same permutation.
  if (lhs->opcode() != HloOpcode::kTranspose ||
      rhs->opcode() != HloOpcode::kTranspose ||
      lhs->dimensions() != rhs->dimensions()) {
    return false;
  }
  absl::Span<const int64_t> permutation = lhs->dimensions();

  // Dot must be "somewhat canonical": batch dimensions at the beginning, one
  // contracting dimension, and one non-contracting dim.
  if (absl::MakeSpan(dnums.lhs_batch_dimensions()) !=
          absl::MakeSpan(dnums.rhs_batch_dimensions()) ||
      dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_contracting_dimensions(0) != rank - 1 ||
      dnums.rhs_contracting_dimensions(0) != rank - 2 ||
      rank != dnums.lhs_batch_dimensions_size() + 2) {
    return false;
  }
  // Skip sparse dots.
  if (Cast<HloDotInstruction>(dot)->sparse_operands()) {
    return false;
  }

  // The last two elements of the permutation must be either [rank-2, rank-1]
  // (i.e. no permutation) or [rank-1, rank-2].  Otherwise, this means that
  // we're permuting batch dimensions with the non-batch dimensions, which isn't
  // allowed.
  //
  // If the permutation ends with [rank - 1, rank - 2] then we're going to flip
  // the order of dot operands to dot(b,a).  Otherwise it stays dot(a,b).
  bool reorder_operands;
  if (permutation.subspan(rank - 2) ==
      std::array<int64_t, 2>{rank - 2, rank - 1}) {
    reorder_operands = false;
  } else if (permutation.subspan(rank - 2) ==
             std::array<int64_t, 2>{rank - 1, rank - 2}) {
    reorder_operands = true;
  } else {
    return false;
  }

  HloInstruction* new_lhs =
      reorder_operands ? rhs->mutable_operand(0) : lhs->mutable_operand(0);
  HloInstruction* new_rhs =
      reorder_operands ? lhs->mutable_operand(0) : rhs->mutable_operand(0);
  auto new_dot = dot->AddInstruction(HloInstruction::CreateDot(
      ShapeUtil::PermuteDimensions(permutation, dot->shape()), new_lhs, new_rhs,
      dnums,
      reorder_operands
          ? SwapOperandsInDotPrecisionConfig(dot->precision_config())
          : dot->precision_config()));
  dot->SetupDerivedInstruction(new_dot);
  TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      dot,
      HloInstruction::CreateTranspose(dot->shape(), new_dot, permutation)));
  return true;
}

namespace {
// Whether an HloInstruction is either a kParameter directly, or produced from
// a parameter through a series of trivial operations.
bool IsParameterLike(const HloInstruction* inst) {
  while (true) {
    if (inst->opcode() == HloOpcode::kParameter) {
      return true;
    }
    if (inst->operand_count() != 1) {
      return false;
    }
    inst = inst->operand(0);
  }
}
}  // namespace

absl::StatusOr<bool> AlgebraicSimplifierVisitor::MoveDotParamToRhs(
    HloDotInstruction* dot) {
  const bool swap_operands =
      IsParameterLike(dot->operand(0)) && !IsParameterLike(dot->operand(1));
  if (!swap_operands || !options_.enable_move_dot_param_to_rhs()) {
    return false;
  }
  // If operand0 is parameter-like, but operand1 is not, swap the two operands.
  DotDimensionNumbers dot_dims = dot->dot_dimension_numbers();
  std::swap(*dot_dims.mutable_lhs_contracting_dimensions(),
            *dot_dims.mutable_rhs_contracting_dimensions());
  std::swap(*dot_dims.mutable_lhs_batch_dimensions(),
            *dot_dims.mutable_rhs_batch_dimensions());
  PrecisionConfig precision_config = dot->precision_config();
  std::swap(precision_config.mutable_operand_precision()->at(0),
            precision_config.mutable_operand_precision()->at(1));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_inst,
      MakeDotHlo(dot->mutable_operand(1), dot->mutable_operand(0), dot_dims,
                 precision_config, dot->shape().element_type()));
  HloDotInstruction* new_dot = Cast<HloDotInstruction>(new_inst);
  VLOG(10) << "Replacing: " << dot->ToString() << " with "
           << new_dot->ToString();
  std::vector<int64_t> permutation;
  const int64_t num_batch_dims = dot_dims.lhs_batch_dimensions_size();
  const int64_t lhs_non_contracting_batch =
      new_dot->operand(0)->shape().dimensions_size() - num_batch_dims -
      dot_dims.lhs_contracting_dimensions_size();
  const int64_t rhs_non_contracting_batch =
      new_dot->operand(1)->shape().dimensions_size() - num_batch_dims -
      dot_dims.rhs_contracting_dimensions_size();
  for (int i = 0; i != num_batch_dims; ++i) {
    permutation.push_back(i);
  }
  for (int i = 0; i != rhs_non_contracting_batch; ++i) {
    permutation.push_back(num_batch_dims + lhs_non_contracting_batch + i);
  }
  for (int i = 0; i != lhs_non_contracting_batch; ++i) {
    permutation.push_back(num_batch_dims + i);
  }
  TF_ASSIGN_OR_RETURN(HloInstruction * new_transpose,
                      MakeTransposeHlo(new_dot, permutation));
  dot->SetupDerivedInstruction(new_dot);
  dot->SetupDerivedInstruction(new_transpose);
  TF_RETURN_IF_ERROR(ReplaceInstruction(dot, new_transpose));
  return true;
}

absl::StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::NormalizeDotOperandToBatchMajorAndContractingMinor(
    HloInstruction* dot_operand, absl::Span<const int64_t> batch_dimensions,
    absl::Span<const int64_t> contracting_dimensions) {
  std::vector<int64_t> transpose_dimensions(batch_dimensions.begin(),
                                            batch_dimensions.end());
  for (int64_t i = 0; i < dot_operand->shape().dimensions_size(); ++i) {
    if (!(absl::c_linear_search(batch_dimensions, i) ||
          absl::c_linear_search(contracting_dimensions, i))) {
      transpose_dimensions.push_back(i);
    }
  }
  transpose_dimensions.insert(transpose_dimensions.end(),
                              contracting_dimensions.begin(),
                              contracting_dimensions.end());
  if (absl::c_is_sorted(transpose_dimensions)) {
    return dot_operand;
  }
  return MakeTransposeHlo(dot_operand, transpose_dimensions);
}

HloInstruction* AlgebraicSimplifierVisitor::AddReduce(
    HloInstruction* hlo, absl::Span<const int64_t> dims, PrimitiveType type) {
  HloInstruction* zero =
      computation_->AddInstruction(simplifier_->CreateConstantWithLayoutUpdated(
          LiteralUtil::Zero(hlo->shape().element_type()).Clone()));
  HloComputation* AddReduce_computation = GetOrCreateScalarAddComputation(type);
  Shape shape = ShapeUtil::DeleteDimensions(dims, hlo->shape());
  simplifier_->UpdateLayout(&shape);
  return computation_->AddInstruction(HloInstruction::CreateReduce(
      shape, hlo, zero, dims, AddReduce_computation));
}

absl::StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfConcat(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      Cast<HloDotInstruction>(dot)->sparse_operands() ||
      dot->shape().dimensions_size() != 2) {  // dot output 2D
    return nullptr;
  }

  const int64_t lhs_contracting_dim = dnums.lhs_contracting_dimensions(0);
  const int64_t rhs_contracting_dim = dnums.rhs_contracting_dimensions(0);
  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));

  TF_ASSIGN_OR_RETURN(
      HloInstruction * optimized_lhs_concat,
      OptimizeDotOfConcatHelper(dot, lhs, lhs_contracting_dim, rhs,
                                rhs_contracting_dim, /*swapped=*/false));
  if (optimized_lhs_concat) {
    return optimized_lhs_concat;
  }

  return OptimizeDotOfConcatHelper(dot, rhs, rhs_contracting_dim, lhs,
                                   lhs_contracting_dim, /*swapped=*/true);
}

absl::StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::OptimizeDotOfConcatHelper(
    HloInstruction* dot, HloInstruction* lhs, int64_t lhs_contracting_dim,
    HloInstruction* rhs, int64_t rhs_contracting_dim, bool swapped) {
  bool can_optimize = lhs->opcode() == HloOpcode::kConcatenate &&
                      lhs->concatenate_dimension() == lhs_contracting_dim &&
                      rhs->opcode() == HloOpcode::kConstant;
  if (!can_optimize) {
    return nullptr;
  }

  // We're replacing this:
  //
  //   +-----+-----+-----+      +-------------------+
  //   |     |     |     |      |                   |
  //   |     |     |     |      |        R_0        |
  //   |     |     |     |      |                   |
  //   |     |     |     |      +-------------------+
  //   |     |     |     |      |                   |
  //   | L_0 | L_1 | L_2 |   *  |        R_1        |
  //   |     |     |     |      |                   |
  //   |     |     |     |      +-------------------+
  //   |     |     |     |      |                   |
  //   |     |     |     |      |        R_2        |
  //   |     |     |     |      |                   |
  //   +-----+-----+-----+      +-------------------+
  //
  // with this:
  //
  // [Sum over i]
  //
  //   +-----+     +-------------------+
  //   |     |     |                   |
  //   |     |  *  |        R_i        |
  //   |     |     |                   |
  //   |     |     +-------------------+
  //   |     |
  //   | L_i |
  //   |     |
  //   |     |
  //   |     |
  //   |     |
  //   |     |
  //   +-----+
  //
  // where the LHS is a concatenate operation (so we can "split" the LHS tensor
  // for free) and the RHS is a constant tensor (and thus can be split at
  // compile time).  In the future, we may also want to do this when both the
  // LHS and the RHS are concatenate operations that line up along the dimension
  // being contracted over.
  //
  // We should be able to generalize this transform to work on a non-constant
  // RHS when/if we have in-place slices or support input-fusing slices into
  // Dots.

  // Dimension numbers for the new dot instructions we'll create (L_i * R_i in
  // the diagram above).
  DotDimensionNumbers new_dot_dnums;
  new_dot_dnums.add_lhs_contracting_dimensions(swapped ? rhs_contracting_dim
                                                       : lhs_contracting_dim);
  new_dot_dnums.add_rhs_contracting_dimensions(swapped ? lhs_contracting_dim
                                                       : rhs_contracting_dim);

  // Here we use the MKN notation, where the contracted dimension has K
  // elements and the two non-contracted dimensions have M and N elements.
  HloInstruction* add_result = nullptr;
  int64_t rhs_contracting_dim_offset = 0;
  int64_t n = rhs->shape().dimensions(1 - rhs_contracting_dim);
  for (HloInstruction* concat_op : lhs->operands()) {
    int64_t sub_k = concat_op->shape().dimensions(lhs_contracting_dim);
    Shape rhs_slice_shape(rhs->shape());
    rhs_slice_shape.set_dimensions(rhs_contracting_dim, sub_k);
    simplifier_->UpdateLayout(&rhs_slice_shape);

    std::array<int64_t, 2> start_indices;
    start_indices[rhs_contracting_dim] = rhs_contracting_dim_offset;
    start_indices[1 - rhs_contracting_dim] = 0;

    std::array<int64_t, 2> limit_indices;
    limit_indices[rhs_contracting_dim] = rhs_contracting_dim_offset + sub_k;
    limit_indices[1 - rhs_contracting_dim] = n;

    HloInstruction* rhs_slice = rhs->AddInstruction(HloInstruction::CreateSlice(
        rhs_slice_shape, rhs, /*start_indices=*/start_indices,
        /*limit_indices=*/limit_indices, /*strides=*/{1, 1}));

    // "non-canonical" contraction dimensions (that contracts dimension 1 of the
    // LHS with dimension 0 of the RHS).  But for now we keep the same
    // contraction dimensions as the incoming dot operation to ensure the new
    // dot operations can be lowered.
    HloInstruction *new_dot_lhs, *new_dot_rhs;
    if (swapped) {
      new_dot_lhs = rhs_slice;
      new_dot_rhs = concat_op;
    } else {
      new_dot_lhs = concat_op;
      new_dot_rhs = rhs_slice;
    }

    auto* new_dot = dot->AddInstruction(
        HloInstruction::CreateDot(dot->shape(), new_dot_lhs, new_dot_rhs,
                                  new_dot_dnums, dot->precision_config()));
    dot->SetupDerivedInstruction(new_dot);

    if (add_result) {
      add_result = dot->AddInstruction(HloInstruction::CreateBinary(
          dot->shape(), HloOpcode::kAdd, add_result, new_dot));
    } else {
      add_result = new_dot;
    }

    rhs_contracting_dim_offset += sub_k;
  }

  return add_result;
}

absl::StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfGather(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      Cast<HloDotInstruction>(dot)->sparse_operands() ||
      dot->shape().dimensions_size() != 2) {  // dot output 2D
    VLOG(10) << "DotOfGather: Can only optimize 2D, non-batch dot operations.";
    return nullptr;
  }

  // Optimize either dot(DS(ctA), ctB)) or dot(ctB, DS(ctA)).
  // Currently a Gather is a DynamicSlice.
  auto is_dynamic_slice_constant_combination =
      [](HloInstruction* a, HloInstruction* b, int a_contracting_dimension) {
        // First operand is a DynamicSlice(Constant).
        if (a->opcode() != HloOpcode::kDynamicSlice) {
          return false;
        }
        auto* dynamic_slice_op = a->operand(0);
        if (dynamic_slice_op->opcode() != HloOpcode::kConstant) {
          return false;
        }
        // Second operand is a Constant.
        if (b->opcode() != HloOpcode::kConstant) {
          return false;
        }
        // The DynamicSlice output is a vector.
        const Shape& dynamic_slice_shape = a->shape();
        if (dynamic_slice_shape.dimensions(1 - a_contracting_dimension) != 1) {
          return false;
        }
        // Constant size is the same before and after slice in the contracting
        // dimension, otherwise we either must precompute for all possible slice
        // indices or dot is invalid.
        const Shape& dynamic_slice_op_shape = dynamic_slice_op->shape();
        if (dynamic_slice_op_shape.dimensions(a_contracting_dimension) !=
            dynamic_slice_shape.dimensions(a_contracting_dimension)) {
          return false;
        }
        return true;
      };

  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);
  int lhs_contracting_dimension = dnums.lhs_contracting_dimensions(0);
  int rhs_contracting_dimension = dnums.rhs_contracting_dimensions(0);

  if (!is_dynamic_slice_constant_combination(
          lhs, rhs, /*a_contracting_dimension=*/lhs_contracting_dimension) &&
      !is_dynamic_slice_constant_combination(
          rhs, lhs, /*a_contracting_dimension=*/rhs_contracting_dimension)) {
    VLOG(10) << "DotOfGather: Can only optimize dot(DS(ctA), ctB)) or "
                "dot(ctB, DS(ctA)), where the two constants have equal "
                "contracting dimensions.";
    return nullptr;
  }

  // LHS is DynamicSlice:
  // input: dot(DS(ctA), ctB))
  // where DS(ctA) = DS({M x K}, {start, 0}, {1, K}) and ctB = {K x N}.
  // => input dimensions: dot({1 x K}, {K x N}) => {1 x N}.
  // output: DS(dot(ctA, ctB))
  // => output dimensions: DS ({M x N}, {start, 0}, {1, N}) => {1 x N}.

  // RHS is DynamicSlice:
  // input: dot(ctA, DS(ctB))
  // where ctA = {M x K} and DS(ctB) = DS({K x N}, {0, start}, {K, 1}).
  // => input dimensions: dot({M x K}, {K x 1}) => {M x 1}.
  // output: DS(dot(ctA, ctB))
  // => output dimensions: DS ({M x N}, {0, start}, {M, 1}) => {M x 1}.

  bool lhs_is_dynamic_slice = lhs->opcode() == HloOpcode::kDynamicSlice;
  HloDynamicSliceInstruction* dynamic_slice =
      lhs_is_dynamic_slice ? Cast<HloDynamicSliceInstruction>(lhs)
                           : Cast<HloDynamicSliceInstruction>(rhs);

  // ctA:
  HloInstruction* left_operand =
      lhs_is_dynamic_slice ? lhs->mutable_operand(0) : lhs;
  // ctB:
  HloInstruction* right_operand =
      lhs_is_dynamic_slice ? rhs : rhs->mutable_operand(0);
  // Build ctA x ctB.
  const int m = left_operand->shape().dimensions(1 - lhs_contracting_dimension);
  const int n =
      right_operand->shape().dimensions(1 - rhs_contracting_dimension);
  auto memoized_shape =
      ShapeUtil::MakeShape(dot->shape().element_type(), {m, n});
  simplifier_->UpdateLayout(&memoized_shape);
  auto* memoized_inst = dot->AddInstruction(
      HloInstruction::CreateDot(memoized_shape, left_operand, right_operand,
                                dnums, dot->precision_config()));
  dot->SetupDerivedInstruction(memoized_inst);
  // Get pair {start, 0} or {0, start}.
  // Position of start:
  int index_of_non_zero_start = lhs_is_dynamic_slice
                                    ? 1 - lhs_contracting_dimension
                                    : 1 - rhs_contracting_dimension;
  // Position of zero:
  int index_of_zero_start = 1 - index_of_non_zero_start;

  // Slice out start and 0 components and reorder if necessary.
  auto indices_type = dynamic_slice->operand(1)->shape().element_type();
  Shape s_shape = ShapeUtil::MakeShape(indices_type, {1});
  simplifier_->UpdateLayout(&s_shape);
  Shape d_shape = ShapeUtil::MakeShape(indices_type, {2});
  simplifier_->UpdateLayout(&d_shape);
  HloInstruction* non_zero_start =
      dynamic_slice->mutable_operand(1 + index_of_non_zero_start);
  HloInstruction* zero_start =
      dynamic_slice->mutable_operand(1 + index_of_zero_start);
  std::vector<HloInstruction*> new_start_indices;
  if (lhs_is_dynamic_slice) {
    new_start_indices = {non_zero_start, zero_start};
  } else {
    new_start_indices = {zero_start, non_zero_start};
  }

  // Build DynamicSlice(ctA x ctB).
  const int new_slice_m = lhs_is_dynamic_slice ? 1 : m;
  const int new_slice_n = lhs_is_dynamic_slice ? n : 1;
  auto* memoized_lookup =
      dot->AddInstruction(HloInstruction::CreateDynamicSlice(
          dot->shape(), memoized_inst, new_start_indices,
          {new_slice_m, new_slice_n}));

  return memoized_lookup;
}

// This function tries to transform
//   dot(reshape(transpose(A)), Const) to
//   dot(reshape(A), reshape(transpose(reshape(Const)))),
// so that the reshape and transpose on the Const side can be constant folded.
//
// The basic idea is that since the accumulation in the dot operation is
// associative, so as long as we permute the elements of the contracting
// dimensions on both sides of the dot in the same way, the result of the
// dot is not affected.
absl::StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::OptimizeDotOfReorderContractingDims(
    HloInstruction* dot) {
  // This transformation assumes layout is not assigned yet.
  if (options_.is_layout_sensitive()) {
    return nullptr;
  }

  // Canonicalize dot(<constant>, rhs) to dot(rhs, <constant>) to make the
  // remainder of this function easier.
  auto dnums = dot->dot_dimension_numbers();
  auto lhs_contracting_dims = dnums.lhs_contracting_dimensions();
  auto rhs_contracting_dims = dnums.rhs_contracting_dimensions();
  auto* lhs = dot->mutable_operand(0);
  auto* rhs = dot->mutable_operand(1);
  if (dot->operand(0)->IsConstant()) {
    std::swap(lhs, rhs);
    std::swap(lhs_contracting_dims, rhs_contracting_dims);
  }

  // Require single contracting dim to make the implementation easier to
  // track contracting dims.
  if (dnums.lhs_contracting_dimensions_size() != 1) {
    return nullptr;
  }

  // Pattern match Dot(reshape(transpose(input), constant))
  HloInstruction* reshape;
  HloInstruction* transpose;
  HloInstruction* input;
  HloInstruction* constant;
  if (!Match(lhs,
             m::Reshape(&reshape, m::Transpose(&transpose, m::Op(&input)))) ||
      !Match(rhs, m::Constant(&constant)) ||
      Cast<HloDotInstruction>(dot)->sparse_operands()) {
    return nullptr;
  }

  // Check that reshape squishes some dims into one dim and that this one
  // dim is the dot's lhs contracting dim. The size of unmodified_dims should
  // be N - 1, where N is the rank of the reshape output. This means that the
  // reshape squishes some dims into one dim. lhs contracting dim should not
  // be in unmodified_dims. This means that the squishing target dim is the
  // lhs contracting dim.
  auto unmodified_dims = ShapeUtil::DimensionsUnmodifiedByReshape(
      reshape->operand(0)->shape(), reshape->shape());
  CHECK_EQ(lhs_contracting_dims.size(), 1);
  if ((unmodified_dims.size() != reshape->shape().dimensions_size() - 1) ||
      absl::c_any_of(unmodified_dims,
                     [&](const std::pair<int64_t, int64_t>& p) {
                       return p.second == lhs_contracting_dims[0];
                     })) {
    return nullptr;
  }

  // Virtually pull the reshape into the dot so the dot operates on the
  // transpose, with "unsquished" lhs contracting dims.  The new contracting
  // dims are all of the dims that are modified by the reshape -- that is, every
  // dimension that's not in `unmodified_dims[i].first`.
  //
  // (We don't need to actually create a new dot instruction. We can just keep
  // track of lhs and lhs_contracting_dims.)
  absl::flat_hash_set<int64_t> unmodified_transpose_dims;
  for (const auto& pair : unmodified_dims) {
    unmodified_transpose_dims.insert(pair.first);
  }
  lhs_contracting_dims.Clear();
  for (int64_t i = 0; i < transpose->shape().dimensions_size(); ++i) {
    if (!unmodified_transpose_dims.contains(i)) {
      lhs_contracting_dims.Add(i);
    }
  }
  // We require the "unsquished" lhs contracting dims to be consecutive.
  auto is_iota = [](absl::Span<const int64_t> dims) {
    return absl::c_adjacent_find(dims, [](const int64_t a, const int64_t b) {
             return (b != a + 1);
           }) == dims.end();
  };
  if (!is_iota(lhs_contracting_dims)) {
    return nullptr;
  }
  lhs = lhs->mutable_operand(0);

  // Check that the transpose only permutes the contracting dims.
  const auto& transpose_dims = transpose->dimensions();
  for (int64_t i = 0; i < transpose_dims.size(); ++i) {
    if (transpose_dims[i] != i &&
        !absl::c_linear_search(lhs_contracting_dims, i)) {
      return nullptr;
    }
  }
  // Virtually pull the transpose into the dot. Now the dot is equivalent to
  // a new dot with "permuted" lhs contracting dims.
  std::vector<int64_t> permutation;
  permutation.reserve(lhs_contracting_dims.size());
  for (auto dim : lhs_contracting_dims) {
    permutation.push_back(transpose_dims[dim] - lhs_contracting_dims[0]);
  }
  CHECK(IsPermutation(permutation));
  auto new_lhs_contracting_dims =
      ComposePermutations(lhs_contracting_dims, permutation);
  lhs_contracting_dims.Clear();
  for (auto dim : new_lhs_contracting_dims) {
    lhs_contracting_dims.Add(dim);
  }
  lhs = lhs->mutable_operand(0);

  // All checks are passed at this point.
  //
  // Transform lhs. Remove the transpose and reshape by sorting the lhs
  // contracting dims and squishing them into a single one. We don't actually
  // squish the lhs_contracting_dims here because we still need the unsquished
  // contracting dims to invert reshape and transpose.
  absl::c_sort(lhs_contracting_dims);
  lhs =
      dot->AddInstruction(HloInstruction::CreateReshape(reshape->shape(), lhs));

  // Transform rhs. Say the input HLO is:
  //
  //   t0 = f32[2, 2, 3] parameter(0)
  //   t1 = f32[2, 3, 2] transpose(t0) dimensions={0, 2, 1}
  //   t2 = f32[2, 6] reshape(t1)
  //   t3 = f32[6, 2] constant(...)
  //   dot = f32[2, 2] dot(t2, t3) lhs_contracting_dims={1},
  //                               rhs_contracting_dims={0}
  //
  // At this point in the function, we have decided that the second and third
  // dims of t0 can be switched to remove the transpose, and we have
  // "virtually decomposed" the input HLO to:
  //
  //   t0 = f32[2, 2, 3] parameter(0)
  //   t2' = f32[2, 6] reshape(t0)
  //   t3' = f32[6, 2] ops-to-be-filled ...
  //   dot = f32[2, 2] dot(t2', t3') lhs_contracting_dims={1},
  //                                 rhs_contracting_dims={0}
  //
  // The rest of this function is to fill in the ops of t3'. To do this, we
  // unsquish the contracting dimensions in t3 and then apply the inverse of
  // the transpose from t1.

  // Invert reshape.
  CHECK_EQ(rhs_contracting_dims.size(), 1);
  std::vector<int64_t> rhs_unsquished_shape_dims =
      SpanToVector(constant->shape().dimensions());
  auto it = rhs_unsquished_shape_dims.erase(rhs_unsquished_shape_dims.begin() +
                                            rhs_contracting_dims[0]);
  for (auto dim : lhs_contracting_dims) {
    it = rhs_unsquished_shape_dims.insert(it,
                                          transpose->shape().dimensions(dim));
    ++it;
  }
  HloInstruction* rhs_reshape =
      dot->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(constant->shape().element_type(),
                               rhs_unsquished_shape_dims),
          constant));
  rhs = rhs_reshape;

  // Rhs reshape "unsquishes" the single rhs contracting dim into multiple dims.
  rhs_contracting_dims.Resize(lhs_contracting_dims.size(), 0);
  absl::c_iota(rhs_contracting_dims, rhs_contracting_dims[0]);

  // Invert transpose. First compute the shape.
  std::vector<int64_t> rhs_transpose_shape_dims =
      SpanToVector(rhs_reshape->shape().dimensions());
  it = rhs_transpose_shape_dims.erase(
      rhs_transpose_shape_dims.begin() + rhs_contracting_dims[0],
      rhs_transpose_shape_dims.begin() + rhs_contracting_dims[0] +
          rhs_contracting_dims.size());
  for (auto dim : lhs_contracting_dims) {
    it = rhs_transpose_shape_dims.insert(it, input->shape().dimensions(dim));
    ++it;
  }
  // Then compute the transpose dims.
  std::vector<int64_t> rhs_transpose_dims(
      rhs_reshape->shape().dimensions_size());
  absl::c_iota(rhs_transpose_dims, 0);
  it = rhs_transpose_dims.erase(
      rhs_transpose_dims.begin() + rhs_contracting_dims[0],
      rhs_transpose_dims.begin() + rhs_contracting_dims[0] +
          rhs_contracting_dims.size());
  auto inverse_lhs_transpose_dims = InversePermutation(transpose_dims);
  for (auto dim : lhs_contracting_dims) {
    it = rhs_transpose_dims.insert(it, inverse_lhs_transpose_dims[dim] -
                                           lhs_contracting_dims[0] +
                                           rhs_contracting_dims[0]);
    ++it;
  }
  HloInstruction* rhs_transpose =
      dot->AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(constant->shape().element_type(),
                               rhs_transpose_shape_dims),
          rhs_reshape, rhs_transpose_dims));
  rhs = rhs_transpose;

  // Squish the multiple rhs contracting dims into a single one.
  rhs = dot->AddInstruction(
      HloInstruction::CreateReshape(constant->shape(), rhs));

  // If we virtually swapped lhs and rhs, we need to swap it back before
  // creating new dot.
  if (dot->operand(0)->IsConstant()) {
    std::swap(lhs, rhs);
  }

  HloInstruction* new_dot = dot->AddInstruction(HloInstruction::CreateDot(
      dot->shape(), lhs, rhs, dnums, dot->precision_config()));
  return new_dot;
}

// If appropriate, reorder operation on dot operand to the mirror operation on
// the other dot operand
absl::StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::AssociativeReorderDotOperator(
    HloDotInstruction* dot) {
  if (dot->sparse_operands()) {
    return nullptr;
  }

  DotDimensionNumbers dnums = dot->dot_dimension_numbers();
  HloInstruction* lhs = dot->mutable_operand(0);
  HloInstruction* rhs = dot->mutable_operand(1);

  HloInstruction *reorder_from, *reorder_to;
  bool lhs_to_rhs = false;
  bool rhs_to_lhs = false;

  // Check whether we should try to reorder either operand
  if (lhs->opcode() == HloOpcode::kBroadcast ||
      lhs->opcode() == HloOpcode::kPad ||
      lhs->opcode() == HloOpcode::kReverse) {
    reorder_from = lhs;
    reorder_to = rhs;
    lhs_to_rhs = true;
  } else if (rhs->opcode() == HloOpcode::kBroadcast ||
             rhs->opcode() == HloOpcode::kPad ||
             rhs->opcode() == HloOpcode::kReverse) {
    reorder_from = rhs;
    reorder_to = lhs;
    rhs_to_lhs = true;
  }

  if (lhs_to_rhs || rhs_to_lhs) {
    HloInstruction* reordered = reorder_to;
    HloInstruction* unreordered = reorder_from->mutable_operand(0);
    DotDimensionNumbers new_dnums = dnums;
    HloOpcode opcode = reorder_from->opcode();
    double threshold_multiplier = 1.0;
    bool make_hlo = false;

    // Construct maps between corresponding dot contracting dimensions
    std::vector<int64_t> contracting_dim_map_forward(
        reorder_from->shape().dimensions_size(), -1);
    std::vector<int64_t> contracting_dim_map_backward(
        reorder_to->shape().dimensions_size(), -1);
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
      auto from_index = lhs_to_rhs ? dnums.lhs_contracting_dimensions()[i]
                                   : dnums.rhs_contracting_dimensions()[i];
      auto to_index = lhs_to_rhs ? dnums.rhs_contracting_dimensions()[i]
                                 : dnums.lhs_contracting_dimensions()[i];
      contracting_dim_map_forward[from_index] = to_index;
      contracting_dim_map_backward[to_index] = from_index;
    }

    // Perform computations specific to each opcode
    if (opcode == HloOpcode::kReverse) {
      // We should check that the reordering will be beneficial before we
      // create the new Hlo to avoid OOM issues
      if (ShapeUtil::ElementsIn(reorder_from->shape()) /
              static_cast<double>(ShapeUtil::ElementsIn(reorder_to->shape())) <
          options_.associative_reordering_threshold()) {
        return nullptr;
      }

      // Reverses of dot contracting dimensions can be reordered to
      // reverses of the corresponding contracting dimensions in the other dot
      // operand
      DimensionVector reordered_dims, unreordered_dims;
      for (auto dim : reorder_from->dimensions()) {
        if (contracting_dim_map_forward[dim] != -1) {
          reordered_dims.push_back(contracting_dim_map_forward[dim]);
          make_hlo = true;
        } else {
          unreordered_dims.push_back(dim);
        }
      }

      // Create Hlo for reordered reverse and unreordered reverse
      if (!make_hlo) {
        return nullptr;
      }
      if (!reordered_dims.empty()) {
        TF_ASSIGN_OR_RETURN(reordered,
                            MakeReverseHlo(reorder_to, reordered_dims));
      }
      if (!unreordered_dims.empty()) {
        // Want to use a greater threshold if reordering means increasing the
        // number of Hlos
        threshold_multiplier = 2.0;
        TF_ASSIGN_OR_RETURN(
            unreordered,
            MakeReverseHlo(reorder_from->mutable_operand(0), unreordered_dims));
      }
    } else if (opcode == HloOpcode::kPad) {
      // Padding of dot contracting dimensions can be reordered to slices of
      // the corresponding contracting dimensions in the other dot operand
      DimensionVector start_indices, limit_indices, strides;
      PaddingConfig new_padding_config = reorder_from->padding_config();

      // Compute start_indices, limit_indices, and strides for slicing from
      // the padding dimensions
      for (int64_t to_dim = 0; to_dim < reorder_to->shape().dimensions_size();
           to_dim++) {
        int64_t start_index = 0;
        int64_t limit_index = reorder_to->shape().dimensions(to_dim);
        int64_t stride = 1;
        if (contracting_dim_map_backward[to_dim] != -1) {
          // If it's a contracting dimension, we want to slice it according to
          // the corresponding padding in the other operand
          const int64_t from_dim = contracting_dim_map_backward[to_dim];
          auto padding_dimension =
              reorder_from->padding_config().dimensions(from_dim);

          // Edge padding can be negative which acts as a slice. If this is
          // the case, we don't want to reorder
          if (padding_dimension.edge_padding_low() > 0 ||
              padding_dimension.edge_padding_high() > 0 ||
              padding_dimension.interior_padding() > 0) {
            make_hlo = true;
            start_index += padding_dimension.edge_padding_low();
            limit_index -= padding_dimension.edge_padding_high();
            stride += padding_dimension.interior_padding();

            // We then remove this dimension from the padding
            new_padding_config.mutable_dimensions(from_dim)
                ->set_edge_padding_low(0);
            new_padding_config.mutable_dimensions(from_dim)
                ->set_edge_padding_high(0);
            new_padding_config.mutable_dimensions(from_dim)
                ->set_interior_padding(0);
          }
        }
        start_indices.push_back(start_index);
        limit_indices.push_back(limit_index);
        strides.push_back(stride);
      }

      // Create Hlo for slice
      if (!make_hlo) {
        return nullptr;
      }
      TF_ASSIGN_OR_RETURN(reordered, MakeSliceHlo(reorder_to, start_indices,
                                                  limit_indices, strides));

      // Check if we still need a padding instruction, and create Hlo if so
      for (auto& dim : new_padding_config.dimensions()) {
        if (dim.edge_padding_low() != 0 || dim.edge_padding_high() != 0) {
          // Want to use a greater threshold if reordering means increasing
          // the number of Hlos
          threshold_multiplier = 2.0;
          TF_ASSIGN_OR_RETURN(
              unreordered,
              MakePadHlo(reorder_from->mutable_operand(0),
                         reorder_from->mutable_operand(1), new_padding_config));
          break;
        }
      }
    } else if (opcode == HloOpcode::kBroadcast) {
      // Broadcasts of dot contracting dimensions can be reordered to reduces
      // of the corresponding contracting dimensions in the other dot operand
      DimensionVector reduce_dims;
      const int64_t pre_broadcast_rank =
          reorder_from->mutable_operand(0)->shape().dimensions_size();
      int64_t post_broadcast_rank = reorder_from->shape().dimensions_size();
      Shape new_broadcast_shape = reorder_from->shape();

      // Construct map from broadcasted shape to its original shape. Broadcast
      // dimensions are mapped to -1 since they were not present
      std::vector<int64_t> map_broadcast_dims(post_broadcast_rank, -1);
      for (int64_t i = 0; i < pre_broadcast_rank; i++) {
        map_broadcast_dims[reorder_from->dimensions(i)] = i;
      }

      // Use maps to create new dot dnums and vector of reduce dims
      new_dnums.clear_lhs_contracting_dimensions();
      new_dnums.clear_rhs_contracting_dimensions();
      int64_t deleted_dims = 0;
      for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); i++) {
        auto from_index = lhs_to_rhs ? dnums.lhs_contracting_dimensions()[i]
                                     : dnums.rhs_contracting_dimensions()[i];
        auto to_index = lhs_to_rhs ? dnums.rhs_contracting_dimensions()[i]
                                   : dnums.lhs_contracting_dimensions()[i];
        if (map_broadcast_dims[from_index] == -1) {
          // This is a contracting broadcast dimension
          reduce_dims.push_back(to_index);
          new_broadcast_shape.DeleteDimension(from_index - deleted_dims);
          deleted_dims++;
          make_hlo = true;
        } else {
          // This is a contracting nonbroadcast dimension
          if (lhs_to_rhs) {
            new_dnums.add_lhs_contracting_dimensions(
                map_broadcast_dims[from_index]);
            new_dnums.add_rhs_contracting_dimensions(to_index);
          } else {
            new_dnums.add_lhs_contracting_dimensions(to_index);
            new_dnums.add_rhs_contracting_dimensions(
                map_broadcast_dims[from_index]);
          }
        }
      }

      if (!make_hlo) {
        return nullptr;
      }
      // Create constant 0 to use as the init_value for reduce
      HloInstruction* zero = dot->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(reorder_from->shape().element_type())));

      // Create Hlo for unreordered broadcast and reordered reduce
      if (reorder_from->mutable_operand(0)->shape() != new_broadcast_shape) {
        // Want to use a greater threshold if reordering means increasing
        // the number of Hlos
        threshold_multiplier = 2.0;
        unreordered =
            MakeBroadcastHlo(reorder_from->mutable_operand(0),
                             reorder_from->dimensions(), new_broadcast_shape);
      }
      TF_ASSIGN_OR_RETURN(
          reordered,
          MakeReduceHlo(reorder_to, zero, reduce_dims, HloOpcode::kAdd));
    }

    if (!make_hlo) {
      return nullptr;
    }

    // Create Hlo for new dot operands, depending on the direction in which
    // we are reordering
    HloInstruction *new_lhs, *new_rhs;
    if (lhs_to_rhs) {
      new_lhs = unreordered;
      new_rhs = reordered;
    } else {
      new_lhs = reordered;
      new_rhs = unreordered;
    }

    // Create Hlo for new dot
    HloInstruction* new_dot;
    TF_ASSIGN_OR_RETURN(new_dot, MakeDotHlo(new_lhs, new_rhs, new_dnums,
                                            dot->precision_config(),
                                            dot->shape().element_type()));

    // Do cost analysis to determine whether we should reorder. Reverse uses
    // the ratio of the two shapes a heuristic, while the others use the
    // number of dot flops
    const int64_t old_flops =
        HloCostAnalysis::GetDotFlops(lhs->shape(), dot->shape(), dnums);
    const int64_t new_flops = HloCostAnalysis::GetDotFlops(
        new_lhs->shape(), new_dot->shape(), new_dnums);
    bool reorder =
        old_flops / static_cast<double>(new_flops) >
        threshold_multiplier * options_.associative_reordering_threshold();

    if (reorder || opcode == HloOpcode::kReverse) {
      return new_dot;
    }
  }
  return nullptr;
}

absl::Status
AlgebraicSimplifierVisitor::RewriteAsMultiplyDotWithZeroLhsContractingDim(
    HloInstruction* dot, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dnums) {
  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      NormalizeDotOperandToBatchMajorAndContractingMinor(
                          lhs, dnums.lhs_batch_dimensions(),
                          dnums.lhs_contracting_dimensions()));
  if (!ShapeUtil::SameElementType(dot->shape(), new_lhs->shape())) {
    new_lhs = MakeConvertToHlo(new_lhs, dot->shape().element_type());
  }
  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      NormalizeDotOperandToBatchMajorAndContractingMinor(
                          rhs, dnums.rhs_batch_dimensions(),
                          dnums.rhs_contracting_dimensions()));
  if (!ShapeUtil::SameElementType(dot->shape(), new_rhs->shape())) {
    new_rhs = MakeConvertToHlo(new_rhs, dot->shape().element_type());
  }
  if (dot->shape().dimensions_size() != lhs->shape().dimensions_size()) {
    std::vector<int64_t> lhs_broadcast_dims(lhs->shape().dimensions_size());
    absl::c_iota(lhs_broadcast_dims, 0);
    new_lhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
        dot->shape(), new_lhs, lhs_broadcast_dims));
  }
  if (dot->shape().dimensions_size() != rhs->shape().dimensions_size()) {
    std::vector<int64_t> rhs_broadcast_dims(dnums.lhs_batch_dimensions_size());
    absl::c_iota(rhs_broadcast_dims, 0);
    for (int64_t i = lhs->shape().dimensions_size();
         i < dot->shape().dimensions_size(); ++i) {
      rhs_broadcast_dims.push_back(i);
    }
    new_rhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
        dot->shape(), new_rhs, rhs_broadcast_dims));
  }
  auto new_instruction = HloInstruction::CreateBinary(
      dot->shape(), HloOpcode::kMultiply, new_lhs, new_rhs);
  dot->SetupDerivedInstruction(new_lhs);
  dot->SetupDerivedInstruction(new_rhs);
  dot->SetupDerivedInstruction(new_instruction.get());
  return ReplaceWithNewInstruction(dot, std::move(new_instruction));
}

absl::StatusOr<AlgebraicSimplifierVisitor::RewriteResult>
AlgebraicSimplifierVisitor::AssociativeReorderNestedDot(HloDotInstruction* dot,
                                                        HloInstruction* lhs,
                                                        HloInstruction* rhs) {
  HloInstruction *inner, *outer;
  HloInstruction *new_inner, *new_outer;

  // We only proceed if one of this dot's operands is itself a dot
  bool outer_lhs_dot = false;
  bool outer_rhs_dot = false;

  if (lhs->opcode() == HloOpcode::kDot) {
    outer = dot;
    inner = lhs;
    outer_lhs_dot = true;
  } else if (rhs->opcode() == HloOpcode::kDot) {
    outer = dot;
    inner = rhs;
    outer_rhs_dot = true;
  }

  if ((!outer_lhs_dot && !outer_rhs_dot)) {
    return RewriteResult::kNoRewrite;
  }

  if (Cast<HloDotInstruction>(inner)->sparse_operands()) {
    return RewriteResult::kNoRewrite;
  }

  DotDimensionNumbers ab_dnums, ac_dnums, bc_dnums;

  // We will now use inner and outer to build up ab_dnums, ac_dnums, and
  // bc_dnums. One of these three comes for free from inner
  if (outer_lhs_dot) {
    ab_dnums = inner->dot_dimension_numbers();
  } else if (outer_rhs_dot) {
    bc_dnums = inner->dot_dimension_numbers();
  }

  // For the other two, it's more complicated. First, we construct maps from
  // the dimensions of inner to the dimensions of inner's operands
  std::vector<int64_t> map_inner_lhs, map_inner_rhs;
  std::tie(map_inner_lhs, map_inner_rhs) = ConstructFromDotMaps(
      inner, inner->operand(0)->shape(), inner->operand(1)->shape());
  DotDimensionNumbers outer_dnums = outer->dot_dimension_numbers();

  // We now iterate through the batch dimensions of outer, and recover
  // the batch dimensions shared between each operand of inner and the
  // other operand of outer
  for (int64_t i = 0; i < outer_dnums.lhs_batch_dimensions_size(); ++i) {
    // First we retrieve inner_index and other_index depending on which side
    // of outer that inner is on
    int64_t inner_index, other_index;
    if (outer_lhs_dot) {
      inner_index = outer_dnums.lhs_batch_dimensions(i);
      other_index = outer_dnums.rhs_batch_dimensions(i);
    } else {
      inner_index = outer_dnums.rhs_batch_dimensions(i);
      other_index = outer_dnums.lhs_batch_dimensions(i);
    }

    auto add_batch_dims = [](DotDimensionNumbers& dnums, int64_t lhs_ix,
                             int64_t rhs_ix) {
      dnums.add_lhs_batch_dimensions(lhs_ix);
      dnums.add_rhs_batch_dimensions(rhs_ix);
    };

    for (auto& map : {map_inner_lhs, map_inner_rhs}) {
      int64_t mapped_index = map[inner_index];
      if (mapped_index != -1) {
        // Whether the mapped value is the lhs or rhs of the new dnums
        // depends on whether inner is the lhs or rhs operand of outer. The
        // dnums itself depends on this and also on which map we are
        // iterating through
        if (outer_lhs_dot) {
          add_batch_dims(map == map_inner_lhs ? ac_dnums : bc_dnums,
                         mapped_index, other_index);
        } else {
          add_batch_dims(map == map_inner_lhs ? ab_dnums : ac_dnums,
                         other_index, mapped_index);
        }
      }
    }
  }

  // We now do the same thing for the contracting dimensions of outer
  for (int64_t i = 0; i < outer_dnums.lhs_contracting_dimensions_size(); ++i) {
    // First we retrieve inner_index and other_index depending on which side
    // of outer that inner is on
    int64_t inner_index, other_index;
    if (outer_lhs_dot) {
      inner_index = outer_dnums.lhs_contracting_dimensions(i);
      other_index = outer_dnums.rhs_contracting_dimensions(i);
    } else {
      inner_index = outer_dnums.rhs_contracting_dimensions(i);
      other_index = outer_dnums.lhs_contracting_dimensions(i);
    }

    // Once we have the inner_index, we determine whether this index
    // corresponds to a dimension coming from the lhs or rhs of inner
    bool from_inner_lhs = map_inner_lhs[inner_index] != -1;
    bool from_inner_rhs = map_inner_rhs[inner_index] != -1;

    // If a dimension of inner is the result of batching and it is
    // contracted in outer, we stop trying to reorder
    if (from_inner_lhs && from_inner_rhs) {
      return RewriteResult::kStopRewrites;
    }

    // The map we use depends on which operand of inner this dim comes from
    std::vector<int64_t> map;
    if (from_inner_lhs) {
      map = map_inner_lhs;
    } else {
      map = map_inner_rhs;
    }

    // Whether the mapped value goes into the lhs or rhs of the new dnums
    // depends on whether inner was the lhs or rhs operand of outer
    int64_t lhs_index, rhs_index;
    if (outer_lhs_dot) {
      lhs_index = map[inner_index];
      rhs_index = other_index;
    } else {
      lhs_index = other_index;
      rhs_index = map[inner_index];
    }

    // Finally, we have to determine which dnums to add to
    DotDimensionNumbers* dnums;
    if (outer_lhs_dot) {
      if (from_inner_lhs) {
        dnums = &ac_dnums;
      } else {
        dnums = &bc_dnums;
      }
    } else {
      if (from_inner_lhs) {
        dnums = &ab_dnums;
      } else {
        dnums = &ac_dnums;
      }
    }

    // Add the contracting dimensions
    dnums->add_lhs_contracting_dimensions(lhs_index);
    dnums->add_rhs_contracting_dimensions(rhs_index);
  }

  // ab_dnums, ac_dnums, and bc_dnums are now complete. We can now use these
  // dnums to construct the dnums for the new_inner and new_outer.
  HloInstruction *new_inner_lhs, *new_inner_rhs;
  DotDimensionNumbers new_inner_dnums;
  if (outer_lhs_dot) {
    new_inner_lhs = inner->mutable_operand(1);
    new_inner_rhs = outer->mutable_operand(1);
    new_inner_dnums = bc_dnums;
  } else {
    new_inner_lhs = outer->mutable_operand(0);
    new_inner_rhs = inner->mutable_operand(0);
    new_inner_dnums = ab_dnums;
  }

  // For dnums for new_outer, we will need some additional maps
  std::vector<int64_t> map_lhs_new_inner, map_rhs_new_inner;
  std::tie(map_lhs_new_inner, map_rhs_new_inner) = ConstructToDotMaps(
      new_inner_dnums, new_inner_lhs->shape(), new_inner_rhs->shape());
  DotDimensionNumbers new_outer_dnums;

  // To build up new_outer dnums, we need to combine two "pairs". If the
  // inner dot was originally on lhs, these pairs are ab and ac. If the
  // inner dot was originally on the rhs, these pairs ac and bc
  std::vector<DotDimensionNumbers> dnums_to_reorder;
  if (outer_lhs_dot) {
    dnums_to_reorder.push_back(ab_dnums);
    dnums_to_reorder.push_back(ac_dnums);
  } else {
    dnums_to_reorder.push_back(ac_dnums);
    dnums_to_reorder.push_back(bc_dnums);
  }

  // We now iterate through the batch and contracting dimensions of each
  // pair, using the previously constructed maps to add to new_outer dnums
  for (int pair = 0; pair < 2; ++pair) {
    DotDimensionNumbers dnums = dnums_to_reorder[pair];
    std::vector<int64_t> map =
        (pair % 2) == 0 ? map_lhs_new_inner : map_rhs_new_inner;

    for (int64_t i = 0; i < dnums.lhs_batch_dimensions_size(); ++i) {
      int64_t new_inner_index, other_index;
      if (outer_lhs_dot) {
        new_inner_index = dnums.rhs_batch_dimensions(i);
        other_index = dnums.lhs_batch_dimensions(i);
      } else {
        new_inner_index = dnums.lhs_batch_dimensions(i);
        other_index = dnums.rhs_batch_dimensions(i);
      }

      int64_t lhs_index, rhs_index;
      if (outer_lhs_dot) {
        lhs_index = other_index;
        rhs_index = map[new_inner_index];
      } else {
        lhs_index = map[new_inner_index];
        rhs_index = other_index;
      }

      if (!absl::c_linear_search(new_outer_dnums.lhs_batch_dimensions(),
                                 lhs_index)) {
        new_outer_dnums.add_lhs_batch_dimensions(lhs_index);
        new_outer_dnums.add_rhs_batch_dimensions(rhs_index);
      }
    }
    for (int64_t i = 0; i < dnums.lhs_contracting_dimensions_size(); ++i) {
      int64_t new_inner_index, other_index;
      if (outer_lhs_dot) {
        new_inner_index = dnums.rhs_contracting_dimensions(i);
        other_index = dnums.lhs_contracting_dimensions(i);
      } else {
        new_inner_index = dnums.lhs_contracting_dimensions(i);
        other_index = dnums.rhs_contracting_dimensions(i);
      }

      int64_t lhs_index, rhs_index;
      if (outer_lhs_dot) {
        lhs_index = other_index;
        rhs_index = map[new_inner_index];
      } else {
        lhs_index = map[new_inner_index];
        rhs_index = other_index;
      }

      new_outer_dnums.add_lhs_contracting_dimensions(lhs_index);
      new_outer_dnums.add_rhs_contracting_dimensions(rhs_index);
    }
  }

  // Get Shape for new_inner
  TF_ASSIGN_OR_RETURN(
      Shape new_inner_shape,
      ShapeInference::InferDotOpShape(new_inner_lhs->shape(),
                                      new_inner_rhs->shape(), new_inner_dnums,
                                      new_inner_lhs->shape().element_type()));
  Shape new_outer_lhs_shape =
      outer_lhs_dot ? inner->operand(0)->shape() : new_inner_shape;

  // Use HloCostAnalysis to compute flops for both the original and
  // reordered instructions, and reorder if doing so decreases flops by a
  // factor of the reordering threshold.
  const int64_t old_flops =
      HloCostAnalysis::GetDotFlops(inner->operand(0)->shape(), inner->shape(),
                                   inner->dot_dimension_numbers()) +
      HloCostAnalysis::GetDotFlops(outer->operand(0)->shape(), outer->shape(),
                                   outer_dnums);
  const int64_t new_flops =
      HloCostAnalysis::GetDotFlops(new_inner_lhs->shape(), new_inner_shape,
                                   new_inner_dnums) +
      HloCostAnalysis::GetDotFlops(new_outer_lhs_shape, outer->shape(),
                                   new_outer_dnums);

  if (old_flops / static_cast<double>(new_flops) >
      options_.associative_reordering_threshold()) {
    // We can now make the Hlo for new_inner and new_outer
    TF_ASSIGN_OR_RETURN(
        new_inner,
        MakeDotHlo(new_inner_lhs, new_inner_rhs, new_inner_dnums,
                   dot->precision_config(), dot->shape().element_type()));
    HloInstruction *new_outer_lhs, *new_outer_rhs;
    if (outer_lhs_dot) {
      new_outer_lhs = inner->mutable_operand(0);
      new_outer_rhs = new_inner;
    } else {
      new_outer_lhs = new_inner;
      new_outer_rhs = inner->mutable_operand(1);
    }
    TF_ASSIGN_OR_RETURN(
        new_outer,
        MakeDotHlo(new_outer_lhs, new_outer_rhs, new_outer_dnums,
                   dot->precision_config(), dot->shape().element_type()));

    // Depending on the batch dimensions of the original instruction,
    // reordering may permute the dimensions of the shape. To correct for
    // this, we build a map from old_outer dimensions to new_outer
    // dimensions and use it to transpose new_outer.
    DimensionVector permutation(new_outer->shape().dimensions_size());

    // Construct additional maps to make the permutation
    std::vector<int64_t> map_outer_lhs, map_outer_rhs;
    std::tie(map_outer_lhs, map_outer_rhs) = ConstructFromDotMaps(
        outer, outer->operand(0)->shape(), outer->operand(1)->shape());

    std::vector<int64_t> map_outer_inner, map_outer_other;
    map_outer_inner = outer_lhs_dot ? map_outer_lhs : map_outer_rhs;
    map_outer_other = outer_lhs_dot ? map_outer_rhs : map_outer_lhs;

    std::vector<int64_t> map_inner_new_other;
    map_inner_new_other = outer_lhs_dot ? map_inner_lhs : map_inner_rhs;

    std::vector<int64_t> map_other_new_inner;
    map_other_new_inner = outer_lhs_dot ? map_rhs_new_inner : map_lhs_new_inner;

    std::vector<int64_t> map_lhs_new_outer, map_rhs_new_outer;
    std::tie(map_lhs_new_outer, map_rhs_new_outer) =
        ConstructToDotMaps(new_outer_dnums, new_outer->operand(0)->shape(),
                           new_outer->operand(1)->shape());

    std::vector<int64_t> map_new_inner_new_outer, map_new_other_new_outer;
    map_new_inner_new_outer =
        outer_lhs_dot ? map_rhs_new_outer : map_lhs_new_outer;
    map_new_other_new_outer =
        outer_lhs_dot ? map_lhs_new_outer : map_rhs_new_outer;

    // Create permutation to do the transpose
    bool add_transpose = false;
    for (int64_t i = 0; i < outer->shape().dimensions_size(); i++) {
      int64_t new_outer_index;
      if (map_outer_other[i] == -1) {
        int64_t inner_index = map_outer_inner[i];
        if (map_inner_new_other[inner_index] == -1) {
          int64_t new_inner_index;
          if (outer_lhs_dot) {
            new_inner_index = map_lhs_new_inner[map_inner_rhs[inner_index]];
          } else {
            new_inner_index = map_rhs_new_inner[map_inner_lhs[inner_index]];
          }
          new_outer_index = map_new_inner_new_outer[new_inner_index];
        } else {
          int64_t new_other_index = map_inner_new_other[inner_index];
          new_outer_index = map_new_other_new_outer[new_other_index];
        }
      } else {
        // Dimension i in outer comes from other
        int64_t other_index = map_outer_other[i];
        new_outer_index =
            map_new_inner_new_outer[map_other_new_inner[other_index]];
      }
      permutation[i] = new_outer_index;
      if (i != new_outer_index) {
        add_transpose = true;
      }
    }

    if (add_transpose) {
      HloInstruction* transposed_new_outer;
      TF_ASSIGN_OR_RETURN(transposed_new_outer,
                          MakeTransposeHlo(new_outer, permutation));
      VLOG(10) << "Reordering with associativity and transpose";
      TF_RETURN_IF_ERROR(ReplaceInstruction(dot, transposed_new_outer));
    } else {
      VLOG(10) << "Reordering with associativity";
      TF_RETURN_IF_ERROR(ReplaceInstruction(dot, new_outer));
    }
    return RewriteResult::kRewritten;
  }
  return RewriteResult::kNoRewrite;
}

absl::Status AlgebraicSimplifierVisitor::RewriteBatchPlusContractingAsReduce(
    HloDotInstruction* dot, HloInstruction* lhs, HloInstruction* rhs,
    const DotDimensionNumbers& dnums) {
  TF_ASSIGN_OR_RETURN(HloInstruction * new_lhs,
                      NormalizeDotOperandToBatchMajorAndContractingMinor(
                          lhs, dnums.lhs_batch_dimensions(),
                          dnums.lhs_contracting_dimensions()));
  if (!ShapeUtil::SameElementType(dot->shape(), new_lhs->shape())) {
    new_lhs = MakeConvertToHlo(new_lhs, dot->shape().element_type());
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_rhs,
                      NormalizeDotOperandToBatchMajorAndContractingMinor(
                          rhs, dnums.rhs_batch_dimensions(),
                          dnums.rhs_contracting_dimensions()));
  if (!ShapeUtil::SameElementType(dot->shape(), new_rhs->shape())) {
    new_rhs = MakeConvertToHlo(new_rhs, dot->shape().element_type());
  }

  int64_t lhs_outer_dims = lhs->shape().dimensions_size() -
                           (dnums.lhs_batch_dimensions_size() +
                            dnums.lhs_contracting_dimensions_size());
  int64_t rhs_outer_dims = rhs->shape().dimensions_size() -
                           (dnums.rhs_batch_dimensions_size() +
                            dnums.rhs_contracting_dimensions_size());
  CHECK(lhs_outer_dims == 0 || rhs_outer_dims == 0);
  if (rhs_outer_dims > 0) {
    std::vector<int64_t> lhs_broadcast_dims(dnums.lhs_batch_dimensions_size());
    absl::c_iota(lhs_broadcast_dims, 0);
    lhs_broadcast_dims.resize(lhs->shape().dimensions_size());
    std::iota(lhs_broadcast_dims.begin() + dnums.lhs_batch_dimensions_size(),
              lhs_broadcast_dims.end(),
              dnums.lhs_batch_dimensions_size() + rhs_outer_dims);
    new_lhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
        new_rhs->shape(), new_lhs, lhs_broadcast_dims));
  } else if (lhs_outer_dims > 0) {
    std::vector<int64_t> rhs_broadcast_dims(dnums.rhs_batch_dimensions_size());
    absl::c_iota(rhs_broadcast_dims, 0);
    rhs_broadcast_dims.resize(rhs->shape().dimensions_size());
    std::iota(rhs_broadcast_dims.begin() + dnums.rhs_batch_dimensions_size(),
              rhs_broadcast_dims.end(),
              dnums.rhs_batch_dimensions_size() + lhs_outer_dims);
    new_rhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
        new_lhs->shape(), new_rhs, rhs_broadcast_dims));
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                      MakeMultiplyForPrecisionAlgorithm(dot, new_lhs, new_rhs));

  std::vector<int64_t> reduce_dims(dnums.lhs_contracting_dimensions_size());
  PrimitiveType dot_type =
      ShapeUtil::ElementIsFloating(dot->shape())
          ? (dot->shape().element_type() == F64 ? F64 : F32)
          : dot->shape().element_type();
  new_dot = AsType(new_dot, dot_type);
  const int64_t outer_dims = std::max(rhs_outer_dims, lhs_outer_dims);
  absl::c_iota(reduce_dims, outer_dims + dnums.lhs_batch_dimensions_size());
  new_dot = AddReduce(new_dot, reduce_dims, dot_type);
  new_dot = AsType(new_dot, dot->shape().element_type());
  return ReplaceInstruction(dot, new_dot);
}

bool AlgebraicSimplifierVisitor::SupportedDotPrecisionConfig(
    const PrecisionConfig& config, bool has_contracting_dim) {
  return config.algorithm() == PrecisionConfig::ALG_UNSET ||
         // TODO(loislo): Fixes a failure on a test with CPU backend.
         config.algorithm() == PrecisionConfig::ALG_DOT_F32_F32_F32;
}

absl::StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::MakeMultiplyForPrecisionAlgorithm(
    HloInstruction*, HloInstruction* lhs, HloInstruction* rhs) {
  return MakeBinaryHlo(HloOpcode::kMultiply, lhs, rhs);
}

absl::Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  CHECK(computation_ == dot->parent());
  HloDotInstruction* dot_cast = Cast<HloDotInstruction>(dot);
  const auto& dnums = dot->dot_dimension_numbers();

  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  if (options_.is_layout_sensitive()) {
    return absl::OkStatus();
  }

  // Replace a zero element dot with a broadcast of the constant 0.
  if (ShapeUtil::IsZeroElementArray(dot->shape()) ||
      ShapeUtil::IsZeroElementArray(lhs->shape()) ||
      ShapeUtil::IsZeroElementArray(rhs->shape())) {
    auto zero =
        dot->AddInstruction(simplifier_->CreateConstantWithLayoutUpdated(
            LiteralUtil::Zero(dot->shape().element_type())));
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBroadcast(dot->shape(), zero, {}));
  }

  // If there are no contracting dimensions, a dot can be rewritten as
  // mul(broadcast(transpose(x)),broadcast(transpose(y)))
  if (SupportedDotPrecisionConfig(dot->precision_config(),
                                  /*has_contracting_dim=*/false) &&
      options_.enable_dot_to_multiply_rewrite() &&
      dnums.lhs_contracting_dimensions_size() == 0) {
    return RewriteAsMultiplyDotWithZeroLhsContractingDim(dot, lhs, rhs, dnums);
  }

  // Reorder nested dots with associativity using flops as a heuristic
  if (options_.use_associative_reordering() && !dot_cast->sparse_operands()) {
    TF_ASSIGN_OR_RETURN(RewriteResult result,
                        AssociativeReorderNestedDot(dot_cast, lhs, rhs));
    if (result == RewriteResult::kRewritten ||
        result == RewriteResult::kStopRewrites) {
      return absl::OkStatus();
    }
  }

  if (options_.use_associative_reordering()) {
    TF_ASSIGN_OR_RETURN(HloInstruction * dot_operator_reordered,
                        AssociativeReorderDotOperator(dot_cast));
    if (dot_operator_reordered) {
      VLOG(10) << "Reordering dot operand to its mirror";
      return ReplaceInstruction(dot, dot_operator_reordered);
    }
  }

  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  if (SupportedDotPrecisionConfig(dot->precision_config(),
                                  /*has_contracting_dim=*/true) &&
      options_.enable_dot_strength_reduction() &&
      DotHasOnlyBatchAndContractingOnOneOperand(lhs->shape().dimensions_size(),
                                                rhs->shape().dimensions_size(),
                                                dnums) &&
      ShouldStrengthReduceDotToReduce(dot)) {
    return RewriteBatchPlusContractingAsReduce(dot_cast, lhs, rhs, dnums);
  }

  // Simplify dot(reshape(transpose(A)), Const) to:
  // dot(reshape(A), reshape(transpose(reshape(Const)))), so that the reshape
  // and transpose on the Const side can be constant folded.
  TF_ASSIGN_OR_RETURN(HloInstruction * dot_of_reorder_optimized,
                      OptimizeDotOfReorderContractingDims(dot));
  if (dot_of_reorder_optimized) {
    VLOG(10) << " Replaced dot " << dot->ToString()
             << " with new dot operation: "
             << dot_of_reorder_optimized->ToString();
    return ReplaceInstruction(dot, dot_of_reorder_optimized);
  }

  TF_ASSIGN_OR_RETURN(HloInstruction * dot_of_concat_optimized,
                      OptimizeDotOfConcat(dot));
  if (dot_of_concat_optimized) {
    VLOG(10) << "Replaced dot(concat(...), constant) with add(dot(..., "
                "constant)...)";
    return ReplaceInstruction(dot, dot_of_concat_optimized);
  }

  // Simplify dot(ConstA, Gather(Index, ConstB)) to:
  // Gather(Index, dot*(ConstA, ConstB)), where dot* is an appropriately
  // batched version of dot.
  TF_ASSIGN_OR_RETURN(HloInstruction * dot_of_gather_optimized,
                      OptimizeDotOfGather(dot));
  if (dot_of_gather_optimized) {
    VLOG(10) << "Replaced dot(constA, gather(i, constB)) with "
                "gather(i, dot*(constA, constB))";
    return ReplaceInstruction(dot, dot_of_gather_optimized);
  }

  TF_ASSIGN_OR_RETURN(bool removed_degenerate_dimensions,
                      RemoveDegenerateDimensionFromDot(dot_cast));
  if (removed_degenerate_dimensions) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(bool removed_transposes,
                      RemoveTransposesFromDotOperands(dot_cast));
  if (removed_transposes) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(bool moved_param_to_rhs, MoveDotParamToRhs(dot_cast));
  if (moved_param_to_rhs) {
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

namespace {
std::vector<int64_t> GetPaddedDims(const HloInstruction* pad) {
  CHECK_EQ(pad->opcode(), HloOpcode::kPad);
  std::vector<int64_t> padded_dims;
  for (int64_t i = 0; i != pad->shape().dimensions_size(); ++i) {
    if (pad->padding_config().dimensions(i).edge_padding_high() != 0 ||
        pad->padding_config().dimensions(i).edge_padding_low() != 0 ||
        pad->padding_config().dimensions(i).interior_padding() != 0) {
      padded_dims.push_back(i);
    }
  }
  return padded_dims;
}

struct GatherOfPadInfo {
  bool should_transform;
  bool has_padded_batching_dims;
};

// Returns a GatherOfPadInfo struct containing two booleans should_transform and
// has_padded_batching_dims.
//
// The returned value should_transform is true if each dim in
// padded_operand_dims is either (1) an operand-passthrough dim or (2) an
// explicit operand batching dim whose corresponding start_indices batching dim
// is padded the same way, in this case the pad instruction that produces
// start_indices should only pad the needed explicit batching dims and not pad
// any other dims.
//
// If should_transform is true, has_padded_batching_dims indicates whether case
// (2) happens, and adds such explicit operand batching dims and their
// corresponding result dims to padded_operand_dims_to_output_dims and
// output_dims_to_padded_operand_dims.
//
// Precondition: operand is produced by a pad instruction.
GatherOfPadInfo CheckPaddedDimsForGatherOfPad(
    const HloInstruction* gather,
    const std::vector<int64_t>& padded_operand_dims,
    absl::flat_hash_map<int64_t, int64_t>& padded_operand_dims_to_output_dims,
    absl::flat_hash_map<int64_t, int64_t>& output_dims_to_padded_operand_dims) {
  const GatherDimensionNumbers& dnums = gather->gather_dimension_numbers();
  absl::Span<const int64_t> operand_batching_dims =
      dnums.operand_batching_dims();
  absl::Span<const int64_t> start_indices_batching_dims =
      dnums.start_indices_batching_dims();
  const HloInstruction* operand = gather->operand(0);
  const HloInstruction* start_indices = gather->operand(1);
  auto operand_batching_dims_to_start_indices_batching_dims = [&](int64_t dim) {
    return start_indices_batching_dims[absl::c_find(operand_batching_dims,
                                                    dim) -
                                       operand_batching_dims.begin()];
  };

  int64_t num_padded_batching_dims = 0;
  struct GatherOfPadInfo skip_transform{false, false};
  for (int64_t operand_dim : padded_operand_dims) {
    if (padded_operand_dims_to_output_dims.contains(operand_dim)) {
      continue;
    }
    if (!absl::c_linear_search(operand_batching_dims, operand_dim)) {
      // An operand dim that is neither a passthrough dim nor an explicit
      // operand batching dim is padded. Can't perform the transformation.
      return skip_transform;
    }

    if (start_indices->opcode() != HloOpcode::kPad) {
      // An explicit operand batching dim is padded, but start indices is not
      // produced by a pad instruction. can't perform the transformation.
      return skip_transform;
    }

    int64_t start_indices_dim =
        operand_batching_dims_to_start_indices_batching_dims(operand_dim);
    const PaddingConfig::PaddingConfigDimension& start_indices_pad =
        start_indices->padding_config().dimensions(start_indices_dim);
    const PaddingConfig::PaddingConfigDimension& operand_pad =
        operand->padding_config().dimensions(operand_dim);
    if (!tsl::protobuf::util::MessageDifferencer::Equals(start_indices_pad,
                                                         operand_pad)) {
      return skip_transform;
    }

    num_padded_batching_dims++;
  }

  if (num_padded_batching_dims == 0) {
    return {true, false};
  }

  if (num_padded_batching_dims != GetPaddedDims(start_indices).size()) {
    // The start_indices pad instructions pads dims beyond the needed
    // explicit batching dims, we can't perform the transformation.
    return skip_transform;
  }

  // Add padded explicit operand batching dims and their corresponding result
  // dims to padded_operand_dims_to_output_dims and
  // output_dims_to_padded_operand_dims.
  const absl::flat_hash_map<int64_t, int64_t>
      start_indices_dims_to_output_dims =
          GetStartIndicesDimToOutputDimForExplicitBatchingDims(
              dnums.start_indices_batching_dims(), dnums.index_vector_dim(),
              dnums.offset_dims(), start_indices->shape().dimensions_size(),
              gather->shape().dimensions_size());
  for (int64_t operand_dim : padded_operand_dims) {
    if (!absl::c_linear_search(operand_batching_dims, operand_dim)) {
      continue;
    }

    int64_t start_indices_dim =
        operand_batching_dims_to_start_indices_batching_dims(operand_dim);
    int64_t output_dim =
        start_indices_dims_to_output_dims.at(start_indices_dim);
    padded_operand_dims_to_output_dims[operand_dim] = output_dim;
    output_dims_to_padded_operand_dims[output_dim] = operand_dim;
  }

  return {true, true};
}

}  // namespace

absl::Status AlgebraicSimplifierVisitor::HandleGather(HloInstruction* gather) {
  const Shape& operand_shape = gather->operand(0)->shape();
  if (ShapeUtil::IsZeroElementArray(operand_shape)) {
    return ReplaceInstruction(gather, MakeScalarLike(gather, 0));
  }

  // Gathering from a scalar operand is simply a broadcast of that scalar
  if (ShapeUtil::IsEffectiveScalar(operand_shape)) {
    HloInstruction* new_operand = gather->mutable_operand(0);
    if (operand_shape.dimensions_size()) {
      TF_ASSIGN_OR_RETURN(new_operand,
                          MakeReshapeHlo(ShapeUtil::MakeScalarShape(
                                             operand_shape.element_type()),
                                         new_operand));
    }
    HloInstruction* new_gather =
        MakeBroadcastHlo(new_operand, {}, gather->shape());
    return ReplaceInstruction(gather, new_gather);
  }
  // If the operand of a gather is very small, it is easier to fuse a
  // sequence of selects.
  const Shape& index_shape = gather->operand(1)->shape();
  if (operand_shape.dimensions_size() == 1 &&
      operand_shape.dimensions(0) <= options_.very_small_gather_size() &&
      gather->gather_dimension_numbers().index_vector_dim() ==
          index_shape.dimensions_size() &&
      gather->gather_dimension_numbers().collapsed_slice_dims_size() == 1) {
    const int64_t operand_elements = operand_shape.dimensions(0);
    auto get_value = [&](int64_t i) {
      auto slice = gather->AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(operand_shape.element_type(), {1}),
          gather->mutable_operand(0), {i}, {i + 1}, {1}));
      auto scalar = gather->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(operand_shape.element_type(), {}), slice));
      return gather->AddInstruction(
          HloInstruction::CreateBroadcast(gather->shape(), scalar, {}));
    };
    auto result = get_value(0);
    auto pred_shape = ShapeUtil::ChangeElementType(gather->shape(), PRED);
    simplifier_->UpdateLayout(&pred_shape);
    auto iter_shape = ShapeUtil::ChangeElementType(gather->shape(),
                                                   index_shape.element_type());
    simplifier_->UpdateLayout(&iter_shape);
    for (int64_t i = 0; i < operand_elements; ++i) {
      auto index_mask = gather->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, gather->mutable_operand(1),
          MakeScalarLike(gather->mutable_operand(1), i),
          ComparisonDirection::kGe));
      result = gather->AddInstruction(
          HloInstruction::CreateTernary(gather->shape(), HloOpcode::kSelect,
                                        index_mask, get_value(i), result));
    }
    return ReplaceInstruction(gather, result);
  }

  const hlo_sharding_util::GatherScatterDims operand_passthrough_dims =
      hlo_sharding_util::GetGatherOperandPassthroughDims(
          *gather, gather->gather_slice_sizes());

  absl::flat_hash_map<int64_t, int64_t>
      gather_operand_passthrough_operand_to_output_dims;
  absl::flat_hash_map<int64_t, int64_t>
      gather_operand_passthrough_output_to_operand_dims;
  CHECK_EQ(operand_passthrough_dims.operand_dims.size(),
           operand_passthrough_dims.output_dims.size());
  for (int64_t i = 0; i != operand_passthrough_dims.operand_dims.size(); ++i) {
    int64_t operand_dim = operand_passthrough_dims.operand_dims[i];
    int64_t output_dim = operand_passthrough_dims.output_dims[i];
    gather_operand_passthrough_operand_to_output_dims[operand_dim] = output_dim;
    gather_operand_passthrough_output_to_operand_dims[output_dim] = operand_dim;
  }
  // If the gather operand is a pad on the pass-through dimensions, then we can
  // gather the unpadded operand and then pad.
  if (HloInstruction* pad = gather->mutable_operand(0);
      pad->opcode() == HloOpcode::kPad) {
    std::vector<int64_t> padded_dims = GetPaddedDims(pad);
    GatherOfPadInfo info = CheckPaddedDimsForGatherOfPad(
        gather, padded_dims, gather_operand_passthrough_operand_to_output_dims,
        gather_operand_passthrough_output_to_operand_dims);
    // Change gather(pad(...)) to pad(gather(...)).
    if (info.should_transform) {
      Shape gather_shape = gather->shape();
      for (int64_t padded_dim : padded_dims) {
        gather_shape.mutable_dimensions()
            [gather_operand_passthrough_operand_to_output_dims[padded_dim]] =
            pad->operand(0)->shape().dimensions()[padded_dim];
      }
      auto gather_inst = Cast<HloGatherInstruction>(gather);
      std::vector<int64_t> slice_sizes;
      for (int i = 0; i != gather_inst->gather_slice_sizes().size(); ++i) {
        if (absl::c_linear_search(padded_dims, i) &&
            !absl::c_linear_search(
                gather->gather_dimension_numbers().operand_batching_dims(),
                i)) {
          slice_sizes.push_back(pad->operand(0)->shape().dimensions()[i]);
        } else {
          slice_sizes.push_back(gather_inst->gather_slice_sizes()[i]);
        }
      }
      HloInstruction* mutable_start_indices = gather->mutable_operand(1);
      HloInstruction* result =
          gather->AddInstruction(HloInstruction::CreateGather(
              gather_shape, pad->mutable_operand(0),
              info.has_padded_batching_dims
                  ? mutable_start_indices->mutable_operand(0)
                  : mutable_start_indices,
              gather_inst->gather_dimension_numbers(), slice_sizes,
              gather_inst->indices_are_sorted()));
      PaddingConfig pad_config;
      for (int64_t i = 0; i != gather->shape().dimensions_size(); ++i) {
        auto dimension = pad_config.add_dimensions();
        if (gather_operand_passthrough_output_to_operand_dims.contains(i) &&
            absl::c_linear_search(
                padded_dims,
                gather_operand_passthrough_output_to_operand_dims[i])) {
          int64_t padded_dim =
              gather_operand_passthrough_output_to_operand_dims[i];
          dimension->set_edge_padding_low(
              pad->padding_config().dimensions(padded_dim).edge_padding_low());
          dimension->set_edge_padding_high(
              pad->padding_config().dimensions(padded_dim).edge_padding_high());
          dimension->set_interior_padding(
              pad->padding_config().dimensions(padded_dim).interior_padding());
        }
      }
      result = gather->AddInstruction(HloInstruction::CreatePad(
          gather->shape(), result, pad->mutable_operand(1), pad_config));
      return ReplaceInstruction(gather, result);
    }
  }

  // If the gather operand is a reshape of a pad on the pass-through dimensions,
  // then we can gather the unpadded reshape and then pad.
  if (HloInstruction* reshape = gather->mutable_operand(0);
      reshape->opcode() == HloOpcode::kReshape &&
      ShapeUtil::ReshapeIsBitcast(reshape->operand(0)->shape(),
                                  reshape->shape())) {
    absl::flat_hash_map<int64_t, int64_t> reshape_unmodified_dims;
    for (const auto& [from_dim, to_dim] :
         ShapeUtil::DimensionsUnmodifiedByReshape(reshape->operand(0)->shape(),
                                                  reshape->shape())) {
      reshape_unmodified_dims[from_dim] = to_dim;
    }
    if (HloInstruction* pad = reshape->mutable_operand(0);
        pad->opcode() == HloOpcode::kPad) {
      bool padded_on_reshape_unmodified_dims = true;
      bool padded_on_gather_operand_passthrough_operand_dims = true;
      std::vector<int64_t> padded_dims = GetPaddedDims(pad);
      for (int64_t padded_dim : padded_dims) {
        if (!reshape_unmodified_dims.contains(padded_dim)) {
          padded_on_reshape_unmodified_dims = false;
          break;
        }
      }
      absl::flat_hash_map<int64_t, int64_t> reshape_dims_to_padded_dims;
      for (int64_t padded_dim : padded_dims) {
        reshape_dims_to_padded_dims[reshape_unmodified_dims[padded_dim]] =
            padded_dim;
      }
      for (auto& [padded_reshape_dim, _] : reshape_dims_to_padded_dims) {
        if (!gather_operand_passthrough_operand_to_output_dims.contains(
                padded_reshape_dim)) {
          padded_on_gather_operand_passthrough_operand_dims = false;
          break;
        }
      }
      // Change gather(reshape(pad(...))) to pad(gather(reshape(...))).
      if (padded_on_reshape_unmodified_dims &&
          padded_on_gather_operand_passthrough_operand_dims) {
        Shape reshape_shape = reshape->shape();
        Shape gather_shape = gather->shape();
        for (int64_t padded_dim : padded_dims) {
          int64_t to_dim = reshape_unmodified_dims[padded_dim];
          reshape_shape.mutable_dimensions()[to_dim] =
              pad->operand(0)->shape().dimensions()[padded_dim];
          gather_shape.mutable_dimensions()
              [gather_operand_passthrough_operand_to_output_dims[to_dim]] =
              pad->operand(0)->shape().dimensions()[padded_dim];
        }
        HloInstruction* result =
            gather->AddInstruction(HloInstruction::CreateReshape(
                reshape_shape, pad->mutable_operand(0)));
        auto gather_inst = Cast<HloGatherInstruction>(gather);
        std::vector<int64_t> slice_sizes;
        for (int i = 0; i != gather_inst->gather_slice_sizes().size(); ++i) {
          if (reshape_dims_to_padded_dims.contains(i)) {
            slice_sizes.push_back(
                pad->operand(0)
                    ->shape()
                    .dimensions()[reshape_dims_to_padded_dims[i]]);
          } else {
            slice_sizes.push_back(gather_inst->gather_slice_sizes()[i]);
          }
        }
        result = gather->AddInstruction(HloInstruction::CreateGather(
            gather_shape, result, gather->mutable_operand(1),
            gather_inst->gather_dimension_numbers(), slice_sizes,
            gather_inst->indices_are_sorted()));
        PaddingConfig pad_config;
        for (int64_t i = 0; i != gather->shape().dimensions_size(); ++i) {
          auto dimension = pad_config.add_dimensions();
          if (gather_operand_passthrough_output_to_operand_dims.contains(i) &&
              reshape_dims_to_padded_dims.contains(
                  gather_operand_passthrough_output_to_operand_dims[i])) {
            int64_t padded_dim = reshape_dims_to_padded_dims
                [gather_operand_passthrough_output_to_operand_dims[i]];
            dimension->set_edge_padding_low(pad->padding_config()
                                                .dimensions(padded_dim)
                                                .edge_padding_low());
            dimension->set_edge_padding_high(pad->padding_config()
                                                 .dimensions(padded_dim)
                                                 .edge_padding_high());
            dimension->set_interior_padding(pad->padding_config()
                                                .dimensions(padded_dim)
                                                .interior_padding());
          }
        }
        result = gather->AddInstruction(HloInstruction::CreatePad(
            gather->shape(), result, pad->mutable_operand(1), pad_config));
        return ReplaceInstruction(gather, result);
      }
    }
  }
  return absl::OkStatus();
}

namespace {
absl::StatusOr<std::unique_ptr<HloInstruction>> MinMaxToClamp(
    HloInstruction* clamp_lower_bound_bcast, HloInstruction* to_clamp,
    HloInstruction* clamp_upper_bound_bcast, AlgebraicSimplifier* simplifier) {
  HloInstruction* clamp_lower_bound;
  CHECK(Match(clamp_lower_bound_bcast,
              m::Broadcast(m::ConstantEffectiveScalar(&clamp_lower_bound))))
      << clamp_lower_bound_bcast->ToString();

  HloInstruction* clamp_upper_bound;
  CHECK(Match(clamp_upper_bound_bcast,
              m::Broadcast(m::ConstantEffectiveScalar(&clamp_upper_bound))))
      << clamp_upper_bound_bcast->ToString();

  const Literal& lower_bound =
      Cast<HloConstantInstruction>(clamp_lower_bound)->literal();
  const Literal& upper_bound =
      Cast<HloConstantInstruction>(clamp_upper_bound)->literal();

  TF_ASSIGN_OR_RETURN(Literal lower_bound_literal_reshaped,
                      lower_bound.Reshape({}));
  TF_ASSIGN_OR_RETURN(Literal upper_bound_literal_reshaped,
                      upper_bound.Reshape({}));
  std::unique_ptr<HloInstruction> lower_bound_instr =
      HloInstruction::CreateConstant(std::move(lower_bound_literal_reshaped));
  std::unique_ptr<HloInstruction> upper_bound_instr =
      HloInstruction::CreateConstant(std::move(upper_bound_literal_reshaped));

  Shape compare_shape =
      ShapeUtil::ChangeElementType(lower_bound_instr->shape(), PRED);
  simplifier->UpdateLayout(&compare_shape);
  std::unique_ptr<HloInstruction> cloned_instruction =
      HloInstruction::CreateCompare(compare_shape, lower_bound_instr.get(),
                                    upper_bound_instr.get(),
                                    ComparisonDirection::kLt);

  HloEvaluator evaluator;
  TF_ASSIGN_OR_RETURN(auto result,
                      evaluator.Evaluate(cloned_instruction.get()));
  if (result.IsAll(true)) {
    return HloInstruction::CreateTernary(to_clamp->shape(), HloOpcode::kClamp,
                                         clamp_lower_bound_bcast, to_clamp,
                                         clamp_upper_bound_bcast);
  }
  return std::unique_ptr<HloInstruction>();
}
}  // namespace

bool AlgebraicSimplifierVisitor::IsNondecreasingSublinear(
    const HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kCbrt:
    case HloOpcode::kErf:
    case HloOpcode::kLogistic:
    case HloOpcode::kTanh:
      return true;
    default:
      return false;
  }
}

absl::Status AlgebraicSimplifierVisitor::HandleMaximum(
    HloInstruction* maximum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(maximum, m::Maximum(m::Op(&lhs), m::Op(&rhs))));

  // max(x, x) -> x
  if (lhs == rhs) {
    return ReplaceInstruction(maximum, lhs);
  }

  // max(x, -inf) -> x
  PrimitiveType ty = maximum->shape().element_type();
  if (primitive_util::IsIntegralType(ty) ||
      (primitive_util::IsFloatingPointType(ty) &&
       options_.minmax_propagate_nan())) {
    // Note that `lhs` and `rhs` can have a different element type than
    // `maximum` if we are dealing with floating point types.
    PrimitiveType lhs_ty = lhs->shape().element_type();
    PrimitiveType rhs_ty = rhs->shape().element_type();
    Literal min_val_lhs = LiteralUtil::MinValue(lhs_ty);
    if (rhs_ty == ty && IsAll(lhs, min_val_lhs)) {
      return ReplaceInstruction(maximum, rhs);
    }
    Literal min_val_rhs = LiteralUtil::MinValue(rhs_ty);
    if (lhs_ty == ty && IsAll(rhs, min_val_rhs)) {
      return ReplaceInstruction(maximum, lhs);
    }
  }

  // max(max(x, y), y) -> max(x, y)
  // max(max(x, y), x) -> max(x, y)
  if (Match(lhs, m::MaximumAnyOrder(m::Op(), m::Op().Is(rhs)))) {
    return ReplaceInstruction(maximum, lhs);
  }
  // max(x, max(x, y)) -> max(x, y)
  if (Match(rhs, m::Maximum(m::Op().Is(lhs), m::Op()))) {
    return ReplaceInstruction(maximum, rhs);
  }
  // max(y, max(x, y)) -> max(y, x)
  // Note that we cannot simplify to max(x, y) here, as for the case that x and
  // y are NaN but with different sign, it will make a difference.
  if (Match(rhs, m::Maximum(m::Op(), m::Op().Is(lhs)))) {
    TF_RETURN_IF_ERROR(maximum->ReplaceOperandWith(1, rhs->mutable_operand(0)));
    MarkAsChanged();
    return absl::OkStatus();
  }

  HloInstruction* clamp_upper_bound_bcast;
  HloInstruction* clamp_lower_bound_bcast;
  HloInstruction* to_clamp;
  if (Match(maximum, m::MaximumAnyOrder(
                         m::Broadcast(&clamp_lower_bound_bcast,
                                      m::ConstantEffectiveScalar()),
                         m::MinimumAnyOrder(
                             m::Op(&to_clamp),
                             m::Broadcast(&clamp_upper_bound_bcast,
                                          m::ConstantEffectiveScalar()))))) {
    TF_ASSIGN_OR_RETURN(auto clamp,
                        MinMaxToClamp(clamp_lower_bound_bcast, to_clamp,
                                      clamp_upper_bound_bcast, simplifier_));
    if (clamp) {
      return ReplaceWithNewInstruction(maximum, std::move(clamp));
    }
  }

  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* max_operand;
  HloInstruction* clamp;
  if (Match(maximum,
            m::MaximumAnyOrder(
                m::Op(&max_operand),
                m::Clamp(&clamp, m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                         m::Op(&clamp_upper_bound))))) {
    if (max_operand == clamp_lower_bound &&
        ReplaceInstructionIfCompatible(maximum, clamp)) {
      return absl::OkStatus();
    }
  }

  // If the operands of the max are the same non-decreasing function, then we
  // can sink it; i.e. max(tanh(x), tanh(y)) to tanh(max(x, y))
  // We only do this if the function asymptotically satisfies |f(x)| <= |x| to
  // guarantee that no overflow occurs. Proof of correctness:
  /* https://cvc5.github.io/app/
  (set-logic ALL)
  (declare-fun f (Float32) Float32)
  (assert (forall ((x Float32) (y Float32))
                  (=> (fp.lt x y) (fp.leq (f x) (f y))))) ; NonDecreasing
  (assert (forall ((x Float32))
                  (fp.leq (fp.abs (f x)) (fp.abs x)))) ; Sublinear
  (assert (not (forall ((x Float32) (y Float32))
                       (fp.eq (fp.max (f x) (f y))
                              (f (fp.max x y)))))) ; Expect unsat
  (check-sat)
  */
  if (lhs->opcode() == rhs->opcode() && IsNondecreasingSublinear(lhs)) {
    TF_ASSIGN_OR_RETURN(
        auto new_maximum,
        MakeBinaryHlo(HloOpcode::kMaximum, lhs->mutable_operand(0),
                      rhs->mutable_operand(0)));
    VLOG(10) << "Sinking nondecreasing op through max";
    return ReplaceWithNewInstruction(
        maximum, HloInstruction::CreateUnary(maximum->shape(), lhs->opcode(),
                                             new_maximum));
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleMinimum(
    HloInstruction* minimum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(minimum, m::Minimum(m::Op(&lhs), m::Op(&rhs))));

  // min(x, x) -> x
  if (lhs == rhs) {
    return ReplaceInstruction(minimum, lhs);
  }

  // min(x, inf) -> x
  PrimitiveType ty = minimum->shape().element_type();
  if (primitive_util::IsIntegralType(ty) ||
      (primitive_util::IsFloatingPointType(ty) &&
       options_.minmax_propagate_nan())) {
    // Note that `lhs` and `rhs` can have a different element type than
    // `minimum` if we are dealing with floating point types.
    PrimitiveType lhs_ty = lhs->shape().element_type();
    PrimitiveType rhs_ty = rhs->shape().element_type();
    Literal max_val_lhs = LiteralUtil::MaxValue(lhs_ty);
    if (rhs_ty == ty && IsAll(lhs, max_val_lhs)) {
      return ReplaceInstruction(minimum, rhs);
    }
    Literal max_val_rhs = LiteralUtil::MaxValue(rhs_ty);
    if (lhs_ty == ty && IsAll(rhs, max_val_rhs)) {
      return ReplaceInstruction(minimum, lhs);
    }
  }

  // min(min(x, y), y) -> min(x, y)
  // min(min(x, y), x) -> min(x, y)
  if (Match(lhs, m::MinimumAnyOrder(m::Op(), m::Op().Is(rhs)))) {
    return ReplaceInstruction(minimum, lhs);
  }
  // min(x, min(x, y)) -> min(x, y)
  if (Match(rhs, m::Minimum(m::Op().Is(lhs), m::Op()))) {
    return ReplaceInstruction(minimum, rhs);
  }
  // min(y, min(x, y)) -> min(y, x)
  // Note that we cannot simplify to min(x, y) here, as for the case that x and
  // y are NaN but with different sign, it will make a difference.
  if (Match(rhs, m::Minimum(m::Op(), m::Op().Is(lhs)))) {
    TF_RETURN_IF_ERROR(minimum->ReplaceOperandWith(1, rhs->mutable_operand(0)));
    MarkAsChanged();
    return absl::OkStatus();
  }

  HloInstruction* clamp_upper_bound_bcast;
  HloInstruction* clamp_lower_bound_bcast;
  HloInstruction* to_clamp;
  if (Match(minimum, m::MinimumAnyOrder(
                         m::Broadcast(&clamp_upper_bound_bcast,
                                      m::ConstantEffectiveScalar()),
                         m::MaximumAnyOrder(
                             m::Op(&to_clamp),
                             m::Broadcast(&clamp_lower_bound_bcast,
                                          m::ConstantEffectiveScalar()))))) {
    TF_ASSIGN_OR_RETURN(auto clamp,
                        MinMaxToClamp(clamp_lower_bound_bcast, to_clamp,
                                      clamp_upper_bound_bcast, simplifier_));
    if (clamp) {
      return ReplaceWithNewInstruction(minimum, std::move(clamp));
    }
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleClamp(HloInstruction* clamp) {
  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* to_clamp;
  CHECK(Match(clamp, m::Clamp(m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                              m::Op(&clamp_upper_bound))));

  // clamp(a, clamp(a, x, b), b) -> clamp(a, x, b)
  if (Match(to_clamp, m::Clamp(m::Op().Is(clamp_lower_bound), m::Op(),
                               m::Op().Is(clamp_upper_bound))) &&
      ReplaceInstructionIfCompatible(clamp, to_clamp)) {
    return absl::OkStatus();
  }

  // Eliminate redundant clamping of replica-id or partition-id.
  if ((Match(to_clamp, m::PartitionId()) || Match(to_clamp, m::ReplicaId())) &&
      Match(clamp_lower_bound, m::ConstantScalar(0U)) &&
      Match(clamp_upper_bound, m::ConstantScalar())) {
    int64_t upper_bound = Cast<HloConstantInstruction>(clamp_upper_bound)
                              ->literal()
                              .GetFirstElement<uint32_t>();
    const HloModuleConfig& config = clamp->GetModule()->config();
    int64_t runtime_bound = Match(to_clamp, m::PartitionId())
                                ? config.num_partitions()
                                : config.replica_count();

    // If num_partitions or replica_count is 1, infer it as unknown.
    // pid/rid < runtime_bound => The clamp(0, pid/rid, upper_bound) is
    // redundant if the runtime_bound <= upper_bound + 1;
    if (runtime_bound != 1 && runtime_bound <= upper_bound + 1) {
      return ReplaceInstruction(clamp, to_clamp);
    }
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::TryToReorderConvAddMultiply(
    HloInstruction* multiply) {
  if (!options_.enable_conv_add_multiply_reorder()) return absl::OkStatus();
  HloInstruction *input, *filter, *bias, *constant, *convolution, *broadcast,
      *add;
  // We conservatively only consider the case where the multiplier is a
  // broadcast of a 1D constant to the output feature dimension and the filter
  // is a constant so that they can be constant-folded.
  if (!Match(multiply,
             m::MultiplyAnyOrder(
                 m::AddAnyOrder(&add,
                                m::Convolution(&convolution, m::Op(&input),
                                               m::Constant(&filter))
                                    .WithOneUser(),
                                m::Op(&bias).WithOneUser()),
                 m::Broadcast(&broadcast, m::Constant(&constant).WithShape(
                                              m::Shape().WithRank(1)))
                     .WithOneUser()))) {
    return absl::OkStatus();
  }
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();
  if (broadcast->dimensions().size() != 1 ||
      broadcast->dimensions()[0] != dnums.output_feature_dimension()) {
    return absl::OkStatus();
  }

  HloInstruction* bcast_to_filter_dim =
      multiply->AddInstruction(HloInstruction::CreateBroadcast(
          filter->shape(), constant,
          {dnums.kernel_output_feature_dimension()}));
  HloInstruction* filter_multiply =
      multiply->AddInstruction(HloInstruction::CreateBinary(
          filter->shape(), HloOpcode::kMultiply, filter, bcast_to_filter_dim));
  HloInstruction* new_conv =
      multiply->AddInstruction(convolution->CloneWithNewOperands(
          convolution->shape(), {input, filter_multiply}));
  HloInstruction* bias_multiply =
      multiply->AddInstruction(HloInstruction::CreateBinary(
          bias->shape(), HloOpcode::kMultiply, bias, broadcast));
  std::unique_ptr<HloInstruction> new_add =
      add->CloneWithNewOperands(add->shape(), {new_conv, bias_multiply});
  return ReplaceWithNewInstruction(multiply, std::move(new_add));
}

absl::Status AlgebraicSimplifierVisitor::HandleMultiply(
    HloInstruction* multiply) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(multiply, m::Multiply(m::Op(&lhs), m::Op(&rhs))));
  // LHS*1 => LHS
  VLOG(10) << "trying transform [LHS*1 => LHS]: " << multiply->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(multiply, lhs)) {
    return absl::OkStatus();
  }
  // 1*RHS => RHS
  VLOG(10) << "trying transform [1*RHS => RHS]: " << multiply->ToString();
  if (IsAll(lhs, 1) && ReplaceInstructionIfCompatible(multiply, rhs)) {
    return absl::OkStatus();
  }

  // 0*RHS => 0. Only applies for integral types for correct NaN-handling.
  if (IsAll(lhs, 0) &&
      primitive_util::IsIntegralType(multiply->shape().element_type()) &&
      ReplaceInstructionIfCompatible(multiply, lhs)) {
    return absl::OkStatus();
  }
  // LHS*0 => 0
  if (IsAll(rhs, 0) &&
      primitive_util::IsIntegralType(multiply->shape().element_type()) &&
      ReplaceInstructionIfCompatible(multiply, rhs)) {
    return absl::OkStatus();
  }

  {
    // Mul(Negate(A), Negate(B)) => Mul(A, B)
    HloInstruction *a, *b;
    if (Match(multiply,
              m::Multiply(m::Negate(m::Op(&a)), m::Negate(m::Op(&b))))) {
      TF_RETURN_IF_ERROR(multiply->ReplaceOperandWith(0, a));
      TF_RETURN_IF_ERROR(multiply->ReplaceOperandWith(1, b));
      MarkAsChanged();
      return absl::OkStatus();
    }
  }

  {
    HloInstruction* abs_operand;
    if (lhs == rhs && Match(lhs, m::Abs(m::Op(&abs_operand))) &&
        !ShapeUtil::ElementIsComplex(abs_operand->shape())) {
      TF_RETURN_IF_ERROR(multiply->ReplaceOperandWith(0, abs_operand));
      TF_RETURN_IF_ERROR(multiply->ReplaceOperandWith(1, abs_operand));
      MarkAsChanged();
      return absl::OkStatus();
    }
  }

  {
    HloInstruction *convert_operand, *operand;
    // Mul(Convert(Pred), operand) => select(pred, operand, 0)
    if (Match(multiply,
              m::MultiplyAnyOrder(
                  m::Op(&operand),
                  m::Convert(
                      m::Op(&convert_operand)
                          .WithShape(m::Shape().WithElementType(PRED)))))) {
      HloInstruction* zero_like_multiply =
          BroadcastZeros(computation_, multiply->shape());
      return ReplaceWithNewInstruction(
          multiply, HloInstruction::CreateTernary(
                        multiply->shape(), HloOpcode::kSelect, convert_operand,
                        operand, zero_like_multiply));
    }
  }

  {
    HloInstruction *a, *b, *c1, *c2;
    // Mul(Mul(x, constant1), Mul(y, constant2)) => Mul(Mul(x, y),
    // constant1*constant2)
    if (Match(multiply,
              m::MultiplyAnyOrder(
                  m::MultiplyAnyOrder(m::NonConstant(&a), m::Constant(&c1)),
                  m::MultiplyAnyOrder(m::NonConstant(&b), m::Constant(&c2))))) {
      TF_ASSIGN_OR_RETURN(auto* product_of_constants,
                          MakeBinaryHlo(HloOpcode::kMultiply, c1, c2));
      if (ShapeUtil::IsScalar(product_of_constants->shape()) &&
          !ShapeUtil::IsScalar(multiply->shape())) {
        product_of_constants =
            multiply->AddInstruction(HloInstruction::CreateBroadcast(
                multiply->shape(), product_of_constants, {}));
      }

      return ReplaceWithNewInstruction(
          multiply, HloInstruction::CreateBinary(
                        multiply->shape(), HloOpcode::kMultiply,
                        multiply->AddInstruction(HloInstruction::CreateBinary(
                            multiply->shape(), HloOpcode::kMultiply, a, b)),
                        product_of_constants));
    }
  }

  {
    HloInstruction *a, *c1, *c2;
    // Mul(Mul(a, constant1), constant2) => Mul(a, constant1*constant2)
    if (Match(multiply,
              m::MultiplyAnyOrder(
                  m::MultiplyAnyOrder(m::NonConstant(&a), m::Constant(&c1)),
                  m::Constant(&c2)))) {
      TF_ASSIGN_OR_RETURN(auto* product_of_constants,
                          MakeBinaryHlo(HloOpcode::kMultiply, c1, c2));
      if (ShapeUtil::IsScalar(product_of_constants->shape()) &&
          !ShapeUtil::IsScalar(multiply->shape())) {
        product_of_constants =
            multiply->AddInstruction(HloInstruction::CreateBroadcast(
                multiply->shape(), product_of_constants, {}));
      }

      return ReplaceWithNewInstruction(
          multiply,
          HloInstruction::CreateBinary(multiply->shape(), HloOpcode::kMultiply,
                                       a, product_of_constants));
    }
  }

  {
    HloInstruction *a, *b, *constant, *op;
    // Mul(Mul(a, constant1), Broadcast(b)) =>
    // Mul(Broadcast(Mul(b, constant1), a))
    if (Match(multiply,
              m::MultiplyAnyOrder(m::MultiplyAnyOrder(m::NonConstant(&a),
                                                      m::Constant(&constant)),
                                  m::Op(&op))) ||
        Match(multiply,
              m::MultiplyAnyOrder(
                  m::MultiplyAnyOrder(m::NonConstant(&a),
                                      m::Broadcast(m::Constant(&constant))),
                  m::Op(&op)))) {
      // Check that the other side was a broadcast, and not of a constant.
      if (ShapeUtil::IsScalar(constant->shape()) &&
          Match(op, m::Broadcast(m::NonConstant()))) {
        auto dims = op->dimensions();
        b = op->mutable_operand(0);
        if (!ShapeUtil::IsScalar(b->shape())) {
          constant = multiply->AddInstruction(
              HloInstruction::CreateBroadcast(b->shape(), constant, {}));
        }

        auto new_mul = multiply->AddInstruction(HloInstruction::CreateBinary(
            b->shape(), HloOpcode::kMultiply, b, constant));

        return ReplaceWithNewInstruction(
            multiply,
            HloInstruction::CreateBinary(
                multiply->shape(), HloOpcode::kMultiply, a,
                multiply->AddInstruction(HloInstruction::CreateBroadcast(
                    multiply->shape(), new_mul, dims))));
      }
    }
  }

  VLOG(10) << "trying transform [(A * C1) * C2 => A * (C1 * C2)]";
  HloInstruction *a, *c1, *c2;
  if (Match(multiply,
            m::Multiply(m::Multiply(m::NonConstant(&a), m::Constant(&c1)),
                        m::Constant(&c2))) ||
      Match(multiply,
            m::Multiply(
                m::Multiply(m::Op(&a), m::Broadcast(m::ConstantScalar(&c1))),
                m::Broadcast(m::ConstantScalar(&c2))))) {
    TF_ASSIGN_OR_RETURN(auto* product_of_constants,
                        MakeBinaryHlo(HloOpcode::kMultiply, c1, c2));
    if (ShapeUtil::IsScalar(product_of_constants->shape()) &&
        !ShapeUtil::IsScalar(multiply->shape())) {
      product_of_constants =
          multiply->AddInstruction(HloInstruction::CreateBroadcast(
              multiply->shape(), product_of_constants, {}));
    }
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateBinary(multiply->shape(), HloOpcode::kMultiply, a,
                                     product_of_constants));
  }

  VLOG(10) << "trying to transform exp(LHS) * exp(RHS) => exp(LHS+RHS) "
           << multiply->ToString();
  if (Match(multiply, m::Multiply(m::Exp(m::Op(&lhs)), m::Exp(m::Op(&rhs))))) {
    auto add = multiply->AddInstruction(HloInstruction::CreateBinary(
        multiply->shape(), HloOpcode::kAdd, lhs, rhs));
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateUnary(multiply->shape(), HloOpcode::kExp, add));
  }

  VLOG(10) << "trying transform [sqrt(x) * sqrt(x) => x], for x >= 0 "
           << multiply->ToString();
  if (Match(multiply,
            m::Multiply(m::Sqrt(m::Op(&lhs)), m::Sqrt(m::Op(&rhs)))) &&
      lhs == rhs && IsNonNegative(lhs, options_)) {
    return ReplaceInstruction(multiply, lhs);
  }

  VLOG(10) << "trying transform [rsqrt(x) * rsqrt(x) => 1/x], for x >= 0 "
           << multiply->ToString();
  if (Match(multiply,
            m::Multiply(m::Rsqrt(m::Op(&lhs)), m::Rsqrt(m::Op(&rhs)))) &&
      lhs == rhs && IsNonNegative(lhs, options_)) {
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateBinary(multiply->shape(), HloOpcode::kDivide,
                                     MakeScalarLike(lhs, 1), lhs));
  }

  return TryToReorderConvAddMultiply(multiply);
}

absl::Status AlgebraicSimplifierVisitor::HandleNegate(HloInstruction* negate) {
  // negate(negate(x)) => x
  HloInstruction* x;
  if (Match(negate, m::Negate(m::Negate(m::Op(&x)))) &&
      ReplaceInstructionIfCompatible(negate, x)) {
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleNot(
    HloInstruction* logical_not) {
  // not(not(x)) => x
  HloInstruction* x;
  if (Match(logical_not, m::Not(m::Not(m::Op(&x)))) &&
      ReplaceInstructionIfCompatible(logical_not, x)) {
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleOr(HloInstruction* logical_or) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_or, m::Or(m::Op(&lhs), m::Op(&rhs))));

  // Simplify logical or
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A || True => True
    VLOG(10) << "trying transform [A || True => True]: "
             << logical_or->ToString();
    if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(logical_or, rhs)) {
      return absl::OkStatus();
    }
    // True || A => True
    VLOG(10) << "trying transform [True || A => True]: "
             << logical_or->ToString();
    if (IsAll(lhs, 1) && ReplaceInstructionIfCompatible(logical_or, lhs)) {
      return absl::OkStatus();
    }
  }

  // A || False => A and A | 0 => A
  VLOG(10) << "trying transform [A || False => A]: " << logical_or->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(logical_or, lhs)) {
    return absl::OkStatus();
  }

  // False || A => A and 0 | A => A
  VLOG(10) << "trying transform [False || A => A]: " << logical_or->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfCompatible(logical_or, rhs)) {
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleLog(HloInstruction* log) {
  // ln(exp(A)) => A
  VLOG(10) << "trying transform [ln(exp(A)) => A]: " << log->ToString();
  HloInstruction *a, *b;
  if (Match(log, m::Log(m::Exp(m::Op(&a)))) &&
      ReplaceInstructionIfCompatible(log, a)) {
    return absl::OkStatus();
  }

  // ln(pow(A,B)) => B*ln(abs(A))
  // or B*ln(A) if A is complex.
  if (Match(log, m::Log(m::Power(m::Op(&a), m::Op(&b))))) {
    auto abs_a = ShapeUtil::ElementIsComplex(a->shape())
                     ? a
                     : log->AddInstruction(HloInstruction::CreateUnary(
                           log->shape(), HloOpcode::kAbs, a));
    auto new_log = log->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, abs_a));
    auto non_zero_b =
        log->mutable_operand(0)->AddInstruction(HloInstruction::CreateBinary(
            log->shape(), HloOpcode::kMultiply, new_log, b));
    TF_ASSIGN_OR_RETURN(
        auto b_is_zero,
        MakeCompareHlo(Comparison::Direction::kEq, b, MakeScalarLike(b, 0.0)));
    simplifier_->UpdateLayout(b_is_zero->mutable_shape());
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateTernary(log->shape(), HloOpcode::kSelect,
                                           b_is_zero, MakeScalarLike(log, 0.0),
                                           non_zero_b));
  }

  if (Match(log, m::Log(m::Sqrt(m::Op(&a))))) {
    auto new_log = log->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a));
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, MakeScalarLike(log, 0.5)));
  }

  if (Match(log, m::Log(m::Rsqrt(m::Op(&a))))) {
    auto new_log = log->AddInstruction(
        HloInstruction::CreateUnary(log->shape(), HloOpcode::kLog, a));
    return ReplaceWithNewInstruction(
        log, HloInstruction::CreateBinary(log->shape(), HloOpcode::kMultiply,
                                          new_log, MakeScalarLike(log, -0.5)));
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleGetTupleElement(
    HloInstruction* get_tuple_element) {
  auto operand = get_tuple_element->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kTuple) {
    // get_tuple_element(make_tuple({A_0, A_1, ..., A_n}), i) => A_i
    VLOG(10) << "trying transform "
             << "[get_tuple_element(make_tuple({...,A_i,...}), i)] => A_i: "
             << get_tuple_element->ToString();
    if (ReplaceInstructionIfCompatible(
            get_tuple_element,
            operand->mutable_operand(get_tuple_element->tuple_index()))) {
      return absl::OkStatus();
    }
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleOptimizationBarrier(
    HloInstruction* barrier) {
  if (!barrier->shape().IsTuple() ||
      barrier == computation_->root_instruction()) {
    return absl::OkStatus();
  }

  // The goal of this transformation is to enable DCE on the tuple elements of
  // an optimization barrier operand. To do this safely, the optimization
  // barrier users must not use the tuple element and the only use of the index
  // of the operand should be the tuple instruction producing the operand of the
  // optimization barrier. Additionally if the operand is a tuple producing
  // instruction it should also be safe to create a sub tuple of only the used
  // components to enable module level dce.
  std::vector<bool> used_elements(barrier->shape().tuple_shapes_size());
  bool has_non_gte_use = false;
  for (auto use : barrier->users()) {
    if (use->opcode() != HloOpcode::kGetTupleElement) {
      has_non_gte_use = true;
      break;
    }
    used_elements[use->tuple_index()] = true;
  }

  HloInstruction* operand = barrier->mutable_operand(0);
  if (operand->opcode() == HloOpcode::kTuple) {
    for (int64_t i = 0; i < operand->operand_count(); ++i) {
      if (used_elements[i]) {
        continue;
      }
      if (operand->operand(i)->user_count() > 1 ||
          operand->operand(i) == computation_->root_instruction()) {
        used_elements[i] = true;
      }
    }
  }

  if (has_non_gte_use || !absl::c_linear_search(used_elements, false)) {
    return absl::OkStatus();
  }

  MarkAsChanged();
  std::vector<int64_t> index_map(used_elements.size(), -1);
  std::vector<HloInstruction*> operands;
  int64_t current_index = 0;
  for (int64_t element = 0; element < used_elements.size(); ++element) {
    if (!used_elements[element]) {
      continue;
    }
    index_map[element] = current_index++;
    if (operand->opcode() == HloOpcode::kTuple) {
      operands.push_back(operand->mutable_operand(element));
    } else {
      operands.push_back(barrier->AddInstruction(
          HloInstruction::CreateGetTupleElement(operand, element)));
    }
  }

  HloInstruction* new_operand =
      operand->AddInstruction(HloInstruction::CreateTuple(operands));
  TF_RETURN_IF_ERROR(barrier->ReplaceOperandWithDifferentShape(0, new_operand));
  *barrier->mutable_shape() = new_operand->shape();
  for (auto use : barrier->users()) {
    CHECK_EQ(use->opcode(), HloOpcode::kGetTupleElement);
    use->set_tuple_index(index_map[use->tuple_index()]);
  }
  return absl::OkStatus();
}

namespace {

std::optional<std::vector<int64_t>> ReshapeLeavesDimensionsUnmodified(
    const HloInstruction* hlo, absl::Span<const int64_t> input_dim_indices) {
  CHECK_EQ(hlo->opcode(), HloOpcode::kReshape);
  return ShapeUtil::ReshapeLeavesDimensionsUnmodified(
      hlo->operand(0)->shape(), hlo->shape(), input_dim_indices);
}

// Returns true if the output of "instruction" is a permutation of the
// elements of "operand". Precondition: "operand" is an operand of
// "instruction".
bool OutputIsPermutationOfOperandElements(HloInstruction* instruction,
                                          HloInstruction* operand) {
  DCHECK(!instruction->OperandIndices(operand).empty());
  switch (instruction->opcode()) {
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTranspose:
      return true;
    case HloOpcode::kSort:
      return (!instruction->shape().IsTuple());
    default:
      return false;
  }
}

// Returns true if the output of "instruction" is a subset of the elements of
// "operand". Precondition: "operand" is an operand of "instruction".
bool OutputIsSubsetOfOperandElements(HloInstruction* instruction,
                                     HloInstruction* operand) {
  const auto operand_indices = instruction->OperandIndices(operand);
  CHECK(!operand_indices.empty());
  if (operand_indices.size() != 1) {
    return false;
  }
  int64_t operand_index = operand_indices[0];
  switch (instruction->opcode()) {
    case HloOpcode::kSlice:
      CHECK_EQ(0, operand_index);
      return true;
    case HloOpcode::kDynamicSlice:
      return operand_index == 0;
    default:
      return false;
  }
}

}  // namespace

absl::Status AlgebraicSimplifierVisitor::HandleBroadcast(
    HloInstruction* broadcast) {
  HloInstruction* operand;
  CHECK(Match(broadcast, m::Broadcast(m::Op(&operand))));
  auto dims = *broadcast->mutable_dimensions();
  // A degenerate broadcast of a reshape that does not change the number of
  // elements can be replaced by a reshape.
  if (std::is_sorted(dims.begin(), dims.end()) &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    VLOG(10) << "transform broadcast(X) -> reshape(X) where "
                "n(broadcast(X)) == n(X)";
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateReshape(broadcast->shape(), operand));
  }

  // A broadcast that has the same input and output rank can be converted into a
  // transpose with the inverse of broadcast's dimensions.
  if (broadcast->shape().dimensions_size() ==
          operand->shape().dimensions_size() &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateTranspose(
                       broadcast->shape(), operand,
                       InversePermutation(broadcast->dimensions())));
  }

  // A broadcast of a reshape which merely inserts 1-sized dimensions can
  // elide its operand.
  {
    std::optional<ShapeUtil::ShapeEqualityDescriptor> reshape_degenerate =
        operand->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
    if (reshape_degenerate.has_value() &&
        reshape_degenerate->deleted_dimensions.empty()) {
      absl::c_reverse(reshape_degenerate->inserted_dimensions);
      for (auto inserted_index : reshape_degenerate->inserted_dimensions) {
        dims.erase(dims.begin() + inserted_index);
      }
      return ReplaceWithNewInstruction(
          broadcast,
          HloInstruction::CreateBroadcast(broadcast->shape(),
                                          operand->mutable_operand(0), dims));
    }
  }

  if (options_.enable_sink_broadcast()) {
    TF_ASSIGN_OR_RETURN(bool sink_succeeded,
                        TryToSinkBroadcastAfterElementwiseOps(broadcast));
    if (sink_succeeded) {
      MarkAsChanged();
      return absl::OkStatus();
    }
  }

  // A scalar broadcast feeding an instruction which only permutes (reshape,
  // transpose, sort, reverse) or selects a subset of operand elements (slice,
  // dynamic slice) can be replaced with a broadcast directly to the output
  // shape of the instruction.
  if (ShapeUtil::IsScalar(operand->shape())) {
    for (HloInstruction* user : broadcast->users()) {
      // Skip if the broadcast user has no uses itself.
      if (user->IsDead()) {
        continue;
      }
      if (OutputIsPermutationOfOperandElements(user, broadcast) ||
          OutputIsSubsetOfOperandElements(user, broadcast)) {
        VLOG(10) << "transform permuting/subset of a scalar broadcast into "
                 << "a single broadcast";
        HloInstruction* new_broadcast =
            user->AddInstruction(HloInstruction::CreateBroadcast(
                ShapeUtil::MakeStaticShape(user->shape()), operand, {}));
        // Use HloInstruction::ReplaceAllUsesWith instead of
        // HloComputation::ReplaceWithNewInstruction because we are replacing an
        // instruction other than the visited instruction.
        MarkAsChanged();
        return user->ReplaceAllUsesWith(new_broadcast);
      }
    }
    return absl::OkStatus();
  }

  // broadcast(iota) -> iota.
  if (operand->opcode() == HloOpcode::kIota) {
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateIota(
            broadcast->shape(),
            dims[Cast<HloIotaInstruction>(operand)->iota_dimension()]));
  }

  // Merge two consecutive broadcasts into a single one.
  if (operand->opcode() == HloOpcode::kBroadcast) {
    std::vector<int64_t> new_dimensions;
    new_dimensions.reserve(operand->dimensions().size());
    for (auto dim : operand->dimensions()) {
      new_dimensions.push_back(dims[dim]);
    }
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateBroadcast(
            broadcast->shape(), operand->mutable_operand(0), new_dimensions));
  }
  if (options_.is_layout_sensitive()) {
    return absl::OkStatus();
  }
  if (options_.enable_broadcast_degenerate_dimension() &&
      ShapeUtil::HasDegenerateDimensions(operand->shape())) {
    auto new_operand = operand->AddInstruction(HloInstruction::CreateReshape(
        ShapeUtil::DropDegenerateDimensions(operand->shape()), operand));
    std::vector<int64_t> new_dims;
    new_dims.reserve(new_operand->shape().dimensions_size());
    for (int64_t i = 0; i < operand->shape().dimensions_size(); ++i) {
      if (operand->shape().dimensions(i) != 1) {
        new_dims.push_back(dims[i]);
      }
    }
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateBroadcast(broadcast->shape(),
                                                   new_operand, new_dims));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleCompare(
    HloInstruction* compare) {
  HloInstruction* lhs;
  HloInstruction* rhs;
  CHECK(Match(compare, m::Compare(m::Op(&lhs), m::Op(&rhs))));

  if (Cast<HloCompareInstruction>(compare)->type() ==
      Comparison::Type::kUnsigned) {
    // X u<  0 -> false
    if (compare->comparison_direction() == ComparisonDirection::kLt &&
        IsAll(rhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, false));
    }
    // X u>= 0 -> true
    if (compare->comparison_direction() == ComparisonDirection::kGe &&
        IsAll(rhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, true));
    }
    // 0 u>  X -> false
    if (compare->comparison_direction() == ComparisonDirection::kGt &&
        IsAll(lhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, false));
    }
    // 0 u<= X -> true
    if (compare->comparison_direction() == ComparisonDirection::kLe &&
        IsAll(lhs, 0)) {
      return ReplaceInstruction(compare, MakeScalarLike(compare, true));
    }
  }

  if (compare->comparison_direction() == ComparisonDirection::kLt &&
      IsNonNegative(lhs, options_) && IsAll(rhs, 0)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, false));
  } else if (compare->comparison_direction() == ComparisonDirection::kGt &&
             IsAll(lhs, 0) && IsNonNegative(rhs, options_)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, false));
  } else if (compare->comparison_direction() == ComparisonDirection::kGe &&
             IsNonNegative(lhs, options_) && IsAll(rhs, 0)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, true));
  } else if (compare->comparison_direction() == ComparisonDirection::kLe &&
             IsAll(lhs, 0) && IsNonNegative(rhs, options_)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, true));
  }
  if (lhs == rhs &&
      primitive_util::IsIntegralType(lhs->shape().element_type())) {
    switch (compare->comparison_direction()) {
      case ComparisonDirection::kGt:
      case ComparisonDirection::kLt:
      case ComparisonDirection::kNe:
        return ReplaceInstruction(compare, MakeScalarLike(compare, false));
      case ComparisonDirection::kEq:
      case ComparisonDirection::kGe:
      case ComparisonDirection::kLe:
        return ReplaceInstruction(compare, MakeScalarLike(compare, true));
    }
  }
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    if (compare->comparison_direction() == ComparisonDirection::kNe) {
      // A != false -> A
      if (IsAll(rhs, false)) {
        return ReplaceInstruction(compare, lhs);
      }
      // false != A -> A
      if (IsAll(lhs, false)) {
        return ReplaceInstruction(compare, rhs);
      }
    } else if (compare->comparison_direction() == ComparisonDirection::kEq) {
      // A == true -> A
      if (IsAll(rhs, true)) {
        return ReplaceInstruction(compare, lhs);
      }
      // true == A -> A
      if (IsAll(lhs, true)) {
        return ReplaceInstruction(compare, rhs);
      }
    }
  }

  // Below is a common JAX code issue encountered when generating a Causal mask
  // The user either neglected to specify `dtype=bool` in `ones()`
  // or mistakenly applied `.astype(bool)` to the result of `tril()` instead of
  // to `ones()`. Consequently, the mask will be converted from f32 to bool,
  // resulting in suboptimal HLO.
  //
  // mask = jnp.tril(jnp.ones((seq_len, seq_len)))
  // res = jnp.where(mask, x, -jnp.inf)
  //
  // # it will be lowered to the following suboptimal HLO
  // %cmp0 = pred compare(s32, s32, direction=GE)
  // %sel0 = f32 select(%cmp0, ones, zeros)
  // %cmp1 = pred compare(%sel0, zeros, direction=NE)
  //
  // # which can be simplified to just
  // %cmp0 = pred compare(s32, s32, direction=GE)
  //
  // Simplification:
  // Ne(select(Ge(a, b), ones, zeros), zeros) -> Ge(a, b)
  if (compare->comparison_direction() == ComparisonDirection::kNe &&
      IsAll(rhs, 0)) {
    HloInstruction* compare0;
    HloInstruction* sel_on_true;
    HloInstruction* sel_on_false;
    if (Match(lhs,
              m::Select(m::Op(&compare0)
                            .WithOpcode(HloOpcode::kCompare)
                            .WithComparisonDirection(ComparisonDirection::kGe),
                        m::Op(&sel_on_true), m::Op(&sel_on_false))) &&
        IsAll(sel_on_true, 1) && IsAll(sel_on_false, 0) &&
        SameShape(compare->shape(), compare0->shape())) {
      return ReplaceInstruction(compare, compare0);
    }
  }

  // Gt(Max(a,b), a) -> Gt(b,a)
  // Gt(Max(a,b), b) -> Gt(a,b)
  // Gt(a, Min(a,b)) -> Gt(a,b)
  // Gt(b, Min(a,b)) -> Gt(b,a)
  if (compare->comparison_direction() == ComparisonDirection::kGt) {
    HloInstruction* a;
    HloInstruction* b;
    if (Match(lhs, m::Maximum(m::Op(&a), m::Op(&b)))) {
      if (rhs == a) {  // Gt(Max(a,b), a) -> Gt(b,a)
        TF_RETURN_IF_ERROR(compare->ReplaceOperandWith(0, b));
        MarkAsChanged();
        return absl::OkStatus();
      } else if (rhs == b) {  // Gt(Max(a,b), b) -> Gt(a,b)
        TF_RETURN_IF_ERROR(compare->ReplaceOperandWith(0, a));
        MarkAsChanged();
        return absl::OkStatus();
      }
    } else if (Match(rhs, m::Minimum(m::Op(&a), m::Op(&b)))) {
      if (lhs == a) {  // Gt(a, Min(a,b)) -> Gt(a,b)
        TF_RETURN_IF_ERROR(compare->ReplaceOperandWith(1, b));
        MarkAsChanged();
        return absl::OkStatus();
      } else if (lhs == b) {  // Gt(b, Min(a,b)) -> Gt(b,a)
        TF_RETURN_IF_ERROR(compare->ReplaceOperandWith(1, a));
        MarkAsChanged();
        return absl::OkStatus();
      }
    }
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleConvert(
    HloInstruction* convert) {
  PrimitiveType src_type = convert->operand(0)->shape().element_type();
  PrimitiveType dest_type = convert->shape().element_type();
  // A conversion to the same element type as the operand is a nop and can be
  // removed.  A conversion of a constant can be simplified by making a new
  // constant.
  if (src_type == dest_type) {
    return ReplaceInstruction(convert, convert->mutable_operand(0));
  }

  // Eliminate a convert pair if it is a no-op. The following are a few
  // example cases that are being handled:
  // 1. convert(convert(A, $TYPE1), $TYPE2) is simplified to A if A is of $TYPE2
  //    and convert(A, $TYPE1) is an upcast
  // 2. convert(convert(A, $TYPE1),$TYPE2) is simplified to A if A is of $TYPE2
  //    and convert(A, $TYPE1) is an upcast and is an integral conversion from
  //    unsigned to signed (only signed to unsigned conversion is NOT allowed)
  // 3. Tuple(convert(A, $TYPE1) , floor(convert(convert(A, $TYPE1), $TYPE2)),
  //    convert(convert(A, $TYPE1), $TYPE2)) is simplified to Tuple(convert(A,
  //    $TYPE1) , floor(A), A) -> a case where the first convert has a
  //    fan-out
  if (IsConvertPairNoOp(convert)) {
    return ReplaceInstruction(convert,
                              convert->mutable_operand(0)->mutable_operand(0));
  }

  // Try to replace convert(constant) with a constant of the right type to begin
  // with. Disallow moving sub-byte types since they may not be supported for
  // some ops.
  HloInstruction* constant;
  if (options_.use_convert_constant_folding() &&
      Match(convert, m::Convert(m::Constant(&constant))) &&
      primitive_util::BitWidth(dest_type) <=
          primitive_util::BitWidth(src_type) &&
      constant->user_count() == 1 && primitive_util::BitWidth(dest_type) >= 8) {
    TF_ASSIGN_OR_RETURN(Literal dest_literal,
                        constant->literal().Convert(dest_type));
    VLOG(10) << "Replacing convert(constant) with constant";
    return ReplaceWithNewInstruction(
        convert, HloInstruction::CreateConstant(std::move(dest_literal)));
  }

  return TryRemoveUpcastAndDowncastSurroundingBinaryOp(convert);
}

absl::Status AlgebraicSimplifierVisitor::HandleCustomCall(
    HloInstruction* custom_call) {
  // Remove redundant slice to dynamic of pad to static
  HloInstruction *pad_to_static0, *pad_to_static1, *pad_to_static_operand;
  if (Match(
          custom_call,
          m::CustomCall(
              {"SliceToDynamic"},
              m::GetTupleElement(m::CustomCall(&pad_to_static0, {"PadToStatic"},
                                               m::Op(&pad_to_static_operand)),
                                 0),
              m::GetTupleElement(
                  m::CustomCall(&pad_to_static1, {"PadToStatic"}, m::Op()),
                  1))) &&
      pad_to_static0 == pad_to_static1 &&
      SameShape(custom_call->shape(), pad_to_static_operand->shape())) {
    return ReplaceInstruction(custom_call, pad_to_static_operand);
  }
  if (options_.is_layout_sensitive() &&
      custom_call->IsCustomCall("LayoutConstraint")) {
    if (SameShape(custom_call->shape(), custom_call->operand(0)->shape())) {
      return ReplaceInstruction(custom_call, custom_call->mutable_operand(0));
    }
    return ReplaceWithNewInstruction(
        custom_call,
        HloInstruction::CreateUnary(custom_call->shape(), HloOpcode::kCopy,
                                    custom_call->mutable_operand(0)));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleExp(
    HloInstruction* exponential) {
  // Exp(0) => 1
  if (Match(exponential, m::Exp(m::ConstantScalar(0))) ||
      Match(exponential, m::Exp(m::Broadcast(m::ConstantScalar(0))))) {
    return ReplaceInstruction(exponential, MakeScalarLike(exponential, 1.0));
  }
  return absl::OkStatus();
}

// Complex(Real(c), Imag(c)) -> c
absl::Status AlgebraicSimplifierVisitor::HandleComplex(
    HloInstruction* complex) {
  HloInstruction *c0, *c1;
  if (Match(complex, m::Complex(m::Real(m::Op(&c0)), m::Imag(m::Op(&c1)))) &&
      c0 == c1) {
    return ReplaceInstruction(complex, c0);
  }
  return absl::OkStatus();
}

// Real(Complex(r, i)) -> r
absl::Status AlgebraicSimplifierVisitor::HandleReal(HloInstruction* real) {
  HloInstruction* op;
  if (Match(real, m::Real(m::Complex(m::Op(&op), m::Op())))) {
    return ReplaceInstruction(real, op);
  }
  return absl::OkStatus();
}

// Imag(Complex(r, i)) -> i
absl::Status AlgebraicSimplifierVisitor::HandleImag(HloInstruction* imag) {
  HloInstruction* op;
  if (Match(imag, m::Imag(m::Complex(m::Op(), m::Op(&op))))) {
    return ReplaceInstruction(imag, op);
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleIota(
    HloInstruction* instruction) {
  // iota -> zero if the iota dimension never produces an element other than
  // zero.
  auto* iota = Cast<HloIotaInstruction>(instruction);
  if (iota->shape().dimensions(iota->iota_dimension()) <= 1) {
    return ReplaceInstruction(iota, MakeScalarLike(iota, 0));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandlePad(HloInstruction* pad) {
  if (ShapeUtil::IsZeroElementArray(pad->operand(0)->shape())) {
    return ReplaceWithNewInstruction(
        pad, HloInstruction::CreateBroadcast(pad->shape(),
                                             pad->mutable_operand(1), {}));
  }

  // Interior padding on one sized dimensions have no effect. As a result it
  // makes other simplifications possible if there is no interior padding.
  if (HasInteriorPadding(pad->padding_config())) {
    PaddingConfig padding_config = pad->padding_config();
    bool cleared_interior_padding = false;
    for (int64_t i = 0; i < pad->shape().dimensions_size(); ++i) {
      if (padding_config.dimensions(i).interior_padding() > 0 &&
          pad->operand(0)->shape().dimensions(i) == 1) {
        cleared_interior_padding = true;
        padding_config.mutable_dimensions(i)->set_interior_padding(0);
      }
    }
    if (cleared_interior_padding) {
      return ReplaceWithNewInstruction(
          pad,
          HloInstruction::CreatePad(pad->shape(), pad->mutable_operand(0),
                                    pad->mutable_operand(1), padding_config));
    }
  }

  // Eliminate nop pads (padding all zero), and replace a pad with negative
  // padding with a pad with non-negative padding followed by a slice.
  bool all_zero = true;
  bool has_negative = false;
  // Used to possibly split off the unchanged padding dimensions.
  std::vector<int64_t> padding_dimensions;
  int64_t dimension_index = 0;
  for (auto& padding_dimension : pad->padding_config().dimensions()) {
    if (padding_dimension.edge_padding_low() < 0 ||
        padding_dimension.edge_padding_high() < 0) {
      has_negative = true;
    }
    if (padding_dimension.edge_padding_low() != 0 ||
        padding_dimension.edge_padding_high() != 0) {
      all_zero = false;
      padding_dimensions.push_back(dimension_index);
    } else if (padding_dimension.interior_padding()) {
      padding_dimensions.push_back(dimension_index);
    }
    dimension_index++;
  }

  if (all_zero) {
    if (ReplaceInstructionIfCompatible(pad, pad->mutable_operand(0))) {
      return absl::OkStatus();
    }
  }

  // The context of this optimization can be found at b/163617402
  // It tries to capture the case of pad(broadcast(x)), where
  // x->shape().dimensions(), or broadcast(x)->dimensions(), is
  // a subset of the padded dimensions in pad->config(),
  // and the padded dimensions in pad->config() is in turn a strict
  // subset of broadcast->shape().dimensions(). The combined op can be
  // rewritten to broadcast2(pad(broadcast1(x))), where broadcast1 extends
  // x  with dimensions that need to be padded, and broadcast2 extends
  // the result of padding to full dimensions.
  // TODO(qyi): for future extensions: The condition for broadcast(x)
  // ->dimensions() to be a subset of padded dimensions in pad->config()
  // does not have to be strictly required, but it makes the calculation
  // for optimization easier, so it is required by the current implementation.
  // Only the second condition between the padded dimensions and the
  // dimensions of the final shape have to be enforced for the optimization
  // to make sense. If needed to remove the first constraint, the shape
  // calculations across the implementation need to be re-adjusted.
  auto pad_dims = padding_dimensions.size();
  if (pad_dims < dimension_index &&
      pad->operand(0)->opcode() == HloOpcode::kBroadcast &&
      pad->operand(0)->user_count() == 1 &&
      pad->operand(0)->operand(0)->shape().dimensions_size() <= pad_dims) {
    // Check broadcast operand dimensions is a subset of pading_dimensions.
    // If not, skip the optimization.
    bool opt_is_valid = true;
    std::vector<int64_t> broadcast_dimensions;
    HloBroadcastInstruction* broadcast =
        static_cast<HloBroadcastInstruction*>(pad->mutable_operand(0));
    for (auto broadcast_index : broadcast->dimensions()) {
      bool found = false;
      for (int i = 0; i < pad_dims; ++i) {
        if (broadcast_index == padding_dimensions[i]) {
          broadcast_dimensions.push_back(i);
          found = true;
          break;
        }
      }
      if (!found) {
        opt_is_valid = false;
        break;
      }
    }
    if (opt_is_valid) {
      auto pad_shape = pad->shape();
      auto broadcast_shape = broadcast->shape();
      auto pad_shape1 = pad_shape;
      auto broadcast_shape1 = broadcast_shape;
      PaddingConfig pad_config;
      for (int i = padding_dimensions.size() - 1; i >= 0; --i) {
        int64_t j = padding_dimensions[i];
        while (--dimension_index > j) {
          broadcast_shape1.DeleteDimension(dimension_index);
          pad_shape1.DeleteDimension(dimension_index);
        }
      }
      while (--dimension_index >= 0) {
        broadcast_shape1.DeleteDimension(dimension_index);
        pad_shape1.DeleteDimension(dimension_index);
      }
      for (auto dimension_to_pad : padding_dimensions) {
        auto dimension = pad_config.add_dimensions();
        *dimension = pad->padding_config().dimensions(dimension_to_pad);
      }
      *broadcast->mutable_shape() = broadcast_shape1;
      *broadcast->mutable_dimensions() = broadcast_dimensions;
      simplifier_->UpdateLayout(broadcast->mutable_shape());
      auto pad2 = pad->AddInstruction(pad->CloneWithNewShape(pad_shape1));
      *pad2->mutable_padding_config() = pad_config;
      simplifier_->UpdateLayout(pad2->mutable_shape());
      auto broadcast2 = pad->AddInstruction(
          HloInstruction::CreateBroadcast(pad_shape, pad2, padding_dimensions));
      return ReplaceInstruction(pad, broadcast2);
    }
  }

  if (has_negative && options_.enable_negative_padding_replacement()) {
    // Pad has negative padding. Replace with a pad with the non-negative
    // padding followed by a slice which effectively performs the negative
    // padding.
    // change kPad semantics to disallow negative padding and use slice
    // instead.

    // First construct the padding config with non-negative entries and the
    // compute the shape of this new pad instruction.
    PaddingConfig nonzero_padding = pad->padding_config();
    for (int i = 0; i < pad->padding_config().dimensions_size(); ++i) {
      PaddingConfig::PaddingConfigDimension* padding_dimension =
          nonzero_padding.mutable_dimensions(i);
      // Set negative padding to zero.
      if (padding_dimension->edge_padding_low() < 0) {
        padding_dimension->set_edge_padding_low(0);
      }
      if (padding_dimension->edge_padding_high() < 0) {
        padding_dimension->set_edge_padding_high(0);
      }
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * nonzero_pad,
                        MakePadHlo(pad->mutable_operand(0),
                                   pad->mutable_operand(1), nonzero_padding));
    // MakePadHlo assumes that the return type matches the type of the operand,
    // but that's not required. Use the type from the original pad instruction.
    nonzero_pad->mutable_shape()->set_element_type(pad->shape().element_type());

    // Copy the layout from the original pad instructions. The new pad and the
    // slice instruction should all have the same layout.
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        pad->shape(), nonzero_pad->mutable_shape()));
    simplifier_->UpdateLayout(nonzero_pad->mutable_shape());

    // Second, construct the slice instruction to perform the negative
    // padding.
    std::vector<int64_t> start_indices;
    std::vector<int64_t> end_indices;
    std::vector<int64_t> strides;
    for (int64_t i = 0; i < pad->padding_config().dimensions_size(); ++i) {
      const PaddingConfig::PaddingConfigDimension& padding_dimension =
          pad->padding_config().dimensions(i);
      int64_t start = 0;
      if (padding_dimension.edge_padding_low() < 0) {
        start = -1 * padding_dimension.edge_padding_low();
      }
      int64_t end = nonzero_pad->shape().dimensions(i);
      if (padding_dimension.edge_padding_high() < 0) {
        end += padding_dimension.edge_padding_high();
      }
      start_indices.push_back(start);
      end_indices.push_back(end);
      strides.push_back(1);
    }

    TF_ASSIGN_OR_RETURN(
        HloInstruction * slice,
        MakeSliceHlo(nonzero_pad, start_indices, end_indices, strides));
    TF_RETURN_IF_ERROR(LayoutUtil::CopyLayoutBetweenShapes(
        pad->shape(), slice->mutable_shape()));
    simplifier_->UpdateLayout(slice->mutable_shape());

    // Verify that the slice shape matches the pad shape.
    auto equal = Shape::Equal();
    if (!options_.is_layout_sensitive()) {
      equal.IgnoreTilesInLayout();
    }
    TF_RET_CHECK(equal(slice->shape(), pad->shape()));

    return ReplaceInstruction(pad, slice);
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  HloInstruction *lhs, *rhs;
  CHECK(Match(power, m::Power(m::Op(&lhs), m::Op(&rhs))));
  if (IsAll(rhs, 0)) {
    return ReplaceInstruction(power, MakeScalarLike(power, 1));
  }

  VLOG(10) << "trying transform [pow(A, 1) => A]: " << power->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(power, lhs)) {
    return absl::OkStatus();
  }

  // pow(exp(A),B) => exp(A*B)
  HloInstruction *a, *b;
  if (Match(power, m::Power(m::Exp(m::Op(&a)), m::Op(&b)))) {
    auto a_times_b = power->AddInstruction(HloInstruction::CreateBinary(
        power->shape(), HloOpcode::kMultiply, a, b));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateUnary(power->shape(), HloOpcode::kExp,
                                           a_times_b));
  }

  VLOG(10) << "trying transform [pow(A, 2) => A*A]: " << power->ToString();
  if (IsAll(rhs, 2)) {
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(),
                                            HloOpcode::kMultiply, lhs, lhs));
  }

  // Pow(A, 3) is used in GELU.
  VLOG(10) << "trying transform [pow(A, 3) => A*A*A]: " << power->ToString();
  if (IsAll(rhs, 3)) {
    HloInstruction* tmp = power->AddInstruction(HloInstruction::CreateBinary(
        power->shape(), HloOpcode::kMultiply, lhs, lhs));
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(),
                                            HloOpcode::kMultiply, lhs, tmp));
  }

  VLOG(10) << "trying transform [pow(A, -1) => 1/A]: " << power->ToString();
  if (IsAll(rhs, -1)) {
    return ReplaceWithNewInstruction(
        power, HloInstruction::CreateBinary(power->shape(), HloOpcode::kDivide,
                                            MakeScalarLike(lhs, 1), lhs));
  }

  VLOG(10) << "trying transform [pow(A, 0.5) => sqrt(A)], for A >= 0: "
           << power->ToString();
  if (IsAllFloat(rhs, 0.5) && IsNonNegative(lhs, options_)) {
    return ReplaceWithNewInstruction(
        power,
        HloInstruction::CreateUnary(power->shape(), HloOpcode::kSqrt, lhs));
  }

  return absl::OkStatus();
}

absl::StatusOr<bool>
AlgebraicSimplifierVisitor::TryToSinkBroadcastAfterElementwiseOps(
    HloInstruction* broadcast) {
  TF_RET_CHECK(broadcast->opcode() == HloOpcode::kBroadcast);
  bool changed = false;
  if (ShapeUtil::IsScalar(broadcast->shape())) {
    return false;
  }

  HloInstruction* operand = broadcast->mutable_operand(0);
  auto is_scalar_broadcast = [](const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kBroadcast &&
           ShapeUtil::IsScalar(instruction->operand(0)->shape());
  };
  auto is_equal_broadcast = [operand,
                             broadcast](const HloInstruction* instruction) {
    return instruction->opcode() == HloOpcode::kBroadcast &&
           ShapeUtil::Equal(operand->shape(),
                            instruction->operand(0)->shape()) &&
           broadcast->dimensions() == instruction->dimensions();
  };
  auto is_compatible_broadcast = [&](const HloInstruction* instruction) {
    return is_scalar_broadcast(instruction) || is_equal_broadcast(instruction);
  };

  for (HloInstruction* user : broadcast->users()) {
    if (user->IsDead()) {
      continue;
    }
    // Do not move reshapes or broadcasts past copies since the shape the copy
    // will operate on will change.
    if (user->opcode() == HloOpcode::kCopy) {
      continue;
    }
    // Do not change the shape of fusion nodes in case there a multiple shapes
    // inside the fusion node already.
    if (user->opcode() == HloOpcode::kFusion) {
      continue;
    }
    if (!user->IsElementwise()) {
      continue;
    }
    if (!absl::c_all_of(user->operands(), is_compatible_broadcast)) {
      continue;
    }

    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(user->operand_count());

    Shape changed_shape;
    for (HloInstruction* user_operand : user->operands()) {
      if (is_scalar_broadcast(user_operand)) {
        // If this is a broadcast from a scalar value rewrite a broadcast from
        // the scalar to the new shape enforced from the other broadcast
        // operands.
        changed_shape = ShapeUtil::ChangeElementType(
            operand->shape(), user_operand->shape().element_type());
        simplifier_->UpdateLayout(&changed_shape);
        new_operands.push_back(
            user_operand->AddInstruction(HloInstruction::CreateBroadcast(
                changed_shape, user_operand->mutable_operand(0), {})));
      } else {
        // For the non-scalar broadcasts, it is guaranteed that the shape of the
        // operand of the broadcast is a compatible shape.
        new_operands.push_back(user_operand->mutable_operand(0));
      }
    }
    VLOG(4) << "Sinking broadcast after user:"
            << "\n  old broadcast: " << broadcast->ToString()
            << "\n  old user: " << user->ToString();
    changed_shape = ShapeUtil::ChangeElementType(operand->shape(),
                                                 user->shape().element_type());
    simplifier_->UpdateLayout(&changed_shape);
    HloInstruction* new_user = user->AddInstruction(
        user->CloneWithNewOperands(changed_shape, new_operands));
    VLOG(4) << "  new user: " << new_user->ToString();
    HloInstruction* new_broadcast =
        broadcast->AddInstruction(HloInstruction::CreateBroadcast(
            user->shape(), new_user, broadcast->dimensions()));
    VLOG(4) << "  new broadcast: " << new_broadcast->ToString();
    TF_RETURN_IF_ERROR(user->ReplaceAllUsesWith(new_broadcast));
    changed = true;
  }
  return changed;
}

namespace {
template <typename T>
std::unique_ptr<HloInstruction> TryRemainderToAnd(
    HloInstruction* remainder, HloComputation* computation,
    AlgebraicSimplifier* simplifier) {
  HloInstruction *a, *b, *c;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  if (ShapeUtil::ElementIsIntegral(remainder->shape()) &&
      !Match(b, m::ConstantEffectiveScalar(&c)) &&
      !Match(b, m::Broadcast(m::ConstantEffectiveScalar(&c)))) {
    return nullptr;
  }

  if (ShapeUtil::ElementIsSigned(remainder->shape())) {
    int64_t b_value = static_cast<int64_t>(c->literal().GetFirstElement<T>());
    if (b_value > 0 && absl::has_single_bit(static_cast<uint64_t>(b_value))) {
      // Handle negative dividends by negating the result of the division.
      HloInstruction* zero_like_a = BroadcastZeros(computation, a->shape());

      Shape compare_shape = ShapeUtil::ChangeElementType(a->shape(), PRED);
      simplifier->UpdateLayout(&compare_shape);
      auto* dividend_is_negative =
          remainder->AddInstruction(HloInstruction::CreateCompare(
              compare_shape, a, zero_like_a, ComparisonDirection::kLt));

      auto* negated_dividend = remainder->AddInstruction(
          HloInstruction::CreateUnary(a->shape(), HloOpcode::kNegate, a));

      auto* abs_dividend =
          remainder->AddInstruction(HloInstruction::CreateTernary(
              a->shape(), HloOpcode::kSelect, dividend_is_negative,
              negated_dividend, a));

      auto* quotient = remainder->AddInstruction(HloInstruction::CreateBinary(
          remainder->shape(), HloOpcode::kAnd, abs_dividend,
          MakeScalarLike(abs_dividend, b_value - 1)));

      auto* neqated_quotient =
          remainder->AddInstruction(HloInstruction::CreateUnary(
              quotient->shape(), HloOpcode::kNegate, quotient));

      return HloInstruction::CreateTernary(
          remainder->shape(), HloOpcode::kSelect, dividend_is_negative,
          neqated_quotient, quotient);
    }
  } else {
    uint64_t b_value = static_cast<uint64_t>(c->literal().GetFirstElement<T>());
    if (absl::has_single_bit(b_value)) {
      HloInstruction* mask_amount =
          remainder->AddInstruction(simplifier->CreateConstantWithLayoutUpdated(
              LiteralUtil::CreateR0<T>(static_cast<T>(b_value - 1))));
      if (!ShapeUtil::IsScalar(b->shape())) {
        mask_amount = remainder->AddInstruction(
            HloInstruction::CreateBroadcast(b->shape(), mask_amount, {}));
      }
      return HloInstruction::CreateBinary(remainder->shape(), HloOpcode::kAnd,
                                          a, mask_amount);
    }
  }
  return nullptr;
}
}  // namespace

absl::Status AlgebraicSimplifierVisitor::HandleRemainder(
    HloInstruction* remainder) {
  HloInstruction *a, *b;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  // (A % B) % B == A % B.
  if (Match(a, m::Remainder(m::Op(), m::Op().Is(b)))) {
    return ReplaceInstruction(remainder, a);
  }

  // A % B => A & (B - 1) if B is a power of 2.
  if (std::unique_ptr<HloInstruction> shift =
          primitive_util::PrimitiveTypeSwitch<std::unique_ptr<HloInstruction>>(
              [&](auto kType) -> std::unique_ptr<HloInstruction> {
                if constexpr (primitive_util::IsIntegralType(kType)) {
                  using NativeT = primitive_util::NativeTypeOf<kType>;
                  return TryRemainderToAnd<NativeT>(remainder, computation_,
                                                    simplifier_);
                }
                return nullptr;
              },
              remainder->shape().element_type())) {
    return ReplaceWithNewInstruction(remainder, std::move(shift));
  }

  // If M < N, then {0, ..., M} % N ==> {0, ..., M}.
  //
  // Currently this only covers the case when N is a broadcasted constant
  // scalar.  We could also cover the case when N is a non-broadcasted constant
  // with the same value repeated.
  HloInstruction* iota;
  HloInstruction* divisor;
  if (Match(remainder,
            m::Remainder(m::Iota(&iota),
                         m::Broadcast(m::ConstantEffectiveScalar(&divisor))))) {
    // The iota counts {0, ..., iota_upper_bound - 1}.  (Actually this is
    // conservative; the iota may overflow and count up to a smaller value than
    // this.  But that's OK for our purposes here.)
    int64_t iota_upper_bound = iota->shape().dimensions(
        Cast<HloIotaInstruction>(iota)->iota_dimension());
    std::optional<int64_t> divisor_val = divisor->literal().GetIntegralAsS64(
        std::vector<int64_t>(0, divisor->shape().dimensions_size()));
    if (divisor_val && *divisor_val >= iota_upper_bound) {
      return ReplaceInstruction(remainder, iota);
    }
  }

  // (X + N) % N = X % N, so long as X + N does not overflow.
  //
  // We don't have range tracking in XLA that would let us know whether X + N
  // overflows, so for now we only do this simplification when X is an iota.  We
  // could add other operations where it's easy to see a range, such as
  // remainder, convert, etc., though at some point we'd probably want a
  // range-tracking analysis.
  HloInstruction* bcast;
  HloInstruction* addend;
  if (Match(
          remainder,
          m::Remainder(
              m::AddAnyOrder(m::Iota(&iota),
                             m::Broadcast(m::ConstantEffectiveScalar(&addend))),
              m::Broadcast(&bcast, m::ConstantEffectiveScalar(&divisor)))) &&
      addend == divisor) {
    // The iota counts {0, ...iota_upper_bound - 1}, with the same caveat above
    // that iota_upper_bound is conservative, and the true upper bound may be
    // smaller.
    int64_t iota_upper_bound = iota->shape().dimensions(
        Cast<HloIotaInstruction>(iota)->iota_dimension());
    std::optional<int64_t> divisor_val = divisor->literal().GetIntegralAsS64(
        std::vector<int64_t>(0, divisor->shape().dimensions_size()));
    if (divisor_val) {
      // Check whether divisor_val + iota_upper_bound - 1 overflows.
      std::optional<int64_t> max_val =
          OverflowSafeAdd(*divisor_val, iota_upper_bound);
      if (max_val.has_value() && primitive_util::FitsInIntegralType(
                                     *max_val, iota->shape().element_type())) {
        return ReplaceWithNewInstruction(
            remainder,
            HloInstruction::CreateBinary(remainder->shape(),
                                         HloOpcode::kRemainder, iota, bcast));
      }
    }
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleReshape(
    HloInstruction* reshape) {
  auto operand = reshape->mutable_operand(0);

  // Reshape directly to empty constant if the shape contains zero-element
  // dimension.
  if (ShapeUtil::IsZeroElementArray(reshape->shape())) {
    // If the instruction doesn't have a layout, use a default layout for
    // the literal result.
    Shape reshaped_shape = reshape->shape();
    if (!LayoutUtil::HasLayout(reshaped_shape)) {
      LayoutUtil::SetToDefaultLayout(&reshaped_shape);
    }
    auto empty_constant = simplifier_->CreateConstantWithLayoutUpdated(
        Literal::CreateFromShape(reshaped_shape));

    return ReplaceWithNewInstruction(reshape, std::move(empty_constant));
  }

  // Delete no-op reshapes, i.e. where shape = operand shape.
  if (SameShape(reshape, operand)) {
    VLOG(3) << "deleting no-op reshape";
    return ReplaceInstruction(reshape, operand);
  }

  // Merge reshapes.
  if (HloOpcode::kReshape == operand->opcode()) {
    return ReplaceWithNewInstruction(
        reshape, HloInstruction::CreateReshape(reshape->shape(),
                                               operand->mutable_operand(0)));
  }

  if (operand->opcode() == HloOpcode::kRng && operand->user_count() == 1) {
    *operand->mutable_shape() = reshape->shape();
    return ReplaceInstruction(reshape, operand);
  }

  if (options_.is_layout_sensitive()) {
    // Try to reorder copy-reshape to reshape-copy.
    HloInstruction* copy_before = nullptr;
    HloInstruction* copy_after = nullptr;
    if (Match(reshape,
              m::Reshape(m::Copy(&copy_before, m::Op()).WithOneUser()))) {
      if (reshape->user_count() == 1 &&
          reshape->users().front()->opcode() == HloOpcode::kCopy) {
        copy_after = reshape->users().front();
      }

      bool reshape_is_hardware_bitcast =
          options_.ReshapeIsBitcast(operand->shape(), reshape->shape());

      if (copy_before != nullptr) {
        if (auto aligned_input = ShapeUtil::AlignLayouts(
                copy_before->operand(0)->shape(), reshape->shape())) {
          // We now have the option to do reshape-copy instead of copy-reshape.
          Shape new_reshape_shape = std::move(*aligned_input);
          simplifier_->UpdateLayout(&new_reshape_shape);
          bool new_reshape_is_hardware_bitcast = options_.ReshapeIsBitcast(
              new_reshape_shape, copy_before->operand(0)->shape());

          bool should_rewrite = false;
          if (!reshape_is_hardware_bitcast) {
            if (new_reshape_is_hardware_bitcast) {
              // Can turn a reshape into a bitcast.
              should_rewrite = true;
            } else if (copy_before != nullptr) {
              // Neither reshapes are hardware bitcast.
              // Still can put two copies next to each other for a merge.
              should_rewrite = true;
            }
          } else if (new_reshape_is_hardware_bitcast) {
            if (copy_after != nullptr) {
              // Both reshapes are hardware bitcast.
              // Still can put two copies next to each other for a merge.
              should_rewrite = true;
            }
          }

          if (should_rewrite) {
            // If the "before copy" is from host memory, we cannot do this
            // rewrite.
            HloInstruction* copy_before_operand =
                copy_before->mutable_operand(0);
            if (copy_before_operand->shape().has_layout() &&
                copy_before_operand->shape().layout().memory_space() ==
                    Layout::kHostMemorySpace) {
              should_rewrite = false;
            }
          }

          if (should_rewrite) {
            // Can now cut down the number of ops. Make sure the memory usage
            // does not increase too much.
            int64_t total_shape_size_before_rewrite = 0;
            total_shape_size_before_rewrite +=
                ShapeUtil::ArraySize(copy_before->shape());
            if (!reshape_is_hardware_bitcast) {
              total_shape_size_before_rewrite +=
                  ShapeUtil::ArraySize(reshape->shape());
            }
            if (copy_after != nullptr) {
              total_shape_size_before_rewrite +=
                  ShapeUtil::ArraySize(copy_after->shape());
            }

            int64_t total_shape_size_after_rewrite = 0;
            if (!new_reshape_is_hardware_bitcast) {
              total_shape_size_after_rewrite +=
                  ShapeUtil::ArraySize(new_reshape_shape);
            }
            if (copy_after != nullptr) {
              total_shape_size_after_rewrite +=
                  ShapeUtil::ArraySize(copy_after->shape());
            } else {
              total_shape_size_after_rewrite +=
                  ShapeUtil::ArraySize(reshape->shape());
            }

            if (total_shape_size_after_rewrite >
                10 * total_shape_size_before_rewrite / 9) {
              should_rewrite = false;
            }
          }

          if (should_rewrite) {
            auto new_reshape = reshape->AddInstruction(
                new_reshape_is_hardware_bitcast
                    ? HloInstruction::CreateBitcast(
                          new_reshape_shape, copy_before->mutable_operand(0))
                    : HloInstruction::CreateReshape(
                          new_reshape_shape, copy_before->mutable_operand(0)));
            auto new_copy =
                Shape::Equal().IgnoreMemorySpaceInLayout()(reshape->shape(),
                                                           new_reshape->shape())
                    ? new_reshape
                    : reshape->AddInstruction(HloInstruction::CreateUnary(
                          reshape->shape(), HloOpcode::kCopy, new_reshape));
            VLOG(5) << "Replace copy-reshape with reshape-copy: "
                    << copy_before->ToString() << ", " << reshape->ToString()
                    << " => " << new_reshape->ToString() << ", "
                    << new_copy->ToString();
            if (copy_after != nullptr) {
              VLOG(5) << "Copy-after: " << copy_after->ToString();
            }
            return ReplaceInstruction(reshape, new_copy);
          }
        }
      }
    }
  }

  if (HloOpcode::kBroadcast == operand->opcode()) {
    auto opt_dims =
        ReshapeLeavesDimensionsUnmodified(reshape, operand->dimensions());
    if (opt_dims.has_value()) {
      return ReplaceWithNewInstruction(
          reshape,
          HloInstruction::CreateBroadcast(
              reshape->shape(), reshape->mutable_operand(0)->mutable_operand(0),
              *opt_dims));
    }
  }

  // reshape(iota) -> iota or a mixed radix calculation like
  // s32[2,3,4] reshape(s32[24] iota()) to
  // add(
  //    add(s32[2,3,4] iota() iota_dimension=2,
  //        4 * s32[2,3,4] iota() iota_dimension=1),
  //    12 * s32[2,3,4] iota() iota_dimension=0).
  if (operand->opcode() == HloOpcode::kIota) {
    auto* iota = Cast<HloIotaInstruction>(operand);
    auto common_factors = CommonFactors(operand->shape().dimensions(),
                                        reshape->shape().dimensions());
    auto iota_dim = absl::c_find_if(
        common_factors, [&](const std::pair<int64_t, int64_t>& dim_pair) {
          return dim_pair.first == iota->iota_dimension() &&
                 reshape->shape().dimensions(dim_pair.second) > 1;
        });
    auto next_dim = absl::c_find_if(
        common_factors, [&](const std::pair<int64_t, int64_t>& dim_pair) {
          return dim_pair.first == iota->iota_dimension() + 1;
        });
    if (iota_dim != common_factors.end() && next_dim != common_factors.end()) {
      int64_t multiplier = 1;
      HloInstruction* new_reshape = nullptr;

      for (int64_t dim = (iota_dim + 1)->second - 1; dim >= iota_dim->second;
           --dim) {
        HloInstruction* new_iota = iota->AddInstruction(
            HloInstruction::CreateIota(reshape->shape(), dim));
        if (new_reshape) {
          new_reshape = reshape->AddInstruction(HloInstruction::CreateBinary(
              reshape->shape(), HloOpcode::kAdd, new_reshape,
              reshape->AddInstruction(HloInstruction::CreateBinary(
                  reshape->shape(), HloOpcode::kMultiply, new_iota,
                  MakeScalarLike(reshape, multiplier)))));
        } else {
          new_reshape = new_iota;
        }
        multiplier *= reshape->shape().dimensions(dim);
      }
      return ReplaceInstruction(reshape, new_reshape);
    }
  }

  // Moves the reshape in reshape(dus(...), x, ...)) before dus so that it can
  // enable other optimizations, e.g., merging with broadcast, and sparse update
  // (add(x, dus(broadcast(0), y, ...)) -> dus(x, add(ds(x), y), ...)).
  if (!options_.is_layout_sensitive()) {
    HloInstruction* dus;
    HloInstruction* slice;
    std::optional<ShapeUtil::ShapeEqualityDescriptor> trivial_reshape =
        reshape->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
    // 1-sized dimensions added and removed will be one sized in both the update
    // slice and the dynamic-update-slice result.
    // Make sure dus has only one user; otherwise an extra copy is resulted.
    if (trivial_reshape.has_value() &&
        Match(reshape->mutable_operand(0),
              m::Op(&dus)
                  .WithOpcode(HloOpcode::kDynamicUpdateSlice)
                  .WithOperand(1, m::Op(&slice))) &&
        dus->user_count() == 1 && !dus->has_sharding() &&
        !dus->operand(0)->has_sharding()) {
      auto new_operand = reshape->AddInstruction(HloInstruction::CreateReshape(
          reshape->shape(), dus->mutable_operand(0)));
      std::vector<int64_t> new_slice_shape;
      std::vector<HloInstruction*> new_dus_operands;
      new_dus_operands.push_back(new_operand);
      new_dus_operands.push_back(nullptr);
      auto zero = MakeScalarLike(dus->mutable_operand(2), 0);
      const Shape& old_slice_shape = dus->operand(1)->shape();
      for (int64_t i = 0; i <= old_slice_shape.dimensions_size(); ++i) {
        if (absl::c_linear_search(trivial_reshape->deleted_dimensions, i)) {
          continue;
        }
        while (absl::c_linear_search(trivial_reshape->inserted_dimensions,
                                     new_slice_shape.size())) {
          new_slice_shape.push_back(1);
          new_dus_operands.push_back(zero);
        }
        if (i < old_slice_shape.dimensions_size()) {
          new_slice_shape.push_back(old_slice_shape.dimensions(i));
          new_dus_operands.push_back(dus->mutable_operand(2 + i));
        }
      }
      auto new_slice = reshape->AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(old_slice_shape.element_type(), new_slice_shape),
          slice));
      new_dus_operands[1] = new_slice;
      auto new_dus =
          dus->CloneWithNewOperands(reshape->shape(), new_dus_operands);
      return ReplaceWithNewInstruction(reshape, std::move(new_dus));
    }
  }

  // Make this a bitcast if possible.
  if (HloInstruction* bitcast_operand =
          BitcastingOperandOfReshapeOrCopyChain(reshape, options_)) {
    ReplaceWithBitcast(reshape, bitcast_operand);
  }
  return absl::OkStatus();
}

int64_t CountElementsLessThan(absl::Span<const int64_t> elements,
                              int64_t value) {
  int64_t low = 0;
  int64_t high = elements.size() - 1;
  int64_t count = 0;
  while (low <= high) {
    const int64_t mid = low + (high - low) / 2;
    if (elements.at(mid) < value) {
      count = mid + 1;
      low = mid + 1;
    } else {
      high = mid - 1;
    }
  }
  return count;
}

absl::Status AlgebraicSimplifierVisitor::HandleReverse(
    HloInstruction* reverse) {
  // When all the dimensions to reverse are trivial (i.e. the bound is 1),
  // there is nothing to be done.
  auto dim_is_one = [&](int64_t i) -> bool {
    return reverse->shape().dimensions(i) == 1;
  };
  if (absl::c_all_of(reverse->dimensions(), dim_is_one)) {
    return ReplaceInstruction(reverse, reverse->mutable_operand(0));
  }
  if (!options_.is_layout_sensitive()) {
    absl::Span<const int64_t> reverse_dims = reverse->dimensions();
    HloInstruction* inner = reverse->mutable_operand(0);
    HloOpcode inner_opcode = inner->opcode();
    // handling nested reverse
    // if two reverses are identical, both are removed, otherwise the
    // intersection of the dimensions of two reverses are removed
    if (inner_opcode == HloOpcode::kReverse) {
      absl::c_sort(*(reverse->mutable_dimensions()));
      absl::c_sort(*(inner->mutable_dimensions()));
      std::vector<int64_t> sym_diff, uni, intersect;
      absl::c_set_union(reverse_dims, inner->dimensions(),
                        std::back_inserter(uni));
      absl::c_set_intersection(reverse_dims, inner->dimensions(),
                               std::back_inserter(intersect));
      absl::c_set_difference(uni, intersect, std::back_inserter(sym_diff));
      if (sym_diff.empty()) {
        return ReplaceInstruction(reverse, inner->mutable_operand(0));
      }
      absl::Span<const int64_t> new_dimensions = absl::MakeConstSpan(sym_diff);
      return ReplaceInstruction(
          reverse, *MakeReverseHlo(inner->mutable_operand(0), new_dimensions));
    }
    // reverse(ElementWiseBinOp(x, constant)) ==>
    // ElementWiseBinOp(reverse(x), constant)
    // element-wise binary op inside reverse can be brought out
    auto match_with_scalar = [&](HloInstruction* broadcast) -> HloInstruction* {
      if (broadcast->opcode() == HloOpcode::kBroadcast &&
          broadcast->dimensions().empty() &&
          ShapeUtil::IsScalar(broadcast->operand(0)->shape())) {
        return broadcast->mutable_operand(0);
      }
      return nullptr;
    };
    if (inner->IsElementwiseBinary()) {
      // produces incorrect result for rng.
      if (inner->opcode() == HloOpcode::kRng ||
          inner_opcode == HloOpcode::kCompare) {
        return absl::OkStatus();
      }
      HloInstruction* hlo;
      if (match_with_scalar(inner->mutable_operand(0))) {
        hlo = inner->mutable_operand(1);
        return ReplaceWithNewInstruction(
            reverse,
            HloInstruction::CreateBinary(inner->shape(), inner_opcode,
                                         inner->mutable_operand(0),
                                         *MakeReverseHlo(hlo, reverse_dims)));
      } else if (match_with_scalar(inner->mutable_operand(1))) {
        hlo = inner->mutable_operand(0);
        return ReplaceWithNewInstruction(
            reverse,
            HloInstruction::CreateBinary(inner->shape(), inner_opcode,
                                         *MakeReverseHlo(hlo, reverse_dims),
                                         inner->mutable_operand(1)));
      } else {
        return absl::OkStatus();
      }
    }
    // reverse(DegenerateDimensionAddingReshape(x)) ==>
    // DegenerateDimensionAddingReshape(reverse(x))
    // degenerate adding reshape inside a reverse can be brought out
    if (inner_opcode == HloOpcode::kReshape) {
      Shape* inner_shape = inner->mutable_shape();
      // degenerate adding reshape check
      std::optional<ShapeUtil::ShapeEqualityDescriptor> reshape_degenerate =
          inner->ReshapeMerelyInsertsOrDeletes1SizedDimensions();
      if (reshape_degenerate.has_value() &&
          reshape_degenerate->deleted_dimensions.empty()) {
        std::vector<int64_t> new_reverse_dims;
        // for each reverse dimension dim, count the number of degenerate
        // dimensions that are added 'before' dim by the reshape operation.
        for (auto dim : reverse_dims) {
          // trivial dimensions don't need to be reversed.
          if (inner_shape->dimensions(dim) == 1) {
            continue;
          }
          auto new_dim =
              dim -
              CountElementsLessThan(
                  absl::MakeConstSpan(reshape_degenerate->inserted_dimensions),
                  dim);
          new_reverse_dims.push_back(new_dim);
        }

        return ReplaceInstruction(
            reverse,
            *MakeReshapeHlo(
                *inner_shape,
                *MakeReverseHlo(reverse->mutable_operand(0)->mutable_operand(0),
                                new_reverse_dims)));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::TrySimplifyScalarSlice(
    HloInstruction* slice) {
  // Only try to do this for effective scalars. We could do the same for slicing
  // out larger pieces of padding (replacing with a broadcast of the padding
  // value), but this is probably not worth it.
  if (!ShapeUtil::IsEffectiveScalar(slice->shape())) {
    return false;
  }

  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate) {
    VLOG(10) << "Trying to simplify scalar slice of concat";
    // Only do this for R1, there's no chance of this being useful otherwise.
    if (slice->shape().dimensions_size() != 1) {
      VLOG(10) << "Not folding, slice is not rank 1";
      return false;
    }
    HloConcatenateInstruction* concat =
        Cast<HloConcatenateInstruction>(slice->mutable_operand(0));
    int64_t operand_start = 0;
    int64_t operand_num = 0;
    // Weird loop structure to avoid annoying off-by-one errors.
    while (true) {
      TF_RET_CHECK(operand_num < concat->operand_count());
      const HloInstruction* operand = concat->operand(operand_num);
      int64_t next_operand_start =
          operand_start + operand->shape().dimensions(0);
      if (next_operand_start > slice->slice_starts(0)) {
        break;
      }
      operand_start = next_operand_start;
      operand_num++;
    }

    bool replaced = ReplaceInstructionIfCompatible(
        slice, concat->mutable_operand(operand_num));
    if (replaced) {
      VLOG(10) << "Folding scalar slice of concat into concat operand";
    } else {
      VLOG(10) << "Folding scalar slice of concat into slice of concat operand";
      TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
          slice, HloInstruction::CreateSlice(
                     slice->shape(), concat->mutable_operand(operand_num),
                     {slice->slice_starts(0) - operand_start},
                     {slice->slice_starts(0) - operand_start + 1},
                     slice->slice_strides())));
    }
    return true;
  }

  return false;
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::TryToReorderSliceAndReshape(
    HloInstruction* slice) {
  CHECK_EQ(slice->opcode(), HloOpcode::kSlice);
  if (!hlo_instruction_utils::IsUnstridedSlice(slice)) {
    return false;
  }
  HloInstruction* reshape = slice->mutable_operand(0);
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  HloInstruction* new_slice_operand = reshape->mutable_operand(0);
  int64_t slice_rank = slice->shape().dimensions_size();
  std::vector<int64_t> sliced_dims;
  for (int64_t i = 0; i < slice_rank; ++i) {
    if (slice->slice_starts(i) != 0 ||
        slice->slice_limits(i) != reshape->shape().dimensions(i)) {
      sliced_dims.push_back(i);
    }
  }

  if (sliced_dims.size() == 1 && sliced_dims[0] == 0 &&
      slice->slice_starts(0) == 0) {
    const Shape& new_slice_shape = new_slice_operand->shape();
    const int64_t rank = new_slice_shape.dimensions_size();
    std::vector<int64_t> new_slice_starts(rank, 0);
    std::vector<int64_t> new_slice_stides(rank, 1);
    std::vector<int64_t> new_slice_limits(new_slice_shape.dimensions().begin(),
                                          new_slice_shape.dimensions().end());
    int64_t slice_elements = ShapeUtil::ElementsIn(slice->shape());
    for (int64_t i = rank - 1; i >= 0; --i) {
      if (slice_elements >= new_slice_limits[i]) {
        if (slice_elements % new_slice_limits[i] != 0) {
          return false;
        }
        slice_elements /= new_slice_limits[i];
      } else {
        new_slice_limits[i] = slice_elements;
        slice_elements = 1;
      }
    }
    HloInstruction* new_slice =
        slice->AddInstruction(HloInstruction::CreateSlice(
            ShapeUtil::MakeShape(new_slice_shape.element_type(),
                                 new_slice_limits),
            new_slice_operand, new_slice_starts, new_slice_limits,
            new_slice_stides));
    simplifier_->UpdateLayout(new_slice->mutable_shape());
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        slice, HloInstruction::CreateReshape(slice->shape(), new_slice)));
    return true;
  }
  return false;
}

// Allowing a slice to move through a reverse with any necessary updates to the
// slice config.
absl::StatusOr<bool> AlgebraicSimplifierVisitor::TryToReorderSliceAndReverse(
    HloInstruction* slice) {
  VLOG(2) << "Entered TryToReorderSliceAndReverse for slice:"
          << slice->ToString();
  if (Match(slice, m::Slice(m::Reverse()))) {
    HloInstruction* reverse = slice->mutable_operand(0);
    HloInstruction* reverse_operand = reverse->mutable_operand(0);
    std::vector<int64_t> new_starts = slice->slice_starts();
    std::vector<int64_t> new_limits = slice->slice_limits();
    std::vector<int64_t> new_strides = slice->slice_strides();
    for (auto rdim : reverse->dimensions()) {
      int64_t start = slice->slice_starts(rdim);
      int64_t limit = slice->slice_limits(rdim);
      int64_t stride = slice->slice_strides(rdim);
      // find_nth allows us to compute the appropriate index to begin
      // with during reverse even in the presence of non-unit strides
      int64_t find_nth = (limit - start - 1) / stride;
      find_nth = start + find_nth * stride;
      limit = find_nth + 1;
      new_starts[rdim] =
          (reverse->shape().dimensions(rdim) - start) - (limit - start);
      new_limits[rdim] = reverse->shape().dimensions(rdim) - start;
      VLOG(2) << "Analyzing dim:" << rdim << " (start,limit):" << start << ","
              << limit << " and new (start, limit):" << new_starts[rdim] << ","
              << new_limits[rdim];
    }
    // New slice formed from the reverse_operand, but strides and shape of the
    // slice output remains the same. New slice's starts and limits are updated
    // for ONLY the reversed dimensions as indicated above.
    HloInstruction* new_slice = slice->AddInstruction(
        HloInstruction::CreateSlice(slice->shape(), reverse_operand, new_starts,
                                    new_limits, new_strides));
    simplifier_->UpdateLayout(new_slice->mutable_shape());
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        slice, HloInstruction::CreateReverse(new_slice->shape(), new_slice,
                                             reverse->dimensions())));
    // We do not delete the old reverse, since there might be another
    // consumer of that reverse (i.e., full reverse output). DCE should take
    // care of any deletion that is necessary if there was no use of reverse.
    return true;
  }
  return false;
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::RemoveRedundantStride(
    absl::Nonnull<HloInstruction*> slice) {
  CHECK(slice->opcode() == HloOpcode::kSlice);

  std::vector<int64_t> index_to_change;
  for (int64_t i = 0; i < slice->shape().dimensions_size(); ++i) {
    const int64_t start = slice->slice_starts(i);
    const int64_t stride = slice->slice_strides(i);
    const int64_t limit = slice->slice_limits(i);

    if (stride == 1) {
      // Nothing to update.
      continue;
    }

    if (stride >= limit || start + stride >= limit) {
      index_to_change.push_back(i);
    }
  }

  if (index_to_change.empty()) {
    return false;
  }

  std::vector<int64_t> new_slice_limits = slice->slice_limits();
  std::vector<int64_t> new_slice_strides = slice->slice_strides();
  for (int64_t index : index_to_change) {
    new_slice_limits[index] = slice->slice_starts(index) + 1;
    new_slice_strides[index] = 1;
  }

  HloInstruction* slice_operand = slice->mutable_operand(0);
  TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
      slice, HloInstruction::CreateSlice(slice->shape(), slice_operand,
                                         slice->slice_starts(),
                                         new_slice_limits, new_slice_strides)));
  return true;
}

absl::Status AlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice) {
  // Delete no-op slices, i.e. where shape = operand shape.
  if (ReplaceInstructionIfCompatible(slice, slice->mutable_operand(0))) {
    return absl::OkStatus();
  }

  HloInstruction* pad;
  HloInstruction* pad_operand;
  if (Match(slice, m::Slice(m::Pad(&pad, m::Op(&pad_operand), m::Op())))) {
    // Is the result of the slice the pad operand.
    bool slice_undoes_pad = true;
    // Can the slice be moved to the pad_operand without any padding being read.
    bool slice_inside_pad = true;
    // Does this slice slice out pading only.
    bool slice_in_padding = false;
    std::vector<int64_t> new_starts = slice->slice_starts();
    std::vector<int64_t> new_limits = slice->slice_limits();
    for (int64_t i = 0; i < slice->shape().dimensions_size(); ++i) {
      const int64_t start = slice->slice_starts(i);
      const int64_t stride = slice->slice_strides(i);
      const int64_t limit = slice->slice_limits(i);
      const int64_t size = pad->shape().dimensions(i);

      const auto& dim = pad->padding_config().dimensions(i);
      const int64_t low = dim.edge_padding_low();
      const int64_t high = dim.edge_padding_high();
      const int64_t interior = dim.interior_padding();
      const int64_t edge = size - high;

      if (limit <= low || start >= edge) {
        slice_in_padding = true;
        break;
      }

      if (start != low || stride - 1 != interior) {
        slice_undoes_pad = false;
      }

      if (start < low || limit > edge || interior != 0 || stride != 1) {
        slice_inside_pad = false;
      }
      new_starts[i] -= low;
      new_limits[i] -= low;
    }
    if (slice_in_padding) {
      HloInstruction* broadcast =
          MakeBroadcastHlo(pad->mutable_operand(1), {}, slice->shape());
      *(broadcast->mutable_shape()) = slice->shape();
      return ReplaceInstruction(slice, broadcast);
    }
    if (slice_undoes_pad &&
        ReplaceInstructionIfCompatible(slice, pad_operand)) {
      return absl::OkStatus();
    }
    if (slice_inside_pad) {
      TF_ASSIGN_OR_RETURN(HloInstruction * new_slice,
                          MakeSliceHlo(pad_operand, new_starts, new_limits,
                                       slice->slice_strides()));
      *(new_slice->mutable_shape()) = slice->shape();
      return ReplaceInstruction(slice, new_slice);
    }
  }

  if (slice->operand(0)->opcode() == HloOpcode::kSlice &&
      hlo_instruction_utils::IsUnstridedSlice(slice) &&
      hlo_instruction_utils::IsUnstridedSlice(slice->operand(0))) {
    HloInstruction* operand_slice = slice->mutable_operand(0);
    std::vector<int64_t> new_slice_starts = slice->slice_starts();
    std::vector<int64_t> new_slice_limits = slice->slice_limits();
    for (int64_t i = 0; i < new_slice_starts.size(); ++i) {
      new_slice_starts[i] += operand_slice->slice_starts(i);
      new_slice_limits[i] += operand_slice->slice_starts(i);
    }
    return ReplaceWithNewInstruction(
        slice, HloInstruction::CreateSlice(
                   slice->shape(), operand_slice->mutable_operand(0),
                   new_slice_starts, new_slice_limits, slice->slice_strides()));
  }

  auto only_broadcast_dims_sliced = [&] {
    if (slice->operand(0)->opcode() != HloOpcode::kBroadcast) {
      return false;
    }
    for (int64_t dim : slice->operand(0)->dimensions()) {
      if (slice->slice_starts(dim) != 0 || slice->slice_strides(dim) != 1 ||
          slice->slice_limits(dim) !=
              slice->operand(0)->shape().dimensions(dim)) {
        return false;
      }
    }
    return true;
  };
  if (only_broadcast_dims_sliced()) {
    return ReplaceWithNewInstruction(
        slice,
        HloInstruction::CreateBroadcast(
            slice->shape(), slice->mutable_operand(0)->mutable_operand(0),
            slice->mutable_operand(0)->dimensions()));
  }

  TF_ASSIGN_OR_RETURN(bool replaced, TrySimplifyScalarSlice(slice));
  if (replaced) {
    return absl::OkStatus();
  }

  HloInstruction* broadcast;
  HloInstruction* broadcast_operand;
  if (Match(slice,
            m::Slice(m::Broadcast(&broadcast, m::Op(&broadcast_operand))))) {
    std::vector<int64_t> new_slice_starts;
    std::vector<int64_t> new_slice_strides;
    std::vector<int64_t> new_slice_limits;
    new_slice_starts.reserve(broadcast_operand->shape().dimensions_size());
    new_slice_strides.reserve(broadcast_operand->shape().dimensions_size());
    new_slice_limits.reserve(broadcast_operand->shape().dimensions_size());
    for (int64_t dim : broadcast->dimensions()) {
      new_slice_starts.push_back(slice->slice_starts(dim));
      new_slice_strides.push_back(slice->slice_strides(dim));
      new_slice_limits.push_back(slice->slice_limits(dim));
    }
    VLOG(3) << "Sink broadcast through slice";
    VLOG(3) << "Original slice: " << slice->ToString();
    VLOG(3) << "Original broadcast: " << broadcast->ToString();
    auto new_slice_shape = broadcast_operand->shape();
    for (int64_t i = 0; i < broadcast_operand->shape().dimensions_size(); ++i) {
      int64_t size_i = (new_slice_limits[i] - new_slice_starts[i] +
                        new_slice_strides[i] - 1) /
                       new_slice_strides[i];
      new_slice_shape.set_dimensions(i, size_i);
    }
    simplifier_->UpdateLayout(&new_slice_shape);
    auto new_slice = slice->AddInstruction(HloInstruction::CreateSlice(
        new_slice_shape, broadcast_operand, new_slice_starts, new_slice_limits,
        new_slice_strides));
    auto new_broadcast =
        broadcast->AddInstruction(HloInstruction::CreateBroadcast(
            slice->shape(), new_slice, broadcast->dimensions()));
    VLOG(3) << "New slice: " << slice->ToString();
    VLOG(3) << "New broadcast: " << new_broadcast->ToString();
    return ReplaceInstruction(slice, new_broadcast);
  }

  // Try to reorder slice of dot to the operand it comes from
  if (!options_.is_layout_sensitive() &&
      options_.raise_slice_and_reduce_through_dot() &&
      slice->operand(0)->opcode() == HloOpcode::kDot) {
    // Unpack the dot operands
    HloDotInstruction* dot = Cast<HloDotInstruction>(slice->mutable_operand(0));
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);
    DotDimensionNumbers dnums = dot->dot_dimension_numbers();

    // Construct map from lhs and rhs dimensions to dot dimensions
    std::vector<int64_t> map_lhs_dot, map_rhs_dot;
    std::tie(map_lhs_dot, map_rhs_dot) =
        ConstructToDotMaps(dnums, lhs->shape(), rhs->shape());

    // We use these booleans to keep track of where to add slice instructions
    bool slice_lhs = false;
    bool slice_rhs = false;

    // Sparse metadata may need to be sliced.
    std::array<HloInstruction*, 2> sparse_meta = {nullptr, nullptr};
    for (int i = 0; i < dot->sparse_operands(); ++i) {
      const SparsityDescriptor& descriptor = dot->sparsity()[i];
      sparse_meta[descriptor.index()] =
          dot->mutable_operand(HloDotInstruction::kOperands + i);
    }
    auto slice_meta = [&](const DimensionVector& operand_start_indices,
                          const DimensionVector& operand_limit_indices,
                          const DimensionVector& operand_strides,
                          HloInstruction* meta, int dimension) {
      DimensionVector start_indices, limit_indices, strides;
      for (int64_t i = 0; i < meta->shape().dimensions_size(); ++i) {
        start_indices.push_back(operand_start_indices[i]);
        limit_indices.push_back(i != dimension ? operand_limit_indices[i]
                                               : meta->shape().dimensions(i));
        strides.push_back(operand_strides[i]);
      }
      return MakeSliceHlo(meta, start_indices, limit_indices, strides);
    };

    // Here we build up the slice dimensions for lhs
    DimensionVector lhs_start_indices, lhs_limit_indices, lhs_strides;
    for (int64_t lhs_index = 0; lhs_index < lhs->shape().dimensions_size();
         ++lhs_index) {
      int64_t size = lhs->shape().dimensions(lhs_index);
      // If it is not a contracting dimension, we slice it according to the
      // slicing of the corresponding dimension in dot
      int64_t i = map_lhs_dot[lhs_index];
      int64_t start = i >= 0 ? slice->slice_starts(i) : 0;
      int64_t limit = i >= 0 ? slice->slice_limits(i) : size;
      int64_t stride = i >= 0 ? slice->slice_strides(i) : 1;
      lhs_start_indices.push_back(start);
      lhs_limit_indices.push_back(limit);
      lhs_strides.push_back(stride);
      // Record if any slicing occurs here
      bool update = start != 0 || limit < size || stride != 1;
      slice_lhs |= update;
    }

    // Here we do the same for rhs
    DimensionVector rhs_start_indices, rhs_limit_indices, rhs_strides;
    for (int64_t rhs_index = 0; rhs_index < rhs->shape().dimensions_size();
         ++rhs_index) {
      int64_t size = rhs->shape().dimensions(rhs_index);
      // If it is not a contracting dimension, we slice it according to the
      // slicing of the corresponding dimension in dot
      int64_t i = map_rhs_dot[rhs_index];
      int64_t start = i >= 0 ? slice->slice_starts(i) : 0;
      int64_t limit = i >= 0 ? slice->slice_limits(i) : size;
      int64_t stride = i >= 0 ? slice->slice_strides(i) : 1;
      rhs_start_indices.push_back(start);
      rhs_limit_indices.push_back(limit);
      rhs_strides.push_back(stride);
      // Record if any slicing occurs here
      bool update = start != 0 || limit < size || stride != 1;
      slice_rhs |= update;
    }

    // Create Hlo for new slices
    HloInstruction* new_lhs = lhs;
    HloInstruction* new_rhs = rhs;
    if (slice_lhs) {
      TF_ASSIGN_OR_RETURN(
          new_lhs,
          MakeSliceHlo(lhs, lhs_start_indices, lhs_limit_indices, lhs_strides));
    }
    if (slice_rhs) {
      TF_ASSIGN_OR_RETURN(
          new_rhs,
          MakeSliceHlo(rhs, rhs_start_indices, rhs_limit_indices, rhs_strides));
    }

    // Create Hlo for new metadata (for sparse dot)
    std::vector<SparsityDescriptor> new_sparsity;
    std::vector<HloInstruction*> new_meta;
    if (dot->sparse_operands()) {
      if (auto& lhs = dot->sparsity().front(); lhs.index() == 0) {
        if (slice_lhs) {
          TF_ASSIGN_OR_RETURN(
              sparse_meta[0],
              slice_meta(lhs_start_indices, lhs_limit_indices, lhs_strides,
                         sparse_meta[0], lhs.dimension()));
        }
        new_sparsity.push_back(lhs);
        new_meta.push_back(sparse_meta[0]);
      }
      if (auto& rhs = dot->sparsity().back(); rhs.index() == 1) {
        if (slice_rhs) {
          TF_ASSIGN_OR_RETURN(
              sparse_meta[1],
              slice_meta(rhs_start_indices, rhs_limit_indices, rhs_strides,
                         sparse_meta[1], rhs.dimension()));
        }
        new_sparsity.push_back(rhs);
        new_meta.push_back(sparse_meta[1]);
      }
    }

    // Finally, create Hlo for the new dot and reorder
    TF_ASSIGN_OR_RETURN(
        HloInstruction * new_dot,
        MakeDotHlo(new_lhs, new_rhs, dnums, dot->precision_config(),
                   dot->shape().element_type(), new_sparsity, new_meta));

    // We should only do this reorder if both new_lhs and new_rhs have free
    // dimensions. Otherwise, it will conflict with an existing optimization
    // that converts dot to mul(broadcast)
    if (!DotHasOnlyBatchAndContractingOnOneOperand(
            ShapeUtil::TrueNumDimensions(new_lhs->shape()),
            ShapeUtil::TrueNumDimensions(new_rhs->shape()), dnums)) {
      VLOG(10) << "Reordering slice into dot operands";
      return ReplaceInstruction(slice, new_dot);
    }
  }

  // Try to simplify concat -> slice to an operand of concat.
  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate &&
      hlo_instruction_utils::IsUnstridedSlice(slice)) {
    HloInstruction* concat = slice->mutable_operand(0);
    int64_t concat_dim = concat->concatenate_dimension();
    int64_t piece_start = 0;
    std::optional<int64_t> start_operand;
    std::optional<int64_t> limit_operand;
    int64_t concat_start;
    int64_t concat_limit;
    const int64_t slice_start = slice->slice_starts(concat_dim);
    const int64_t slice_limit = slice->slice_limits(concat_dim);
    for (int64_t i = 0; i < concat->operand_count(); ++i) {
      const HloInstruction* piece = concat->operand(i);
      const int64_t piece_size = piece->shape().dimensions(concat_dim);
      if (!start_operand && piece_start <= slice_start &&
          piece_size + piece_start > slice_start) {
        start_operand = i;
        concat_start = piece_start;
      }
      piece_start += piece_size;
      if (!limit_operand && piece_start >= slice_limit) {
        limit_operand = i + 1;
        concat_limit = piece_start;
        break;
      }
    }
    if (start_operand && limit_operand &&
        *start_operand + 1 == *limit_operand &&
        SameShape(concat->operand(*start_operand), slice)) {
      return ReplaceInstruction(slice, concat->mutable_operand(*start_operand));
    }
    if (start_operand && limit_operand &&
        *limit_operand - *start_operand < concat->operand_count()) {
      std::vector<int64_t> starts = slice->slice_starts();
      starts[concat_dim] = starts[concat_dim] - concat_start;
      std::vector<int64_t> strides = slice->slice_strides();
      std::vector<int64_t> limits = slice->slice_limits();
      limits[concat_dim] =
          starts[concat_dim] + slice->shape().dimensions(concat_dim);
      HloInstruction* operand = concat->mutable_operand(*start_operand);
      if (*start_operand + 1 != *limit_operand) {
        TF_ASSIGN_OR_RETURN(
            HloInstruction * new_concat,
            MakeConcatHlo(
                absl::MakeSpan(concat->operands())
                    .subspan(*start_operand, *limit_operand - *start_operand),
                concat_dim));
        *new_concat->mutable_shape()->mutable_layout() =
            concat->shape().layout();
        simplifier_->UpdateLayout(new_concat->mutable_shape());
        concat->SetupDerivedInstruction(new_concat);
        operand = new_concat;
      }
      return ReplaceWithNewInstruction(
          slice, HloInstruction::CreateSlice(slice->shape(), operand, starts,
                                             limits, strides));
    }
  }

  if (HloInstruction* reduce_window;
      options_.enable_window_reduce_to_reduce_replacement() &&
      hlo_instruction_utils::IsUnstridedSlice(slice) &&
      Match(slice, m::Slice(m::ReduceWindow(&reduce_window).WithOneUse()))) {
    // A reduce_window with window pad + slice[:,-1] can be expressed as
    // reduce + reshape if all dimensions either have a window size of one or
    // the entire dimension. No stride or dilation are expected. reduce_window
    // pad should be present to make output shape equal to input shape.
    // slice_limit[dim] should be equal to reduce_window shape[dim].
    // slice_limit[dim] - slice_start[dim] should be equal to 1 for reduced dim
    //
    // The reshape is a bitcast since it adds one-sized dimensions. Often
    // these ones are immediately removed as well with another reshape. The
    // implementation of reduce tends to be slightly more efficient at
    // reducing entire dimensions compared to reduce window.
    //
    //// Example 1:
    // r = s32[2,8] reduce-window(s32[2,8] p, c), window={size=1x8 pad=0_0x7_0}
    // s = s32[2,1] slice(r), slice={[0:2], [7:8]}
    //// Can be folded to:
    // r = s32[2] reduce(s32[2,8] p, c), dimensions={1},
    // s = s32[2] reshape(r)
    //
    //// Example 2:
    // p = s32[3,4,2]
    // r = s32[3,4,2] reduce-window(p, c), window={size=1x4x2 pad=0_0x3_0x1_0}
    // s = s32[3,1,1] slice(r), slice={[0:3], [3:4], [1:2]}
    //// Can be folded to:
    // r = s32[3] reduce(p, c), dimensions={1,2},
    // s = s32[3,1,1] reshape(r)
    auto effective_reduce_dims = [&] {
      auto& window = reduce_window->window();
      // reduce_window should have padding, but no Strides and Dilation
      if (window_util::HasStride(window) || window_util::HasDilation(window) ||
          !window_util::HasPadding(window)) {
        return DimensionVector{};
      }
      auto rank = reduce_window->shape().dimensions_size();
      auto& slice_starts = slice->slice_starts();
      auto& slice_limits = slice->slice_limits();
      DimensionVector reduce_dims;
      for (auto i = 0; i < rank; ++i) {
        auto window_dim_size = window.dimensions(i).size();
        auto reduce_window_dim_size = reduce_window->shape().dimensions(i);
        auto slice_dim_size = slice->shape().dimensions(i);
        if (reduce_window_dim_size != slice_limits[i] ||
            window.dimensions(i).padding_low() != slice_starts[i] ||
            window.dimensions(i).padding_high() != 0) {
          return DimensionVector{};
        }
        if (window_dim_size == 1 && reduce_window_dim_size == slice_dim_size &&
            slice_starts[i] == 0) {
          continue;
        }
        if (slice_dim_size == 1 && reduce_window_dim_size == window_dim_size &&
            slice_limits[i] - slice_starts[i] == 1) {
          reduce_dims.push_back(i);
        } else {
          return DimensionVector{};
        }
      }
      return reduce_dims;
    }();

    // If a reduce window can be expressed as a reduce, do so and reshape the
    // output.
    if (!effective_reduce_dims.empty()) {
      Shape reduce_shape = ShapeUtil::DeleteDimensions(effective_reduce_dims,
                                                       reduce_window->shape());
      simplifier_->UpdateLayout(&reduce_shape);
      HloInstruction* reduce =
          slice->AddInstruction(HloInstruction::CreateReduce(
              /*shape=*/reduce_shape,
              /*operand=*/reduce_window->mutable_operand(0),
              /*init_value=*/reduce_window->mutable_operand(1),
              /*dimensions_to_reduce=*/effective_reduce_dims,
              /*reduce_computation=*/reduce_window->to_apply()));
      return ReplaceWithNewInstruction(
          slice, HloInstruction::CreateReshape(slice->shape(), reduce));
    }
  }

  // Do not try to reorder slices and reshapes after layout assignment as it may
  // be invalid.
  if (!options_.is_layout_sensitive()) {
    TF_ASSIGN_OR_RETURN(replaced, TryToReorderSliceAndReshape(slice));
  }
  if (replaced) {
    return absl::OkStatus();
  }

  bool reversed = false;
  if (Match(slice, m::Slice(m::Reverse(m::Op())))) {
    TF_ASSIGN_OR_RETURN(reversed, TryToReorderSliceAndReverse(slice));
  }
  if (reversed) {
    return absl::OkStatus();
  }

  TF_ASSIGN_OR_RETURN(bool removed_redundant_stride,
                      RemoveRedundantStride(slice));
  if (removed_redundant_stride) {
    VLOG(10) << "Removed redundant stride for slice op.";
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleRsqrt(HloInstruction* rsqrt) {
  VLOG(10) << "trying transform [rsqrt(pow(A, -2)) => A], for A >= 0 "
           << rsqrt->ToString();
  HloInstruction* rsqrt_operand = rsqrt->mutable_operand(0);
  if (rsqrt_operand->opcode() == HloOpcode::kPower &&
      IsAll(rsqrt_operand->operand(1), -2) &&
      IsNonNegative(rsqrt_operand->operand(0), options_)) {
    return ReplaceInstruction(rsqrt, rsqrt_operand->mutable_operand(0));
  }

  VLOG(10) << "trying transform [rsqrt(1/A)) => sqrt(A)], for A >= 0 "
           << rsqrt->ToString();
  if (rsqrt_operand->opcode() == HloOpcode::kDivide &&
      IsAll(rsqrt_operand->operand(0), 1) &&
      IsNonNegative(rsqrt_operand->operand(1), options_)) {
    return ReplaceWithNewInstruction(
        rsqrt, HloInstruction::CreateUnary(rsqrt->shape(), HloOpcode::kSqrt,
                                           rsqrt_operand->mutable_operand(1)));
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
  // Skip optimizations for async dynamic-slices.
  if (dynamic_slice->parent()->IsAsyncComputation()) {
    return absl::OkStatus();
  }
  auto operand = dynamic_slice->mutable_operand(0);
  if (ShapeUtil::IsScalar(dynamic_slice->shape())) {
    return ReplaceInstruction(dynamic_slice, operand);
  }
  // DynamicSlice where operand has the same size as the output is simply equal
  // to operand.
  if (SameShape(operand, dynamic_slice)) {
    return ReplaceInstruction(dynamic_slice, operand);
  }

  // DynamicSlice clamps the offset. If the slice size has the same size on a
  // dim as the operand, we can replace it with zero.
  std::vector<int> same_size_dims_to_simplify;
  for (int64_t dim = 0; dim < operand->shape().dimensions_size(); ++dim) {
    if (!(dynamic_slice->operand(dim + 1)->IsConstant() &&
          IsAll(dynamic_slice->operand(dim + 1), 0)) &&
        operand->shape().dimensions(dim) ==
            dynamic_slice->shape().dimensions(dim)) {
      same_size_dims_to_simplify.push_back(dim);
    }
  }
  if (!same_size_dims_to_simplify.empty()) {
    HloInstruction* zero = MakeScalarLike(dynamic_slice->mutable_operand(1), 0);
    auto new_operands = dynamic_slice->mutable_operands();
    for (int64_t dim : same_size_dims_to_simplify) {
      new_operands[dim + 1] = zero;
    }
    return ReplaceInstruction(
        dynamic_slice,
        dynamic_slice->AddInstruction(dynamic_slice->CloneWithNewOperands(
            dynamic_slice->shape(), new_operands)));
  }

  HloInstruction* broadcast_operand;
  if (Match(operand, m::Broadcast(m::Op(&broadcast_operand)))) {
    std::vector<HloInstruction*> new_indices;
    new_indices.reserve(broadcast_operand->shape().dimensions_size());
    std::vector<int64_t> new_slice_sizes;
    new_slice_sizes.reserve(broadcast_operand->shape().dimensions_size());

    for (int64_t dim : operand->dimensions()) {
      new_indices.push_back(dynamic_slice->mutable_operand(1 + dim));
      new_slice_sizes.push_back(dynamic_slice->slice_sizes(dim));
    }

    VLOG(3) << "Sink broadcast through dynamic slice";
    VLOG(3) << "Original dynamic slice: " << dynamic_slice->ToString();
    VLOG(3) << "Original broadcast: " << operand->ToString();
    HloInstruction* new_dynamic_slice = broadcast_operand;
    if (!new_slice_sizes.empty()) {
      auto new_ds_shape = broadcast_operand->shape();
      for (int64_t i = 0; i < broadcast_operand->shape().dimensions_size();
           ++i) {
        new_ds_shape.set_dimensions(i, new_slice_sizes[i]);
      }
      simplifier_->UpdateLayout(&new_ds_shape);
      new_dynamic_slice =
          dynamic_slice->AddInstruction(HloInstruction::CreateDynamicSlice(
              new_ds_shape, broadcast_operand, new_indices, new_slice_sizes));
    }
    auto new_broadcast =
        operand->AddInstruction(HloInstruction::CreateBroadcast(
            dynamic_slice->shape(), new_dynamic_slice, operand->dimensions()));
    VLOG(3) << "New dynamic slice: " << dynamic_slice->ToString();
    VLOG(3) << "New broadcast: " << new_broadcast->ToString();
    return ReplaceInstruction(dynamic_slice, new_broadcast);
  }

  HloInstruction *reshape, *reshape_operand;
  if (Match(operand, m::Reshape(&reshape, m::Op(&reshape_operand))) &&
      reshape->ReshapeMerelyInsertsOrDeletes1SizedDimensions().has_value() &&
      !options_.is_layout_sensitive()) {
    int64_t slice_dim = 0;
    HloInstruction* zero = MakeScalarLike(dynamic_slice->mutable_operand(1), 0);
    std::vector<HloInstruction*> starts;
    starts.reserve(reshape_operand->shape().dimensions_size());
    std::vector<int64_t> slice_sizes;
    slice_sizes.reserve(reshape_operand->shape().dimensions_size());
    for (int64_t dim = 0; dim < reshape_operand->shape().dimensions_size();
         ++dim) {
      if (reshape_operand->shape().dimensions(dim) == 1) {
        starts.push_back(zero);
        slice_sizes.push_back(1);
        continue;
      }
      while (dynamic_slice->operand(0)->shape().dimensions(slice_dim) == 1) {
        ++slice_dim;
      }
      starts.push_back(dynamic_slice->mutable_operand(1 + slice_dim));
      slice_sizes.push_back(dynamic_slice->slice_sizes(slice_dim));
      ++slice_dim;
    }
    HloInstruction* new_dynamic_slice =
        dynamic_slice->AddInstruction(HloInstruction::CreateDynamicSlice(
            ShapeUtil::MakeShape(dynamic_slice->shape().element_type(),
                                 slice_sizes),
            reshape_operand, starts, slice_sizes));
    return ReplaceWithNewInstruction(
        dynamic_slice, HloInstruction::CreateReshape(dynamic_slice->shape(),
                                                     new_dynamic_slice));
  }

  HloInstruction *transpose, *transpose_operand;
  if (Match(operand, m::Transpose(&transpose, m::Op(&transpose_operand))) &&
      !options_.is_layout_sensitive()) {
    auto output_to_input = InversePermutation(transpose->dimensions());
    HloInstruction* new_slice =
        dynamic_slice->AddInstruction(HloInstruction::CreateDynamicSlice(
            ShapeUtil::PermuteDimensions(output_to_input,
                                         dynamic_slice->shape()),
            transpose_operand,
            Permute(absl::MakeSpan(dynamic_slice->operands().begin() + 1,
                                   dynamic_slice->operands().end()),
                    output_to_input),
            Permute(dynamic_slice->dynamic_slice_sizes(), output_to_input)));
    return ReplaceWithNewInstruction(
        dynamic_slice,
        HloInstruction::CreateTranspose(dynamic_slice->shape(), new_slice,
                                        transpose->dimensions()));
  }

  // Convert a dynamic slice into a slice if all offsets are constant, the
  // operand is not constant, and the input and output memory spaces are the
  // same.
  if (!options_.disable_dynamic_slice_to_slice_conversion() &&
      operand->opcode() != HloOpcode::kConstant &&
      absl::c_all_of(absl::MakeSpan(dynamic_slice->operands().begin() + 1,
                                    dynamic_slice->operands().end()),
                     [](HloInstruction* operand) {
                       return operand->opcode() == HloOpcode::kConstant &&
                              ShapeUtil::ElementIsIntegral(operand->shape());
                     }) &&
      (!options_.is_layout_sensitive() ||
       (dynamic_slice->shape().has_layout() &&
        dynamic_slice->operand(0)->shape().has_layout() &&
        dynamic_slice->shape().layout().memory_space() ==
            dynamic_slice->operand(0)->shape().layout().memory_space()))) {
    const int64_t rank = operand->shape().dimensions_size();
    std::vector<int64_t> slice_starts(rank);
    std::vector<int64_t> slice_limits(rank);
    std::vector<int64_t> slice_strides(rank, 1);

    for (int64_t i = 0; i < rank; ++i) {
      std::optional<int64_t> offset =
          dynamic_slice->operand(i + 1)->literal().GetFirstInteger();
      if (!offset || *offset < 0) {
        return absl::OkStatus();
      }
      const int64_t max_offset =
          dynamic_slice->operand(0)->shape().dimensions(i) -
          dynamic_slice->shape().dimensions(i);
      slice_starts[i] = std::min(max_offset, *offset);
      slice_limits[i] =
          std::min(max_offset, *offset) + dynamic_slice->shape().dimensions(i);
    }
    return ReplaceWithNewInstruction(
        dynamic_slice,
        HloInstruction::CreateSlice(dynamic_slice->shape(), operand,
                                    slice_starts, slice_limits, slice_strides));
  }

  // Convert the dynamic slice of an iota to just a reference to the index
  // (possibly clamped and scaled). Index is always a scalar integer. Output
  // should be a rank 1 array of size 1 with element type matching that of the
  // scalar index (except the signedness).
  const PrimitiveType element_type = dynamic_slice->shape().element_type();
  if (operand->shape().dimensions_size() == 1 &&
      dynamic_slice->shape().dimensions_size() == 1 &&
      dynamic_slice->shape().dimensions(0) == 1 &&
      (element_type == S32 || element_type == U32)) {
    // Match multiply(x, broadcast(scalar)) and return the scalar
    // constant.
    auto match_multiply_with_scalar =
        [&](HloInstruction* hlo) -> HloInstruction* {
      if (hlo->opcode() != HloOpcode::kMultiply) {
        return nullptr;
      }
      HloInstruction* broadcast = hlo->mutable_operand(1);
      if (broadcast->opcode() == HloOpcode::kBroadcast &&
          broadcast->dimensions().empty() &&
          ShapeUtil::IsScalar(broadcast->operand(0)->shape())) {
        return broadcast->mutable_operand(0);
      }
      return nullptr;
    };

    HloInstruction* multiplier = match_multiply_with_scalar(operand);
    if (multiplier) {
      operand = operand->mutable_operand(0);
    }

    if (operand->opcode() == HloOpcode::kIota) {
      // This dynamic_slice will have a single start_index operand (since its
      // operand is rank 1).
      HloInstruction* index = dynamic_slice->mutable_operand(1);
      const PrimitiveType index_type = index->shape().element_type();

      auto create_constant = [&](int64_t value) {
        if (index_type == S32) {
          return MakeScalarLike<int32_t>(index, value);
        } else {
          return MakeScalarLike<uint32_t>(index, value);
        }
      };

      if (index_type == S32 || index_type == U32) {
        // Clamp the index to the range of the iota.
        int64_t iota_size = operand->shape().dimensions(0);
        HloInstruction* low = create_constant(0);
        HloInstruction* high = create_constant(iota_size - 1);
        HloInstruction* clamped =
            dynamic_slice->AddInstruction(HloInstruction::CreateTernary(
                index->shape(), HloOpcode::kClamp, low, index, high));

        // Convert the clamped index from index_type to element_type and
        // multiply with the multiplier.
        HloInstruction* result = clamped;
        if (index_type != element_type) {
          Shape result_shp = result->shape();
          result_shp.set_element_type(element_type);
          result = dynamic_slice->AddInstruction(
              HloInstruction::CreateConvert(result_shp, clamped));
        }

        if (multiplier) {
          result = dynamic_slice->AddInstruction(HloInstruction::CreateBinary(
              result->shape(), HloOpcode::kMultiply, result, multiplier));
        }

        return ReplaceWithNewInstruction(
            dynamic_slice,
            HloInstruction::CreateReshape(dynamic_slice->shape(), result));
      }
    }
  }

  // ds(ds(x,id),inner_id) -> ds(x, id + inner_id)
  if (operand->opcode() == HloOpcode::kDynamicSlice) {
    TF_RETURN_IF_ERROR(dynamic_slice->ReplaceOperandWithDifferentShape(
        0, operand->mutable_operand(0)));
    for (int64_t i = 1; i < dynamic_slice->operand_count(); ++i) {
      HloInstruction* index = dynamic_slice->mutable_operand(i);
      HloInstruction* inner_index = operand->mutable_operand(i);
      inner_index = inner_index->AddInstruction(HloInstruction::CreateTernary(
          inner_index->shape(), HloOpcode::kClamp,
          MakeScalarLike(inner_index, 0), inner_index,
          MakeScalarLike(inner_index,
                         operand->operand(0)->shape().dimensions(i - 1) -
                             dynamic_slice->dynamic_slice_sizes()[i - 1])));
      if (inner_index->shape().element_type() !=
          index->shape().element_type()) {
        inner_index = inner_index->AddInstruction(
            HloInstruction::CreateConvert(index->shape(), inner_index));
      }
      HloInstruction* combined_index =
          operand->AddInstruction(HloInstruction::CreateBinary(
              index->shape(), HloOpcode::kAdd, index, inner_index));
      TF_RETURN_IF_ERROR(dynamic_slice->ReplaceOperandWith(i, combined_index));
    }
    MarkAsChanged();
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  // Skip optimizations for async dynamic update slices
  if (dynamic_update_slice->parent()->IsAsyncComputation()) {
    return absl::OkStatus();
  }
  // Rewriting DynamicUpdateSlice when it matches
  // dynamic_update_slice(broadcast(constant),data,constant_index0,...)
  // to a Pad(x, constant)
  // Only Broadcast considered currently, other ops need to be considered
  // in the future.
  HloInstruction* updated = dynamic_update_slice->mutable_operand(0);
  HloInstruction* dus_update = dynamic_update_slice->mutable_operand(1);
  HloInstruction* pad_value;
  if (Match(updated,
            m::Broadcast(m::Op(&pad_value).WithShape(m::Shape().IsScalar())))) {
    auto updated_shape = updated->shape();
    auto update_shape = dus_update->shape();
    auto update_start_indx = dynamic_update_slice->operand(2);
    int64_t offset = 0;
    bool compatible = true;
    // Whether the start indices to dynamic update slice is a list,
    // output of a tuple/concatenate, we setup the update_start_indx
    // appropriately.
    if (ShapeUtil::IsScalar(update_start_indx->shape())) {
      update_start_indx = dynamic_update_slice;
      offset = 2;
    } else {
      if (update_start_indx->opcode() == HloOpcode::kTuple ||
          update_start_indx->opcode() == HloOpcode::kConcatenate) {
        offset = 0;
      } else {
        compatible = false;
      }
    }

    // For broadcast that's used for host offloading's DUS, we don't want this
    // to be rewritten to a pad. Unfortunately, the host memory space is only
    // set after HostOffloader is run, so we pattern match here. After
    // HostOffloader is run the broadcast should be rewritten to an
    // AllocateBuffer so this dus->pad rewrite won't apply anymore.
    auto is_host_offloading = [&](HloInstruction* hlo) {
      const auto custom_call_pattern =
          m::CustomCall({memory_annotations::kMoveToHostCustomCallTarget});
      if (Match(hlo, custom_call_pattern)) {
        return true;
      }

      const auto formatting_op =
          m::AnyOf<HloInstruction>(m::Reshape(), m::Bitcast(), m::Copy());
      while (Match(hlo, formatting_op)) {
        hlo = hlo->mutable_operand(0);
        if (Match(hlo, custom_call_pattern)) {
          return true;
        }
      }
      return false;
    };

    if (compatible && is_host_offloading(dus_update)) {
      compatible = false;
    }

    PaddingConfig padding_config;
    if (compatible) {
      for (int64_t dim = 0; dim < updated_shape.dimensions_size(); ++dim) {
        auto padding_config_dim = padding_config.add_dimensions();
        auto slice_dim_start = update_start_indx->operand(dim + offset);
        if (!Match(slice_dim_start, m::ConstantScalar())) {
          compatible = false;
          break;
        }
        VLOG(2) << "slice: " << slice_dim_start->ToString();
        std::optional<int64_t> start =
            slice_dim_start->literal().GetFirstInteger();
        if (!start) {
          compatible = false;
          break;
        }
        VLOG(2) << "start value: " << *start;
        auto update_width = ShapeUtil::GetDimension(update_shape, dim);
        auto bcast_width = ShapeUtil::GetDimension(updated_shape, dim);
        // Clamp start so that it is non-negative.
        *start = std::max<int64_t>(0, *start);
        // Clamp start so that it is in-bounds.
        *start = std::min<int64_t>(bcast_width - update_width, *start);
        VLOG(2) << "adjusted start value: " << *start;
        padding_config_dim->set_edge_padding_low(*start);
        padding_config_dim->set_edge_padding_high(bcast_width -
                                                  (*start + update_width));
        // dynamic_update_slice does not specify a stride
        padding_config_dim->set_interior_padding(0);
      }
    }

    if (compatible) {
      HloInstruction* pad =
          dynamic_update_slice->AddInstruction(HloInstruction::CreatePad(
              updated_shape, dus_update, pad_value, padding_config));
      VLOG(2) << dynamic_update_slice->ToString();
      VLOG(2) << " with pad:" << pad->ToString();
      VLOG(2) << " Computation before rewrite is: "
              << dynamic_update_slice->parent()->ToString();
      return ReplaceInstruction(dynamic_update_slice, pad);
    }
  }

  // DynamicUpdateSlice where operand and dus_update have the same size is
  // equal to dus_update.
  if (SameShape(dynamic_update_slice, dus_update)) {
    return ReplaceInstruction(dynamic_update_slice, dus_update);
  }

  // DynamicUpdateSlice clamps the offset. If the slice size has the same size
  // on a dim as dus_update, we can replace it with zero.
  std::vector<int> same_size_dims_to_simplify;
  for (int64_t dim = 0; dim < dus_update->shape().dimensions_size(); ++dim) {
    if (!(dynamic_update_slice->operand(dim + 2)->IsConstant() &&
          IsAll(dynamic_update_slice->operand(dim + 2), 0)) &&
        dus_update->shape().dimensions(dim) ==
            dynamic_update_slice->shape().dimensions(dim)) {
      same_size_dims_to_simplify.push_back(dim);
    }
  }
  if (!same_size_dims_to_simplify.empty()) {
    HloInstruction* zero =
        MakeScalarLike(dynamic_update_slice->mutable_operand(2), 0);
    auto new_operands = dynamic_update_slice->mutable_operands();
    for (int64_t dim : same_size_dims_to_simplify) {
      new_operands[dim + 2] = zero;
    }
    return ReplaceInstruction(
        dynamic_update_slice,
        dynamic_update_slice->AddInstruction(
            dynamic_update_slice->CloneWithNewOperands(
                dynamic_update_slice->shape(), new_operands)));
  }

  // If any dimension of dus_update is 0, elide the DynamicUpdateSlice.  This
  // optimization becomes invalid should we later prefer to warn about out of
  // bound indices.
  if (ShapeUtil::IsZeroElementArray(dus_update->shape())) {
    return ReplaceInstruction(dynamic_update_slice, updated);
  }

  // dus(a,dus(ds(a,id),c,inner_id)),id) is equivalent to dus(a,c,inner_id + id)
  if (dus_update->opcode() == HloOpcode::kDynamicUpdateSlice &&
      (dus_update->operand(0)->opcode() == HloOpcode::kDynamicSlice &&
       dus_update->operand(0)->operand(0) == dynamic_update_slice->operand(0) &&
       absl::c_equal(
           absl::MakeConstSpan(dynamic_update_slice->operands()).subspan(2),
           absl::MakeConstSpan(dus_update->operand(0)->operands())
               .subspan(1)))) {
    TF_RETURN_IF_ERROR(dynamic_update_slice->ReplaceOperandWithDifferentShape(
        1, dus_update->mutable_operand(1)));
    for (int64_t i = 2; i < dynamic_update_slice->operand_count(); ++i) {
      HloInstruction* index = dynamic_update_slice->mutable_operand(i);
      HloInstruction* inner_index = dus_update->mutable_operand(i);
      inner_index = inner_index->AddInstruction(HloInstruction::CreateTernary(
          inner_index->shape(), HloOpcode::kClamp,
          MakeScalarLike(inner_index, 0), inner_index,
          MakeScalarLike(
              inner_index,
              dus_update->shape().dimensions(i - 2) -
                  dus_update->operand(1)->shape().dimensions(i - 2))));
      if (inner_index->shape().element_type() !=
          index->shape().element_type()) {
        inner_index = inner_index->AddInstruction(
            HloInstruction::CreateConvert(index->shape(), inner_index));
      }
      HloInstruction* combined_index =
          dus_update->AddInstruction(HloInstruction::CreateBinary(
              index->shape(), HloOpcode::kAdd, index, inner_index));
      TF_RETURN_IF_ERROR(
          dynamic_update_slice->ReplaceOperandWith(i, combined_index));
    }
    MarkAsChanged();
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

static bool MatchArgMinMax(const HloInstruction* hlo, bool is_max) {
  // Create matcher for shared sub-expression.
  auto value_pred = m::OrAnyOrder(
      m::Compare(m::Parameter(0), m::Parameter(2))
          .WithComparisonDirection(is_max ? ComparisonDirection::kGt
                                          : ComparisonDirection::kLt),
      m::Compare(m::Parameter(0), m::Parameter(0))
          .WithComparisonDirection(ComparisonDirection::kNe));

  // Match on argmax reduction computation.
  return Match(
      hlo,
      m::Tuple(
          m::Select(value_pred, m::Parameter(0), m::Parameter(2)),
          m::Select(
              m::OrAnyOrder(
                  value_pred,
                  m::And(
                      m::Compare(m::Parameter(0), m::Parameter(2))
                          .WithComparisonDirection(ComparisonDirection::kEq),
                      m::Compare(m::Parameter(1), m::Parameter(3))
                          .WithComparisonDirection(ComparisonDirection::kLt))),
              m::Parameter(1), m::Parameter(3))));
}

// Match on variadic reduce which computes and returns (min, arg_min).
//
//                   p0   p2    p1    p3
//                  /|\ \/ |\    |\   /|
//                 / | \/\ | \   | \ / |
//                /  | /\ \|  |  |  /\ |
//               Ne  Lt |  \  |  | |  ||
//                 \ /  |  |\ |  | /  ||
//                  Or /  /  Eq  Lt   ||
//                  | /  /    \  /    //
//                  | |  |     And   //
//                  | |  |      |  //
//                  select     select
//                      \     /
//                       tuple
//
static bool MatchArgMin(const HloInstruction* hlo) {
  // Match on variadic Reduce ArgMin
  if (hlo->opcode() != HloOpcode::kReduce || hlo->operand_count() != 4 ||
      !hlo->shape().IsTuple() ||
      hlo->operand(1)->opcode() != HloOpcode::kIota ||
      !IsScalarConstantInf(hlo->operand(2)) ||
      !IsScalarConstantZero(hlo->operand(3))) {
    return false;
  }
  return MatchArgMinMax(hlo->to_apply()->root_instruction(), /*is_max=*/false);
}

// Match on variadic reduce which computes and returns (max, arg_max).
//
//                   p0   p2    p1    p3
//                  /|\ \/ |\    |\   /|
//                 / | \/\ | \   | \ / |
//                /  | /\ \|  |  |  /\ |
//               Ne  Gt |  \  |  | |  ||
//                 \ /  |  |\ |  | /  ||
//                  Or /  /  Eq  Lt   ||
//                  | /  /    \  /    //
//                  | |  |     And   //
//                  | |  |      |  //
//                  select     select
//                      \     /
//                       tuple
//
static bool MatchArgMax(const HloInstruction* hlo) {
  // Match on variadic Reduce ArgMax.
  if (hlo->opcode() != HloOpcode::kReduce || hlo->operand_count() != 4 ||
      !hlo->shape().IsTuple() ||
      hlo->operand(1)->opcode() != HloOpcode::kIota ||
      !IsScalarConstantNegInf(hlo->operand(2)) ||
      !IsScalarConstantZero(hlo->operand(3))) {
    return false;
  }
  return MatchArgMinMax(hlo->to_apply()->root_instruction(), /*is_max=*/true);
}

static bool ReductionComputationsEquivalent(const HloComputation& a,
                                            const HloComputation& b) {
  if (a == b) {
    return true;
  }

  // Check for simple commutative reduction functions.
  enum CommutativeFnKind { kAdd, kMul, kAnd, kOr };
  auto categorize_computation =
      [](const HloComputation& c) -> std::optional<CommutativeFnKind> {
    if (c.num_parameters() != 2) {
      return std::nullopt;
    }

    const HloInstruction* root = c.root_instruction();
    if (Match(root, m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
      return kAdd;
    }
    if (Match(root, m::MultiplyAnyOrder(m::Parameter(0), m::Parameter(1)))) {
      return kMul;
    }
    if (Match(root, m::AndAnyOrder(m::Parameter(0), m::Parameter(1)))) {
      return kAnd;
    }
    if (Match(root, m::OrAnyOrder(m::Parameter(0), m::Parameter(1)))) {
      return kOr;
    }
    return std::nullopt;
  };
  auto category_a = categorize_computation(a);
  auto category_b = categorize_computation(b);
  return category_a.has_value() && category_b.has_value() &&
         category_a == category_b;
}

absl::Status AlgebraicSimplifierVisitor::HandleReduce(HloInstruction* hlo) {
  HloReduceInstruction* reduce = Cast<HloReduceInstruction>(hlo);
  bool multi_output_reduce = reduce->shape().IsTuple();
  // For tuple reduce, we require all reduce shapes to be the same, up to the
  // element types, so we can just the first operand and the first result as a
  // representative.
  auto arg = reduce->inputs()[0];
  auto init_value = reduce->init_values()[0];
  const Shape& reduce_result_shape =
      multi_output_reduce ? reduce->shape().tuple_shapes(0) : reduce->shape();

  absl::Span<const int64_t> dimensions(reduce->dimensions());
  HloComputation* function = reduce->to_apply();
  if (ShapeUtil::IsZeroElementArray(arg->shape()) ||
      ShapeUtil::IsZeroElementArray(reduce_result_shape)) {
    if (multi_output_reduce) {
      std::vector<HloInstruction*> broadcast_inits;
      int64_t inputs = reduce->input_count();
      broadcast_inits.reserve(inputs);
      for (int64_t i = 0; i < inputs; ++i) {
        broadcast_inits.push_back(reduce->init_values()[i]->AddInstruction(
            HloInstruction::CreateBroadcast(reduce->shape().tuple_shapes(i),
                                            reduce->init_values()[i], {})));
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateTuple(broadcast_inits));
    } else {
      return ReplaceWithNewInstruction(
          reduce,
          HloInstruction::CreateBroadcast(reduce_result_shape, init_value, {}));
    }
  }

  // Turn trivial variadic reductions into normal reductions.
  if (multi_output_reduce && reduce->shape().tuple_shapes_size() == 1 &&
      reduce->input_count() == 1 &&
      Match(function->root_instruction(), m::Tuple())) {
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;
    replacements[function->root_instruction()] = nullptr;
    auto new_function = computation_->parent()->AddEmbeddedComputation(
        function->CloneWithReplacements(
            &replacements, /*extra_parameters=*/{},
            /*context=*/nullptr,
            /*suffix=*/"clone",
            /*new_root=*/function->root_instruction()->operand(0)));
    auto new_reduce = reduce->AddInstruction(
        HloInstruction::CreateReduce(reduce_result_shape, arg, init_value,
                                     reduce->dimensions(), new_function));
    return ReplaceWithNewInstruction(reduce,
                                     HloInstruction::CreateTuple({new_reduce}));
  }

  // If the reduction results in the same number of elements, then the only
  // possible side effect would be a reshape. Since the init_value is an
  // identity of the reduction function, we can therefore replace the reduce
  // with a simple reshape, ignoring the reduction function completely.
  if (ShapeUtil::ElementsIn(reduce_result_shape) ==
          ShapeUtil::ElementsIn(arg->shape()) &&
      (!options_.is_layout_sensitive() ||
       options_.ReshapeIsBitcast(arg->shape(), reduce_result_shape))) {
    if (multi_output_reduce) {
      std::vector<HloInstruction*> reshaped_args;
      int64_t inputs = reduce->input_count();
      reshaped_args.reserve(inputs);
      for (int64_t i = 0; i < inputs; ++i) {
        reshaped_args.push_back(
            reduce->AddInstruction(HloInstruction::CreateReshape(
                reduce->shape().tuple_shapes(i), reduce->inputs()[i])));
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateTuple(reshaped_args));
    } else {
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReshape(reduce_result_shape, arg));
    }
  }

  if (options_.is_layout_sensitive()) {
    return absl::OkStatus();
  }

  HloInstruction* negate_arg;
  if (ShapeUtil::ElementIsFloating(reduce->shape()) &&
      Match(arg, m::Negate(m::Op(&negate_arg))) &&
      IsScalarConstantZero(init_value) &&
      Match(reduce->to_apply()->root_instruction(),
            m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    TF_RETURN_IF_ERROR(reduce->ReplaceOperandWith(0, negate_arg));
    auto users = reduce->users();
    auto* negated_reduce = arg->AddInstruction(HloInstruction::CreateUnary(
        reduce->shape(), HloOpcode::kNegate, reduce));
    MarkAsChanged();
    return reduce->ReplaceUsesWith(users, negated_reduce);
  }

  // Try to reorder reduce(dot(A, B)) to dot(A, reduce(B))
  if (options_.raise_slice_and_reduce_through_dot()) {
    HloInstruction *a, *b;
    // Reordering does not seem possible if the dot has batch dimensions. We
    // also need the reduction operation to be add, and the reduce to have an
    // initial value of 0.
    if (Match(arg, m::Dot(m::Op(&a), m::Op(&b))) &&
        IsScalarConstantZero(init_value) &&
        Match(reduce->to_apply()->root_instruction(),
              m::AddAnyOrder(m::Parameter(0), m::Parameter(1))) &&
        arg->dot_dimension_numbers().lhs_batch_dimensions().empty() &&
        !Cast<HloDotInstruction>(arg)->sparse_operands()) {
      // Create maps for converting AB dimensions to A and B
      DotDimensionNumbers ab_dnums = arg->dot_dimension_numbers();
      std::vector<int64_t> map_ab_a, map_ab_b;
      std::tie(map_ab_a, map_ab_b) =
          ConstructFromDotMaps(arg, a->shape(), b->shape());

      // Create new reduce dimensions using the maps
      std::vector<int64_t> reduce_a_dims, reduce_b_dims;
      for (int64_t dim : reduce->dimensions()) {
        if (map_ab_a[dim] != -1) {
          reduce_a_dims.push_back(map_ab_a[dim]);
        }
        if (map_ab_b[dim] != -1) {
          reduce_b_dims.push_back(map_ab_b[dim]);
        }
      }

      // Create Hlo for reducing a and b
      TF_ASSIGN_OR_RETURN(
          HloInstruction * reduce_a,
          MakeReduceHlo(a, init_value, reduce_a_dims, function));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * reduce_b,
          MakeReduceHlo(b, init_value, reduce_b_dims, function));

      // Construct maps from reduce_a and reduce_b to a and b
      std::vector<int64_t> map_reduce_a_a(reduce_a->shape().dimensions_size(),
                                          -1),
          map_reduce_b_b(reduce_b->shape().dimensions_size(), -1);
      int64_t reduce_a_index = 0;
      for (int64_t a_index = 0; a_index < a->shape().dimensions_size();
           ++a_index) {
        if (!absl::c_linear_search(reduce_a_dims, a_index)) {
          map_reduce_a_a[reduce_a_index] = a_index;
          ++reduce_a_index;
        }
      }
      int64_t reduce_b_index = 0;
      for (int64_t b_index = 0; b_index < b->shape().dimensions_size();
           ++b_index) {
        if (!absl::c_linear_search(reduce_b_dims, b_index)) {
          map_reduce_b_b[reduce_b_index] = b_index;
          ++reduce_b_index;
        }
      }

      // Construct dot dimension numbers for new dot
      const auto& a_contracting_dims = ab_dnums.lhs_contracting_dimensions();
      const auto& b_contracting_dims = ab_dnums.rhs_contracting_dimensions();
      DotDimensionNumbers new_dot_dnums;
      for (int64_t reduce_a_index = 0;
           reduce_a_index < reduce_a->shape().dimensions_size();
           ++reduce_a_index) {
        if (map_reduce_a_a[reduce_a_index] != -1) {
          int64_t a_index = map_reduce_a_a[reduce_a_index];
          if (absl::c_linear_search(a_contracting_dims, a_index)) {
            new_dot_dnums.add_lhs_contracting_dimensions(reduce_a_index);
          }
        }
      }
      for (int64_t reduce_b_index = 0;
           reduce_b_index < reduce_b->shape().dimensions_size();
           ++reduce_b_index) {
        if (map_reduce_b_b[reduce_b_index] != -1) {
          int64_t b_index = map_reduce_b_b[reduce_b_index];
          if (absl::c_linear_search(b_contracting_dims, b_index)) {
            new_dot_dnums.add_rhs_contracting_dimensions(reduce_b_index);
          }
        }
      }

      // Create Hlo for new dot
      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_dot,
          MakeDotHlo(reduce_a, reduce_b, new_dot_dnums, arg->precision_config(),
                     reduce->shape().element_type()));

      // Compute the number of flops for both old and new operations
      const int64_t old_flops =
          HloCostAnalysis::GetDotFlops(a->shape(), arg->shape(), ab_dnums) +
          GetReduceFlops(reduce);
      const int64_t new_flops =
          GetReduceFlops(reduce_a) + GetReduceFlops(reduce_b) +
          HloCostAnalysis::GetDotFlops(reduce_a->shape(), new_dot->shape(),
                                       new_dot_dnums);

      // Only reorder if it would result in sufficiently fewer flops
      if (old_flops / static_cast<double>(new_flops) >
          options_.raise_slice_and_reduce_through_dot_threshold()) {
        VLOG(10) << "Reordering reduce into dot operands";
        return ReplaceInstruction(reduce, new_dot);
      }
    }
  }

  // TODO(b/131122694): Most of those optimizations below can be done for
  // multi-output reduces.
  if (multi_output_reduce) {
    return absl::OkStatus();
  }

  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field if the output of the reduce is a vector or scalar. Higher ranked
  // result may require a transpose of the output.
  if (arg->opcode() == HloOpcode::kTranspose &&
      (options_.unconditionally_simplify_reduce_of_transpose_or_reshape() ||
       (reduce->shape().dimensions_size() < 2 || arg->user_count() == 1 ||
        absl::c_all_of(arg->users(), [](HloInstruction* use) {
          return use->opcode() == HloOpcode::kReduce;
        })))) {
    auto transpose_dimensions = arg->dimensions();
    std::vector<int64_t> new_reduce_dimensions;
    new_reduce_dimensions.reserve(dimensions.size());
    for (auto dim : dimensions) {
      new_reduce_dimensions.push_back(transpose_dimensions[dim]);
    }

    Shape new_reduce_result_shape = ShapeUtil::DeleteDimensions(
        new_reduce_dimensions, arg->mutable_operand(0)->shape());
    HloInstruction* new_reduce =
        reduce->AddInstruction(HloInstruction::CreateReduce(
            new_reduce_result_shape, arg->mutable_operand(0), init_value,
            new_reduce_dimensions, function));
    std::vector<int64_t> new_transpose_dimensions;
    for (auto dim : transpose_dimensions) {
      if (absl::c_linear_search(new_reduce_dimensions, dim)) {
        continue;
      }
      new_transpose_dimensions.push_back(dim);
    }

    // If new transpose dimensions are sorted, then there is no need to
    // transpose reduce result.
    if (absl::c_is_sorted(new_transpose_dimensions)) {
      return ReplaceInstruction(reduce, new_reduce);
    }
    for (auto& d : new_transpose_dimensions) {
      auto old_dim = d;
      for (auto reduced_dim : new_reduce_dimensions) {
        if (old_dim > reduced_dim) {
          --d;
        }
      }
    }
    TF_ASSIGN_OR_RETURN(HloInstruction * new_transpose,
                        MakeTransposeHlo(new_reduce, new_transpose_dimensions));
    return ReplaceInstruction(reduce, new_transpose);
  }

  // If a reduce feeds a reduce with the same computation and initial value,
  // they can be combined into a single reduce.
  if (arg->opcode() == HloOpcode::kReduce &&
      init_value->Identical(*arg->operand(1)) &&
      ReductionComputationsEquivalent(*function, *arg->to_apply())) {
    // Create a new reduce with the combined reduction dimensions of both
    // reduces.
    std::vector<int64_t> arg_dims = *arg->mutable_dimensions();
    absl::c_sort(arg_dims);
    std::vector<int64_t> reduce_dims = *reduce->mutable_dimensions();
    absl::c_sort(reduce_dims);
    // Transform reduce_dims to the same rank as the operand of the operand.
    for (int64_t arg_dim : arg_dims) {
      for (int64_t& dim : reduce_dims) {
        if (dim >= arg_dim) {
          ++dim;
        }
      }
    }
    std::vector<int64_t> new_dimensions;
    new_dimensions.reserve(arg->dimensions().size() +
                           reduce->dimensions().size());
    std::merge(arg_dims.begin(), arg_dims.end(), reduce_dims.begin(),
               reduce_dims.end(), std::back_inserter(new_dimensions));
    return ReplaceWithNewInstruction(
        reduce, HloInstruction::CreateReduce(
                    reduce_result_shape, arg->mutable_operand(0), init_value,
                    new_dimensions, function));
  }

  // Handle two cases of reduce(reshape(x)).
  //
  // 1. The reshape collapses/expands only dimensions that are being reduced.
  //    In this case we can just reduce those dimensions and skip the reshape.
  // 2. The reshape collapses/expands only dimensions that are *not* being
  //    reduced.  In this case we can do the reshape after the reduce.  This is
  //    beneficial because the reduce will now operate on less data.
  if (options_.enable_reduce_of_reshape() &&
      arg->opcode() == HloOpcode::kReshape) {
    std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
        ShapeUtil::DimensionsUnmodifiedByReshape(arg->operand(0)->shape(),
                                                 arg->shape());

    // True for those dimensions of the reduce input that are not reduced, false
    // for the dims that are reduced.
    absl::InlinedVector<bool, 8> arg_dim_in_output(
        arg->shape().dimensions_size(), true);
    for (auto dim : dimensions) {
      arg_dim_in_output[dim] = false;
    }

    // True for those dimensions of the reduce input that are unmodified by the
    // reshape.
    absl::InlinedVector<bool, 8> arg_dim_unmodified(
        arg->shape().dimensions_size(), false);
    for (auto [input_idx, output_idx] : unmodified_dims) {
      arg_dim_unmodified[output_idx] = true;
    }

    // Case 1: Check whether all dimensions that are not removed in the reduce
    // are unmodified by the reshape. For example:
    // reduce(reshape([A,B*C], a[A,B,C]),[1]) = reduce(a[A, B, C], [1, 2])
    bool can_move_reshape_into_reduce = true;
    for (int64_t i = 0; i < arg_dim_in_output.size(); ++i) {
      if (arg_dim_in_output[i] && !arg_dim_unmodified[i]) {
        can_move_reshape_into_reduce = false;
      }
    }
    if (can_move_reshape_into_reduce) {
      MarkAsChanged();
      absl::flat_hash_set<int64_t> dimensions_not_to_reduce;
      for (auto dim_pair : unmodified_dims) {
        if (arg_dim_in_output[dim_pair.second]) {
          dimensions_not_to_reduce.insert(dim_pair.first);
        }
      }
      std::vector<int64_t> new_reduce_dimensions;
      for (int64_t i = 0; i < arg->operand(0)->shape().dimensions_size(); ++i) {
        if (!dimensions_not_to_reduce.contains(i)) {
          new_reduce_dimensions.push_back(i);
        }
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReduce(
                      reduce_result_shape, arg->mutable_operand(0), init_value,
                      new_reduce_dimensions, function));
    }

    // Case 2: Check whether the reshape only modifies non-reduction dimensions.
    // Equivalently, the reduction dimensions are all preserved by the reshape.
    if ((arg->user_count() == 1 ||
         options_.unconditionally_simplify_reduce_of_transpose_or_reshape()) &&
        absl::c_all_of(dimensions,
                       [&](int64_t dim) { return arg_dim_unmodified[dim]; })) {
      absl::InlinedVector<int64_t, 8> new_reduce_dims;
      for (auto dim : dimensions) {
        auto matching_dim_it = absl::c_find_if(
            unmodified_dims,
            [&](const auto& dim_pair) { return dim_pair.second == dim; });
        CHECK(matching_dim_it != unmodified_dims.end());
        new_reduce_dims.push_back(matching_dim_it->first);
      }

      TF_ASSIGN_OR_RETURN(
          HloInstruction * new_reduce,
          MakeReduceHlo(arg->mutable_operand(0), init_value, new_reduce_dims,
                        reduce->to_apply(), &reduce->metadata()));
      TF_ASSIGN_OR_RETURN(HloInstruction * new_reshape,
                          MakeReshapeHlo(reduce->shape(), new_reduce));
      return ReplaceInstruction(reduce, new_reshape);
    }
  }

  // Convert Reduce(concat({a,b,...})) to
  //  map(reduce(a),map(reduce(b),...,))
  // provided that the shapes of a,b,... have the same dimensions, or
  // enable_unconditional_reduce_of_concat_replacement() is true.
  //
  // This should make fusion easier or use less memory bandwidth in the unfused
  // case.
  if (arg->opcode() == HloOpcode::kConcatenate &&
      absl::c_linear_search(reduce->dimensions(),
                            arg->concatenate_dimension())) {
    bool same_shapes = true;
    for (int64_t i = 1; i < arg->operand_count(); ++i) {
      if (!Shape::Equal().IgnoreLayout()(arg->operand(i)->shape(),
                                         arg->operand(0)->shape())) {
        same_shapes = false;
        break;
      }
    }
    if (options_.enable_unconditional_reduce_of_concat_replacement() ||
        same_shapes || reduce->shape().dimensions_size() == 0) {
      HloInstruction* old_reduce = nullptr;
      for (HloInstruction* operand : arg->operands()) {
        HloInstruction* new_reduce =
            reduce->AddInstruction(HloInstruction::CreateReduce(
                reduce_result_shape, operand, init_value, reduce->dimensions(),
                function));
        if (old_reduce != nullptr) {
          new_reduce = reduce->AddInstruction(HloInstruction::CreateMap(
              reduce_result_shape, {old_reduce, new_reduce}, function));
        }
        old_reduce = new_reduce;
      }
      return ReplaceInstruction(reduce, old_reduce);
    }
  }

  HloInstruction *dot, *lhs, *rhs;
  // Convert Reduce(Dot(X,Y)) to Dot(X,Y) if any of the dimensions reduced were
  // batch dimensions of the dot. The transformation supports reducing other
  // dimensions as well.
  if (options_.supports_non_canonical_dots() &&
      options_.enable_dot_strength_reduction() &&
      Match(arg, m::Dot(&dot, m::Op(&lhs), m::Op(&rhs)).WithOneUser()) &&
      Match(reduce->to_apply()->root_instruction(),
            m::AddAnyOrder(m::Parameter(0), m::Parameter(1))) &&
      absl::c_any_of(reduce->dimensions(), [&](int64_t dim) {
        return dim < dot->dot_dimension_numbers().lhs_batch_dimensions_size();
      })) {
    const auto& dnums = dot->dot_dimension_numbers();
    DotDimensionNumbers new_dnums = dnums;
    new_dnums.clear_lhs_batch_dimensions();
    new_dnums.clear_rhs_batch_dimensions();
    int64_t removed_dims = 0;
    for (int64_t batch_dim = 0; batch_dim < dnums.lhs_batch_dimensions_size();
         ++batch_dim) {
      if (absl::c_linear_search(reduce->dimensions(), batch_dim)) {
        new_dnums.add_rhs_contracting_dimensions(
            dnums.rhs_batch_dimensions(batch_dim));
        new_dnums.add_lhs_contracting_dimensions(
            dnums.lhs_batch_dimensions(batch_dim));
        ++removed_dims;
      } else {
        new_dnums.add_rhs_batch_dimensions(
            dnums.rhs_batch_dimensions(batch_dim));
        new_dnums.add_lhs_batch_dimensions(
            dnums.lhs_batch_dimensions(batch_dim));
      }
    }
    std::vector<int64_t> reduce_dims;
    for (int64_t dim : reduce->dimensions()) {
      if (dim >= dnums.lhs_batch_dimensions_size()) {
        reduce_dims.push_back(dim - removed_dims);
      }
    }
    HloDotInstruction* dot_cast = Cast<HloDotInstruction>(dot);
    std::vector<SparsityDescriptor> sparsity(dot_cast->sparsity().begin(),
                                             dot_cast->sparsity().end());
    auto sparse_meta =
        absl::MakeSpan(dot->operands()).subspan(HloDotInstruction::kOperands);
    TF_ASSIGN_OR_RETURN(
        auto new_dot,
        MakeDotHlo(lhs, rhs, new_dnums, dot->precision_config(),
                   /*preferred_element_type=*/dot->shape().element_type(),
                   std::move(sparsity), sparse_meta));
    dot->SetupDerivedInstruction(new_dot);
    if (reduce_dims.empty()) {
      return ReplaceInstruction(hlo, new_dot);
    }
    TF_ASSIGN_OR_RETURN(
        auto new_reduce,
        MakeReduceHlo(new_dot, init_value, reduce_dims, HloOpcode::kAdd));
    reduce->SetupDerivedInstruction(new_reduce);
    return ReplaceInstruction(hlo, new_reduce);
  }

  // Replace Use(ReduceMax(Arg)) with Use(Gte(ReduceArgMax, 0)).
  // Match on Reduce Max with init value -Inf.
  if (reduce->operand_count() == 2 && IsScalarConstantNegInf(init_value) &&
      Match(reduce->to_apply()->root_instruction(),
            m::MaximumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    // Match on variadic Reduce ArgMax which is also fed by 'arg'.
    auto arg_max_candidate =
        absl::c_find_if(arg->users(), [&](const HloInstruction* user) {
          return user != reduce && user->operand(0) == arg &&
                 MatchArgMax(user) &&
                 reduce->dimensions() == user->dimensions();
        });
    if (arg_max_candidate != arg->users().end()) {
      // Replace 'reduce' uses with GTE(ArgMax, 0).
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateGetTupleElement(*arg_max_candidate,
                                                        /*index=*/0));
    }
  }

  // Replace Use(ReduceMin(Arg)) with Use(Gte(ReduceArgMin, 0)).
  // Match on Reduce Min with init value Inf.
  if (reduce->operand_count() == 2 && IsScalarConstantInf(init_value) &&
      Match(reduce->to_apply()->root_instruction(),
            m::MinimumAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    // Match on variadic Reduce ArgMin which is also fed by 'arg'.
    auto arg_min_candidate =
        absl::c_find_if(arg->users(), [&](const HloInstruction* user) {
          return user != reduce && user->operand(0) == arg &&
                 MatchArgMin(user) &&
                 reduce->dimensions() == user->dimensions();
        });
    if (arg_min_candidate != arg->users().end()) {
      // Replace 'reduce' uses with GTE(ArgMin, 0).
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateGetTupleElement(*arg_min_candidate,
                                                        /*index=*/0));
    }
  }

  // For Computation equal to Min, Max, And or Or, replace Reduce(Broadcast(x),
  // a, Computation()) with Computation(x, a) when x is a scalar and the
  // broadcast is reduced to a scalar.
  if (HloInstruction* broadcast_arg;
      Match(arg, m::Broadcast(m::Op(&broadcast_arg))) &&
      (Match(function->root_instruction(),
             m::MaximumAnyOrder(m::Parameter(0), m::Parameter(1))) ||
       Match(function->root_instruction(),
             m::MinimumAnyOrder(m::Parameter(0), m::Parameter(1))) ||
       Match(function->root_instruction(),
             m::AndAnyOrder(m::Parameter(0), m::Parameter(1))) ||
       Match(function->root_instruction(),
             m::OrAnyOrder(m::Parameter(0), m::Parameter(1))))) {
    if (broadcast_arg->shape().dimensions_size() == 0 &&
        reduce->dimensions().size() == arg->shape().dimensions_size()) {
      return ReplaceWithNewInstruction(
          reduce,
          HloInstruction::CreateBinary(
              reduce_result_shape, function->root_instruction()->opcode(),
              broadcast_arg, reduce->mutable_operand(1)));
    }
  }

  // Replace Reduce(Broadcast(x), +, init_value) with Broadcast(Add(Multiply(x),
  // init_value))) if all reduction dimensions were introduced by Broadcast
  if (arg->opcode() == HloOpcode::kBroadcast &&
      Match(reduce->to_apply()->root_instruction(),
            m::AddAnyOrder(m::Parameter(0), m::Parameter(1)))) {
    TF_RET_CHECK(
        std::is_sorted(arg->dimensions().begin(), arg->dimensions().end()))
        << "Broadcasts need to be canonicalized before algebraic "
           "simplification.";
    bool only_reduce_dims_from_broadcast = true;
    int64_t common_dims_prod = 1;
    int64_t num_common_dims = 0;
    Shape new_broadcast_shape = arg->shape();
    std::vector<int64_t> new_broadcast_dims;

    // Now we build up the new broadcast shape and dims vector
    for (int64_t i = 0; i < arg->shape().dimensions_size(); ++i) {
      bool added_by_broadcast = !absl::c_linear_search(arg->dimensions(), i);
      bool removed_by_reduce = absl::c_linear_search(reduce->dimensions(), i);

      if (removed_by_reduce && !added_by_broadcast) {
        only_reduce_dims_from_broadcast = false;
        break;
      } else if (removed_by_reduce && added_by_broadcast) {
        new_broadcast_shape.DeleteDimension(i - num_common_dims);
        common_dims_prod *= arg->shape().dimensions(i);
        num_common_dims++;
      } else if (!removed_by_reduce && !added_by_broadcast) {
        new_broadcast_dims.push_back(i - num_common_dims);
      }
    }

    if (only_reduce_dims_from_broadcast) {
      // HloConstantFolding will later remove any unnecessary multiply and add
      // instructions.
      HloInstruction* multiplier =
          MakeScalarLike(arg->mutable_operand(0), common_dims_prod);
      TF_ASSIGN_OR_RETURN(HloInstruction * multiplied_scalar,
                          MakeBinaryHlo(HloOpcode::kMultiply,
                                        arg->mutable_operand(0), multiplier));
      TF_ASSIGN_OR_RETURN(
          HloInstruction * add,
          MakeBinaryHlo(
              HloOpcode::kAdd,
              MakeBroadcastHlo(init_value, {}, multiplied_scalar->shape()),
              multiplied_scalar));
      VLOG(10) << "Converting common reduce(broadcast) dimensions to multiply";
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateBroadcast(new_broadcast_shape, add,
                                                  new_broadcast_dims));
    }
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleReducePrecision(
    HloInstruction* hlo) {
  HloReducePrecisionInstruction* reduce_precision =
      Cast<HloReducePrecisionInstruction>(hlo);
  PrimitiveType element_type =
      reduce_precision->operand(0)->shape().element_type();
  if (options_.enable_remove_no_op_reduce_precision() &&
      reduce_precision->exponent_bits() ==
          primitive_util::ExponentWidth(element_type) &&
      reduce_precision->mantissa_bits() + 1 ==
          primitive_util::SignificandWidth(element_type)) {
    return ReplaceInstruction(
        /*old_instruction=*/hlo,
        /*new_instruction=*/reduce_precision->mutable_operand(0));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleReduceWindow(
    HloInstruction* hlo) {
  auto* reduce_window = Cast<HloReduceWindowInstruction>(hlo);
  const bool multi_output_reduce_window = reduce_window->shape().IsTuple();
  auto inputs = reduce_window->inputs();
  auto init_values = reduce_window->init_values();
  auto input_count = reduce_window->input_count();
  auto input_shapes = reduce_window->input_shapes();
  auto output_shapes = reduce_window->output_shapes();
  auto replace_with_span = [&](const std::vector<HloInstruction*>& elements) {
    CHECK(multi_output_reduce_window || elements.size() == 1);
    if (multi_output_reduce_window) {
      return ReplaceWithNewInstruction(reduce_window,
                                       HloInstruction::CreateTuple(elements));
    }
    return ReplaceInstruction(reduce_window, elements[0]);
  };
  // For tuple reduce, we require all reduce shapes to be the same, up to the
  // element types, so we can use just the first operand and the first result as
  // a representative.
  if (ShapeUtil::IsZeroElementArray(*input_shapes[0]) ||
      ShapeUtil::IsZeroElementArray(*output_shapes[0])) {
    std::vector<HloInstruction*> broadcast_inits;
    broadcast_inits.reserve(input_count);
    for (int64_t i = 0; i < input_count; ++i) {
      broadcast_inits.push_back(
          hlo->AddInstruction(HloInstruction::CreateBroadcast(
              *output_shapes[i], init_values[i], {})));
    }
    return replace_with_span(broadcast_inits);
  }
  if (ShapeUtil::IsScalar(*input_shapes[0]) &&
      (!multi_output_reduce_window ||
       reduce_window->to_apply()->root_instruction()->opcode() ==
           HloOpcode::kTuple)) {
    std::vector<HloInstruction*> maps;
    for (int64_t i = 0; i < input_count; ++i) {
      TF_RET_CHECK(ShapeUtil::IsScalar(*input_shapes[i]));
      TF_RET_CHECK(ShapeUtil::IsScalar(*output_shapes[i]));
      HloInstruction* map_computation_root;
      absl::flat_hash_map<const HloInstruction*,
                          std::unique_ptr<HloInstruction>>
          replacements;
      if (multi_output_reduce_window) {
        map_computation_root =
            reduce_window->to_apply()->root_instruction()->mutable_operand(i);
        replacements[reduce_window->to_apply()->root_instruction()] = nullptr;
      } else {
        map_computation_root = reduce_window->to_apply()->root_instruction();
      }
      maps.push_back(inputs[i]);
    }
    return replace_with_span(maps);
  }
  // Turn trivial variadic reduce windows into normal reduce windows.
  auto reduce_function_root = reduce_window->to_apply()->root_instruction();
  if (multi_output_reduce_window && input_count == 1 &&
      Match(reduce_function_root, m::Tuple())) {
    // Make a new reducer which is identical but does not have a tuple
    // instruction at the bottom.
    absl::flat_hash_map<const HloInstruction*, std::unique_ptr<HloInstruction>>
        replacements;
    replacements[reduce_function_root] = nullptr;
    auto new_function = computation_->parent()->AddEmbeddedComputation(
        reduce_window->to_apply()->CloneWithReplacements(
            &replacements, /*extra_parameters=*/{},
            /*context=*/nullptr,
            /*suffix=*/"clone",
            /*new_root=*/reduce_function_root->operand(0)));
    auto new_reduce_window =
        reduce_window->AddInstruction(HloInstruction::CreateReduceWindow(
            *output_shapes[0], inputs[0], init_values[0],
            reduce_window->window(), new_function));
    return ReplaceWithNewInstruction(
        reduce_window, HloInstruction::CreateTuple({new_reduce_window}));
  }
  if (multi_output_reduce_window) {
    return absl::OkStatus();
  }
  auto operand = reduce_window->mutable_operand(0);
  auto init_value = reduce_window->mutable_operand(1);
  auto function = reduce_window->to_apply();
  const Window& window = reduce_window->window();

  // reduce-window with a 1x1x..x1 window and no dilation etc can be replaced
  // with a trivial elementwise operation, plus a pad op if necessary.
  //
  // We cowardly refuse to consider this optimization when the reduce-window
  // subcomputation is anything other than a simple add/min/max.  Supporting
  // more complex subcomputations is possible, but is tantamount to implementing
  // jax.vmap()!
  if (absl::c_all_of(window.dimensions(),
                     [](const WindowDimension& dim) {
                       return dim.size() == 1 &&             //
                              dim.stride() == 1 &&           //
                              dim.window_dilation() == 1 &&  //
                              dim.base_dilation() == 1 &&    //
                              !dim.window_reversal();
                     }) &&
      Match(function->root_instruction(),
            m::AnyOf<HloInstruction>(
                m::AddAnyOrder(m::Parameter(0), m::Parameter(1)),
                m::MinimumAnyOrder(m::Parameter(0), m::Parameter(1)),
                m::MaximumAnyOrder(m::Parameter(0), m::Parameter(1))))) {
    const HloInstruction* nested_root = function->root_instruction();
    DimensionVector broadcast_dims(nested_root->shape().dimensions_size());
    absl::c_iota(broadcast_dims, 0);
    TF_ASSIGN_OR_RETURN(
        auto new_op, MakeBinaryHlo(nested_root->opcode(), operand,
                                   MakeBroadcastHlo(init_value, broadcast_dims,
                                                    operand->shape())));

    if (absl::c_any_of(window.dimensions(), [](const WindowDimension& dim) {
          return dim.padding_low() > 0 || dim.padding_high() > 0;
        })) {
      PaddingConfig padding_config;
      for (const WindowDimension& window_dim : window.dimensions()) {
        auto& padding_dim = *padding_config.add_dimensions();
        padding_dim.set_edge_padding_low(window_dim.padding_low());
        padding_dim.set_edge_padding_high(window_dim.padding_high());
        padding_dim.set_interior_padding(0);
      }
      TF_ASSIGN_OR_RETURN(new_op,
                          MakePadHlo(new_op, init_value, padding_config));
    }

    return ReplaceInstruction(reduce_window, new_op);
  }

  if (options_.enable_window_reduce_to_reduce_replacement()) {
    // A reduce window can be expressed as a reduce and a reshape if all
    // dimensions either have a window size of one or the entire dimension. If
    // there is no stride, dilation, or padding, this is as easy as checking the
    // size of the output shape and window dimension.
    //
    // The reshape is a bitcast since it adds one-sized dimensions. Often these
    // ones are immediately removed as well with another reshape. The
    // implementation of reduce tends to be slightly more efficient at reducing
    // entire dimensions compared to reduce window.
    auto effective_reduce_dims = [&] {
      if (window_util::HasStride(window) || window_util::HasDilation(window) ||
          window_util::HasPadding(window)) {
        return DimensionVector{};
      }
      DimensionVector reduce_dims;
      for (int64_t i = 0; i < window.dimensions_size(); ++i) {
        if (window.dimensions(i).size() == 1) {
          continue;
        } else if (reduce_window->shape().dimensions(i) == 1) {
          reduce_dims.push_back(i);
        } else {
          return DimensionVector{};
        }
      }
      return reduce_dims;
    }();

    // If a reduce window can be expressed as a reduce, do so and reshape the
    // output.
    if (!effective_reduce_dims.empty()) {
      Shape reduce_shape = ShapeUtil::DeleteDimensions(effective_reduce_dims,
                                                       reduce_window->shape());
      simplifier_->UpdateLayout(&reduce_shape);
      HloInstruction* reduce = hlo->AddInstruction(HloInstruction::CreateReduce(
          /*shape=*/reduce_shape,
          /*operand=*/operand,
          /*init_value=*/reduce_window->mutable_operand(1),
          /*dimensions_to_reduce=*/effective_reduce_dims,
          /*reduce_computation=*/function));
      return ReplaceWithNewInstruction(
          reduce_window,
          HloInstruction::CreateReshape(reduce_window->shape(), reduce));
    }
  }

  // This optimization folds a pad op into reduce_window.
  HloInstruction* pad;
  const HloInstruction* convert = nullptr;
  if (operand->opcode() == HloOpcode::kPad) {
    pad = operand;
  } else if (operand->opcode() == HloOpcode::kConvert &&
             operand->operand(0)->opcode() == HloOpcode::kPad) {
    convert = operand;
    pad = operand->mutable_operand(0);
  } else {
    VLOG(10) << "Not folding pad into reduce-window as there is no pad.";
    return absl::OkStatus();
  }

  VLOG(10) << "Considering folding Pad: " << pad->ToString()
           << "\ninto reduce-window: " << reduce_window->ToString()
           << (convert != nullptr
                   ? absl::StrCat("\nvia convert: ", convert->ToString())
                   : "");

  // Do not fold interior padding into ReduceWindow since the backends do not
  // support it.
  const PaddingConfig& pad_config = pad->padding_config();
  if (HasInteriorPadding(pad_config) && window_util::HasBaseDilation(window)) {
    VLOG(10) << "Not folding interior pad into base-dilated reduce-window.";
    return absl::OkStatus();
  }

  // If reduce_window already has padding, the pad value of the pad op and the
  // init value of reduce_window must match to allow folding the pad.
  const HloInstruction* pad_value = pad->operand(1);
  const HloInstruction* reduce_init_value = reduce_window->operand(1);
  if (pad_value != reduce_init_value) {
    auto literals_are_equivalent = [&] {
      auto& pad_literal = pad_value->literal();
      auto& reduce_init_literal = reduce_init_value->literal();
      if (pad_literal == reduce_init_literal) {
        return true;
      }
      auto converted_pad_literal =
          pad_literal.ConvertToShape(reduce_init_value->shape());
      if (!converted_pad_literal.ok()) {
        return false;
      }
      return converted_pad_literal.value() == reduce_init_literal;
    };
    // The pad value is usually a constant, so we handle that case and do not
    // try to get more fancy about proving equivalence in cases beyond that.
    if (pad_value->opcode() != HloOpcode::kConstant ||
        reduce_init_value->opcode() != HloOpcode::kConstant ||
        !literals_are_equivalent()) {
      VLOG(10) << "Not folding pad into reduce-window due to different pad "
                  "values.";
      return absl::OkStatus();
    }
  }

  // If the pad puts a single non-identity value in each window that we're
  // reducing, then this is a broadcast.
  HloInstruction* pad_operand = pad->mutable_operand(0);
  auto is_effective_broadcast = [&] {
    if (window_util::HasStride(window)) {
      VLOG(10) << "Window has stride.";
      return false;
    }
    if (!window_util::HasSymmetricPadding(pad_config)) {
      VLOG(10) << "Window has uneven padding.";
      return false;
    }
    if (HasInteriorPadding(pad_config)) {
      VLOG(10) << "Window has interior padding.";
      return false;
    }
    for (int64_t i = 0; i < pad_config.dimensions_size(); ++i) {
      const auto& pad_dimension = pad_config.dimensions(i);
      if ((pad_dimension.edge_padding_low() != 0 ||
           pad_dimension.edge_padding_high() != 0) &&
          pad_operand->shape().dimensions(i) != 1) {
        VLOG(10) << "Found non-trivial dimension being padded: " << i;
        return false;
      }
    }
    VLOG(10) << "Found to be padding trivial dimensions only.";

    for (int64_t i = 0; i < window.dimensions_size(); ++i) {
      const auto& pad_dimension = pad_config.dimensions(i);
      const WindowDimension& window_dimension = window.dimensions(i);
      bool dimension_has_padding = (pad_dimension.edge_padding_low() != 0 ||
                                    pad_dimension.edge_padding_high() != 0);
      if (dimension_has_padding &&
          window_dimension.size() < pad_dimension.edge_padding_low() + 1) {
        VLOG(10) << "Found window did not cover single unpadded element in "
                    "dimension: "
                 << i;
        return false;
      }
      if (pad_operand->shape().dimensions(i) != 1 &&
          window_dimension.size() != 1) {
        VLOG(10) << "Found window covers more than one element in non-trivial "
                    "dimension: "
                 << i;
        return false;
      }
    }
    VLOG(10) << "Found window covers a single unpadded element.";
    return true;
  };

  HloInstruction* new_reduce_window_operand;
  if (convert != nullptr) {
    Shape changed_shape = ShapeUtil::ChangeElementType(
        pad_operand->shape(), convert->shape().element_type());
    simplifier_->UpdateLayout(&changed_shape);
    new_reduce_window_operand = hlo->AddInstruction(
        HloInstruction::CreateConvert(changed_shape, pad_operand));
  } else {
    new_reduce_window_operand = pad_operand;
  }

  if (is_effective_broadcast()) {
    VLOG(10) << "Replacing pad/reduce-window with broadcast.";
    auto fadd = [hlo](std::unique_ptr<HloInstruction> x) {
      return hlo->AddInstruction(std::move(x));
    };
    return ReplaceWithNewInstruction(
        reduce_window, HloInstruction::CreateBroadcastSequence(
                           /*output_shape=*/reduce_window->shape(),
                           /*operand=*/new_reduce_window_operand, fadd));
  }

  // Carry out the folding of the pad into reduce_window.
  VLOG(10) << "Folding pad into reduce-window.";
  Window new_window = window;
  const int64_t rank = reduce_window->shape().dimensions_size();
  TF_RET_CHECK(pad_config.dimensions_size() == rank);
  TF_RET_CHECK(window.dimensions_size() == rank);
  for (int64_t i = 0; i < rank; ++i) {
    const auto& pad_dim = pad_config.dimensions(i);
    auto& window_dim = *new_window.mutable_dimensions(i);
    window_dim.set_padding_low(window_dim.padding_low() +
                               window_dim.base_dilation() *
                                   pad_dim.edge_padding_low());
    window_dim.set_padding_high(window_dim.padding_high() +
                                window_dim.base_dilation() *
                                    pad_dim.edge_padding_high());
    if (pad_dim.interior_padding() != 0) {
      CHECK_EQ(window_dim.base_dilation(), 1);
      window_dim.set_base_dilation(1 + pad_dim.interior_padding());
    }
  }

  return ReplaceWithNewInstruction(
      reduce_window, HloInstruction::CreateReduceWindow(
                         /*shape=*/reduce_window->shape(),
                         /*operand=*/new_reduce_window_operand,
                         /*init_value=*/reduce_window->mutable_operand(1),
                         /*window=*/new_window,
                         /*reduce_computation=*/function));
}

absl::Status AlgebraicSimplifierVisitor::HandleSelect(HloInstruction* select) {
  // select(x, y, y) -> y.
  if (select->operand(1) == select->operand(2) &&
      ReplaceInstructionIfCompatible(select, select->mutable_operand(1))) {
    return absl::OkStatus();
  }
  // select(true, x, y) -> x.
  if (IsAll(select->operand(0), true) &&
      ReplaceInstructionIfCompatible(select, select->mutable_operand(1))) {
    return absl::OkStatus();
  }
  // select(false, x, y) -> y.
  if (IsAll(select->operand(0), false) &&
      ReplaceInstructionIfCompatible(select, select->mutable_operand(2))) {
    return absl::OkStatus();
  }
  // select(not(pred), a, b) -> select(pred, b, a)
  if (HloOpcode::kNot == select->operand(0)->opcode()) {
    auto pred_operand = select->mutable_operand(0)->mutable_operand(0);
    auto on_true = select->mutable_operand(1);
    auto on_false = select->mutable_operand(2);
    return ReplaceWithNewInstruction(
        select,
        HloInstruction::CreateTernary(select->shape(), HloOpcode::kSelect,
                                      pred_operand, on_false, on_true));
  }
  // select(PRED, PRED, PRED)
  if (ShapeUtil::HasPrimitiveType(select->shape(), xla::PRED)) {
    // select(a, true, false) -> a
    if (IsAll(select->operand(1), true) && IsAll(select->operand(2), false)) {
      return ReplaceInstruction(select, select->mutable_operand(0));
    }
    // select(a, false, true) -> not(a)
    if (IsAll(select->operand(1), false) && IsAll(select->operand(2), true)) {
      return ReplaceWithNewInstruction(
          select, HloInstruction::CreateUnary(
                      select->mutable_operand(0)->shape(), HloOpcode::kNot,
                      select->mutable_operand(0)));
    }
    // select(compare(a, b, GT/GE), a, b) => or(a, b)
    // select(compare(a, b, LT/LE), a, b) => and(a, b)
    // select(compare(a, b, EQ), a, b) => b
    // select(compare(a, b, NE), a, b) => a
    HloInstruction *compare, *lhs, *rhs;
    if (Match(select, m::Select(m::Op(&compare), m::Op(&lhs), m::Op(&rhs))) &&
        Match(compare, m::Compare(m::Op().Is(lhs), m::Op().Is(rhs)))) {
      auto cmp_dir = compare->comparison_direction();
      if (cmp_dir == ComparisonDirection::kGt ||
          cmp_dir == ComparisonDirection::kGe) {
        return ReplaceWithNewInstruction(
            select, HloInstruction::CreateBinary(select->shape(),
                                                 HloOpcode::kOr, lhs, rhs));
      }
      if (cmp_dir == ComparisonDirection::kLt ||
          cmp_dir == ComparisonDirection::kLe) {
        return ReplaceWithNewInstruction(
            select, HloInstruction::CreateBinary(select->shape(),
                                                 HloOpcode::kAnd, lhs, rhs));
      }
      if (cmp_dir == ComparisonDirection::kEq) {
        return ReplaceInstruction(select, rhs);
      }
      if (cmp_dir == ComparisonDirection::kNe) {
        return ReplaceInstruction(select, lhs);
      }
    }
  }

  // select(pred, xs, dynamic_update_slice(xs, x, i))
  //     -> dynamic_update_slice(xs, select(pred, dynamic_slice(xs, i), x), i)
  HloInstruction* update_slice;
  HloInstruction* xs;
  HloInstruction* xs2;
  auto update_slice_op = m::Op(&update_slice)
                             .WithOpcode(HloOpcode::kDynamicUpdateSlice)
                             .WithOperand(0, m::Op(&xs))
                             .WithOneUse();
  bool match_slice_left =
      Match(select, m::Select(m::Op(), m::Op(&xs2), update_slice_op)) &&
      (xs == xs2);
  bool match_slice_right =
      Match(select, m::Select(m::Op(), update_slice_op, m::Op(&xs2))) &&
      (xs == xs2);
  if (match_slice_left || match_slice_right) {
    HloInstruction* pred = select->mutable_operand(0);
    HloInstruction* x = update_slice->mutable_operand(1);
    absl::Span<HloInstruction* const> i =
        absl::MakeSpan(update_slice->operands()).subspan(2);
    HloInstruction* new_pred;
    if (ShapeUtil::IsScalar(pred->shape())) {
      new_pred = pred;
    } else {
      Shape new_pred_shape = x->shape();
      new_pred_shape.set_element_type(pred->shape().element_type());
      simplifier_->UpdateLayout(&new_pred_shape);
      new_pred = select->AddInstruction(HloInstruction::CreateDynamicSlice(
          new_pred_shape, pred, i, x->shape().dimensions()));
    }
    HloInstruction* new_x =
        select->AddInstruction(HloInstruction::CreateDynamicSlice(
            x->shape(), xs, i, x->shape().dimensions()));
    HloInstruction* new_x2 =
        select->AddInstruction(HloInstruction::CreateTernary(
            x->shape(), HloOpcode::kSelect, new_pred,
            match_slice_left ? new_x : x, match_slice_left ? x : new_x));
    std::unique_ptr<HloInstruction> new_xs =
        HloInstruction::CreateDynamicUpdateSlice(select->shape(), xs, new_x2,
                                                 i);
    return ReplaceWithNewInstruction(select, std::move(new_xs));
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleScatter(HloInstruction* hlo) {
  auto* scatter = Cast<HloScatterInstruction>(hlo);

  if (absl::c_all_of(scatter->scatter_updates(),
                     [](const HloInstruction* updates) {
                       return ShapeUtil::IsZeroElementArray(updates->shape());
                     }) &&
      ReplaceInstructionIfCompatible(scatter, scatter->scatter_operands())) {
    return absl::OkStatus();
  }
  if (scatter->scatter_operand_count() == 1 &&
      ShapeUtil::IsZeroElementArray(scatter->scatter_indices()->shape()) &&
      SameShape(scatter, scatter->scatter_operands()[0]) &&
      SameShape(scatter, scatter->scatter_updates()[0])) {
    return ReplaceWithNewInstruction(
        scatter, HloInstruction::CreateMap(scatter->shape(),
                                           {scatter->scatter_operands()[0],
                                            scatter->scatter_updates()[0]},
                                           scatter->to_apply()));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleSort(HloInstruction* sort) {
  auto operand = sort->mutable_operand(0);
  int64_t dimension_to_sort = sort->dimensions(0);
  if (ShapeUtil::IsZeroElementArray(operand->shape()) ||
      operand->shape().dimensions(dimension_to_sort) <= 1) {
    if (sort->operand_count() == 1) {
      return ReplaceInstruction(sort, operand);
    }
    // If it is key/value sort, the output of sort is a tuple.
    return ReplaceWithNewInstruction(
        sort, HloInstruction::CreateTuple(sort->operands()));
  }
  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleSqrt(HloInstruction* sqrt) {
  VLOG(10) << "trying transform [sqrt(A*A) => |A|] " << sqrt->ToString();
  HloInstruction* sqrt_operand = sqrt->mutable_operand(0);
  if (sqrt_operand->opcode() == HloOpcode::kMultiply &&
      sqrt_operand->operand(0) == sqrt_operand->operand(1)) {
    PrimitiveType element_type = sqrt_operand->shape().element_type();
    // For 'A' of type C{64,128}, |A| has type F{32,64}, and the transformation
    // requires an additional cast.
    if (primitive_util::IsComplexType(element_type)) {
      auto abs_shape = sqrt_operand->shape();
      abs_shape.set_element_type(
          primitive_util::ComplexComponentType(element_type));

      HloInstruction* abs =
          sqrt->parent()->AddInstruction(HloInstruction::CreateUnary(
              abs_shape, HloOpcode::kAbs, sqrt_operand->mutable_operand(0)));

      return ReplaceWithNewInstruction(
          sqrt, HloInstruction::CreateConvert(sqrt_operand->shape(), abs));
    }
    return ReplaceWithNewInstruction(
        sqrt, HloInstruction::CreateUnary(
                  sqrt_operand->mutable_operand(0)->shape(), HloOpcode::kAbs,
                  sqrt_operand->mutable_operand(0)));
  }
  return absl::OkStatus();
}
namespace {
bool OnlyPermutesDegenerateDims(const Shape& shape,
                                absl::Span<const int64_t> perm) {
  std::vector<int64_t> new_permutation;
  int64_t degenerate_count = 0;
  for (int64_t i = 0; i < perm.size(); ++i) {
    if (shape.dimensions(i) != 1) {
      new_permutation.push_back(perm[i]);
    } else {
      ++degenerate_count;
    }
  }
  return degenerate_count > 0 && absl::c_is_sorted(new_permutation);
}

bool IsPermutationOfIota(absl::Span<const int64_t> elems) {
  DimensionVector sorted(elems.begin(), elems.end());
  absl::c_sort(sorted);
  for (int i = 0; i < sorted.size(); i++) {
    if (sorted[i] != i) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::Status AlgebraicSimplifierVisitor::HandleTranspose(
    HloInstruction* transpose) {
  auto operand = transpose->mutable_operand(0);
  if (std::is_sorted(transpose->dimensions().begin(),
                     transpose->dimensions().end())) {
    VLOG(10) << "deleting no-op transpose";
    return ReplaceInstruction(transpose, operand);
  }

  if (HloOpcode::kTranspose == operand->opcode()) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateTranspose(
                       transpose->shape(), operand->mutable_operand(0),
                       ComposePermutations(operand->dimensions(),
                                           transpose->dimensions())));
  }

  const auto consider_swapping_dot_operands = [&](HloInstruction* dot) {
    // If the RHS is a parameter-like, and the LHS is not, do not swap the
    // operands, since the dot operands are in a convenient order for layout
    // assignment (even if we have to transpose the batch dimensions of the
    // output).
    return !(options_.enable_move_dot_param_to_rhs() &&
             !IsParameterLike(dot->operand(0)) &&
             IsParameterLike(dot->operand(1)));
  };

  // Convert transpose(dot(a,b)) to dot(b,a).
  auto do_transpose_of_dot = [&]() -> absl::StatusOr<bool> {
    if (options_.supports_non_canonical_dots() ||
        operand->opcode() != HloOpcode::kDot || operand->user_count() != 1 ||
        Cast<HloDotInstruction>(operand)->sparse_operands()) {
      return false;
    }

    if (!consider_swapping_dot_operands(operand)) {
      return false;
    }

    HloInstruction* dot = operand;
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);

    const int64_t rank = dot->shape().dimensions_size();
    const auto& dnums = dot->dot_dimension_numbers();

    // Dot must be "somewhat canonical": batch dimensions at the beginning and
    // one non-contracting dim.  It's the responsibility of DotDecomposer to
    // canonicalize dots.
    if (absl::MakeSpan(dnums.lhs_batch_dimensions()) !=
            absl::MakeSpan(dnums.rhs_batch_dimensions()) ||
        !IsPermutationOfIota(dnums.lhs_batch_dimensions()) ||
        dnums.lhs_contracting_dimensions_size() == 0 ||
        dnums.lhs_contracting_dimensions_size() +
                dnums.lhs_batch_dimensions_size() + 1 !=
            lhs->shape().dimensions_size() ||
        dnums.rhs_contracting_dimensions_size() == 0 ||
        dnums.rhs_contracting_dimensions_size() +
                dnums.rhs_batch_dimensions_size() + 1 !=
            rhs->shape().dimensions_size()) {
      return false;
    }

    // Transpose must just be over the two last dims (i.e. the non-batch dims).
    DimensionVector expected_perm(rank);
    absl::c_iota(expected_perm, 0);
    std::swap(expected_perm.rbegin()[0], expected_perm.rbegin()[1]);
    if (transpose->dimensions() != expected_perm) {
      return false;
    }

    DotDimensionNumbers new_dnums = dnums;
    std::swap(*new_dnums.mutable_lhs_contracting_dimensions(),
              *new_dnums.mutable_rhs_contracting_dimensions());
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        transpose,
        HloInstruction::CreateDot(
            transpose->shape(), /*lhs=*/rhs, /*rhs=*/lhs, new_dnums,
            SwapOperandsInDotPrecisionConfig(dot->precision_config()))));
    return true;
  };
  TF_ASSIGN_OR_RETURN(bool did_transpose_of_dot, do_transpose_of_dot());
  if (did_transpose_of_dot) {
    return absl::OkStatus();
  }

  // Transpose(dot(a,b))->dot(b,a) for any dot.
  HloInstruction *lhs, *rhs, *dot;
  if (options_.supports_non_canonical_dots() &&
      Match(operand, m::Dot(&dot, m::Op(&lhs), m::Op(&rhs))) &&
      dot->user_count() == 1 &&
      !Cast<HloDotInstruction>(dot)->sparse_operands()) {
    TF_ASSIGN_OR_RETURN(bool did_transform, [&]() -> absl::StatusOr<bool> {
      if (!consider_swapping_dot_operands(operand)) {
        return false;
      }

      const auto& dnums = dot->dot_dimension_numbers();
      const int64_t num_batch_dims = dnums.lhs_batch_dimensions_size();
      for (int64_t i = 0; i < num_batch_dims; ++i) {
        if (transpose->dimensions(i) >= num_batch_dims) {
          return false;
        }
      }
      const int64_t num_rhs_outer_dims =
          rhs->shape().dimensions_size() -
          (dnums.rhs_contracting_dimensions_size() +
           dnums.rhs_batch_dimensions_size());
      const int64_t num_lhs_outer_dims =
          lhs->shape().dimensions_size() -
          (dnums.lhs_contracting_dimensions_size() +
           dnums.lhs_batch_dimensions_size());
      for (int64_t i = 0; i < num_rhs_outer_dims; ++i) {
        if (transpose->dimensions(i + num_batch_dims) !=
            i + num_batch_dims + num_lhs_outer_dims) {
          return false;
        }
      }
      for (int64_t i = 0; i < num_lhs_outer_dims; ++i) {
        if (transpose->dimensions(i + num_batch_dims + num_rhs_outer_dims) !=
            i + num_batch_dims) {
          return false;
        }
      }
      DotDimensionNumbers new_dnums;
      *new_dnums.mutable_lhs_contracting_dimensions() =
          dnums.rhs_contracting_dimensions();
      *new_dnums.mutable_rhs_contracting_dimensions() =
          dnums.lhs_contracting_dimensions();
      for (int64_t batch_dim = 0; batch_dim < num_batch_dims; ++batch_dim) {
        new_dnums.add_lhs_batch_dimensions(
            dnums.rhs_batch_dimensions(transpose->dimensions(batch_dim)));
        new_dnums.add_rhs_batch_dimensions(
            dnums.lhs_batch_dimensions(transpose->dimensions(batch_dim)));
      }

      HloInstruction* new_dot =
          MakeDotHlo(rhs, lhs, new_dnums,
                     SwapOperandsInDotPrecisionConfig(dot->precision_config()),
                     dot->shape().element_type())
              .value();
      *new_dot->mutable_shape()->mutable_layout() = transpose->shape().layout();

      dot->SetupDerivedInstruction(new_dot);
      TF_CHECK_OK(ReplaceInstruction(transpose, new_dot));
      return true;
    }());
    if (did_transform) {
      return absl::OkStatus();
    }
  }

  // Replace transpose with a reshape if more than one degenerate method is
  // permuted.
  if (OnlyPermutesDegenerateDims(transpose->shape(), transpose->dimensions())) {
    return ReplaceWithNewInstruction(
        transpose, HloInstruction::CreateReshape(
                       transpose->shape(), transpose->mutable_operand(0)));
  }

  if (operand->opcode() == HloOpcode::kRng && operand->user_count() == 1) {
    *operand->mutable_shape() = transpose->shape();
    return ReplaceInstruction(transpose, operand);
  }

  if (options_.is_layout_sensitive() && TransposeIsBitcast(transpose)) {
    ReplaceWithBitcast(transpose);
    return absl::OkStatus();
  }

  // Replace reshape of a transpose of a reshape with concatenated slicing if
  // the reshape/transpose combination can be interpreted as a space-to-depth
  // transformation.
  if (!options_.is_layout_sensitive() &&
      options_.rewrite_reshape_transpose_as_slice_concatenate() &&
      operand->opcode() == HloOpcode::kReshape &&
      transpose->user_count() == 1 &&
      HloOpcode::kReshape == transpose->users()[0]->opcode()) {
    VLOG(2) << "trying depth-to-space transform";
    HloInstruction* reshape_operand = operand->mutable_operand(0);
    HloInstruction* outer_reshape = transpose->users()[0];
    TF_ASSIGN_OR_RETURN(
        bool did_transform, ([&]() -> absl::StatusOr<bool> {
          if (operand->shape().dimensions_size() !=
              reshape_operand->shape().dimensions_size() + 1) {
            return false;
          }

          // Check that the reshape is splitting a single dimension into two.
          int64_t split_dim = 0;
          bool found_split_dims = false;
          for (int64_t dim = 0;
               dim < reshape_operand->shape().dimensions_size(); dim++) {
            if (operand->shape().dimensions(dim) !=
                reshape_operand->shape().dimensions(dim)) {
              const int64_t expected_size =
                  operand->shape().dimensions(dim) *
                  operand->shape().dimensions(dim + 1);
              if (reshape_operand->shape().dimensions(dim) == expected_size) {
                split_dim = dim;
                found_split_dims = true;
                break;
              }
              return false;
            }
          }
          if (!found_split_dims) {
            return false;
          }
          for (int64_t dim = split_dim + 1;
               dim < reshape_operand->shape().dimensions_size(); dim++) {
            if (operand->shape().dimensions(dim + 1) !=
                reshape_operand->shape().dimensions(dim)) {
              return false;
            }
          }

          const int64_t num_chunks = operand->shape().dimensions(split_dim);
          const int64_t chunk_size = operand->shape().dimensions(split_dim + 1);

          // This optimization is only beneficial for a small number of chunks.
          // TODO(b/196832483): Determine the appropriate upper bound here.
          const int64_t kMaxChunksForTransformation = 5;
          if (num_chunks > kMaxChunksForTransformation) {
            return false;
          }

          // Determine where the smaller split dimension is being placed in the
          // transpose
          int64_t transpose_dim = 0;
          bool found_transpose_dim = false;
          for (int64_t dim = 0; dim < operand->shape().dimensions_size();
               dim++) {
            if (transpose->dimensions(dim) == split_dim) {
              transpose_dim = dim;
              found_transpose_dim = true;
              break;
            }
          }

          // Check that only the small split dimension is reordered in the
          // transpose
          if (!found_transpose_dim || transpose_dim == split_dim ||
              transpose_dim == split_dim + 1) {
            return false;
          }
          for (int64_t dim = 0; dim < operand->shape().dimensions_size();
               dim++) {
            int64_t offset = 0;
            if (dim > transpose_dim) {
              offset--;
            }
            if (dim > split_dim) {
              offset++;
            }

            if (dim != transpose_dim &&
                transpose->dimensions(dim) != dim + offset) {
              return false;
            }
          }

          // Check that the outer reshape has the same shape as the input,
          // with the transformed dimensions appropriately scaled by num_chunks.
          for (int64_t dim = 0;
               dim < reshape_operand->shape().dimensions_size(); dim++) {
            if (dim == transpose_dim - 1) {
              if (outer_reshape->shape().dimensions(dim) !=
                  reshape_operand->shape().dimensions(dim) * num_chunks) {
                return false;
              }
            } else if (dim == split_dim) {
              if (outer_reshape->shape().dimensions(dim) !=
                  reshape_operand->shape().dimensions(dim) / num_chunks) {
                return false;
              }
            } else if (outer_reshape->shape().dimensions(dim) !=
                       reshape_operand->shape().dimensions(dim)) {
              return false;
            }
          }

          // Create a concat-of-slices, slicing to create chunks of the expected
          // size on the smaller split dimension.
          std::vector<HloInstruction*> slices;
          for (int64_t i = 0; i < num_chunks; i++) {
            std::vector<int64_t> start_indices;
            std::vector<int64_t> end_indices;
            std::vector<int64_t> strides;
            const auto rank = reshape_operand->shape().dimensions_size();
            start_indices.reserve(rank);
            end_indices.reserve(rank);
            strides.reserve(rank);
            for (int64_t dim = 0; dim < rank; dim++) {
              if (dim == split_dim) {
                start_indices.push_back(i * chunk_size);
                end_indices.push_back(i * chunk_size + chunk_size);
              } else {
                start_indices.push_back(0);
                end_indices.push_back(reshape_operand->shape().dimensions(dim));
              }
              strides.push_back(1);
            }
            TF_ASSIGN_OR_RETURN(HloInstruction* const slice,
                                MakeSliceHlo(reshape_operand, start_indices,
                                             end_indices, strides));
            slices.push_back(slice);
            VLOG(2) << "slice " << i << " " << slice->ToString();
          }

          TF_ASSIGN_OR_RETURN(HloInstruction* const concat,
                              MakeConcatHlo(slices, transpose_dim));
          VLOG(2) << "concat " << concat->ToString();
          TF_RETURN_IF_ERROR(
              outer_reshape->ReplaceOperandWithDifferentShape(0, concat));

          return true;
        }()));
    if (did_transform) {
      MarkAsChanged();
      return absl::OkStatus();
    }
  }

  TF_RETURN_IF_ERROR(
      SimplifyTransposeOfBroadcast(transpose, transpose->dimensions()));

  return absl::OkStatus();
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvInputPad(
    HloInstruction* convolution) {
  HloInstruction *lhs, *a, *b;
  if (Match(convolution,
            m::Convolution(m::Pad(&lhs, m::Op(&a), m::ConstantScalar(0)),
                           m::Op(&b)))) {
    const auto& window = convolution->window();
    const ConvolutionDimensionNumbers& dnums =
        convolution->convolution_dimension_numbers();

    const auto& padding = lhs->padding_config();

    // Can't pad batch or feature dims.
    for (int64_t dim :
         {dnums.input_batch_dimension(), dnums.input_feature_dimension()}) {
      const auto& p = padding.dimensions(dim);
      if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
          p.interior_padding() != 0) {
        return false;
      }
    }

    // Compute the window which is the result of merging the kPad and the
    // convolution's existing window.
    Window new_window = window;
    for (int64_t dim = 0; dim < dnums.input_spatial_dimensions_size(); ++dim) {
      auto& w = *new_window.mutable_dimensions(dim);
      const auto& p = padding.dimensions(dnums.input_spatial_dimensions(dim));
      // Edge padding composes with itself in the straightforward way, but
      // composing interior padding is nontrivial, and we cowardly refuse to
      // think about it. If we see interior padding in either the kPad or conv,
      // bail if there's any sort of padding in the other.
      if (p.interior_padding() != 0 &&
          (w.padding_low() != 0 || w.padding_high() != 0 ||
           w.base_dilation() != 1)) {
        return false;
      }
      if (w.base_dilation() != 1 &&
          (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
           p.interior_padding() != 0)) {
        return false;
      }

      w.set_padding_low(w.padding_low() + p.edge_padding_low());
      w.set_padding_high(w.padding_high() + p.edge_padding_high());
      if (p.interior_padding() != 0) {
        CHECK_EQ(w.base_dilation(), 1);
        w.set_base_dilation(1 + p.interior_padding());
      }
    }

    auto new_conv =
        convolution->CloneWithNewOperands(convolution->shape(), {a, b});
    new_conv->set_window(new_window);
    TF_RETURN_IF_ERROR(
        ReplaceWithNewInstruction(convolution, std::move(new_conv)));
    return true;
  }
  return false;
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvFilterPad(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (rhs->opcode() != HloOpcode::kPad) {
    return false;
  }

  // Convolution's padding is always zero, so bail if the kPad is adding
  // something other than zero.
  if (!IsAll(rhs->operand(1), 0)) {
    return false;
  }

  const auto& padding = rhs->padding_config();

  // Can't pad or dilate feature dims.
  for (int64_t dim : {dnums.kernel_input_feature_dimension(),
                      dnums.kernel_output_feature_dimension()}) {
    const auto& p = padding.dimensions(dim);
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0 ||
        p.interior_padding() != 0) {
      return false;
    }
  }

  // Compute the window which is the result of merging the kPad and the
  // convolution's existing window.
  Window new_window = convolution->window();
  for (int64_t dim = 0; dim < dnums.kernel_spatial_dimensions_size(); ++dim) {
    auto& w = *new_window.mutable_dimensions(dim);
    const auto& p = padding.dimensions(dnums.kernel_spatial_dimensions(dim));

    // We can only do this transformation if p adds dilation to the filter --
    // edge padding on the filter is not supported in conv.
    if (p.edge_padding_low() != 0 || p.edge_padding_high() != 0) {
      return false;
    }

    // Nothing to do if the kPad for this dim is entirely a nop.
    if (p.interior_padding() == 0) {
      continue;
    }

    // We cowardly refuse to think about how dilation composes with itself;
    // bail if both the kPad and conv have dilation on this dimension.
    if (w.window_dilation() > 1) {
      return false;
    }
    CHECK_EQ(w.window_dilation(), 1);
    w.set_window_dilation(1 + p.interior_padding());
    w.set_size(rhs->operand(0)->shape().dimensions(
        dnums.kernel_spatial_dimensions(dim)));
  }

  auto new_conv = convolution->CloneWithNewOperands(
      convolution->shape(), {lhs, rhs->mutable_operand(0)});
  new_conv->set_window(new_window);
  TF_RETURN_IF_ERROR(
      ReplaceWithNewInstruction(convolution, std::move(new_conv)));
  return true;
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::SwapConvOperands(
    HloInstruction* convolution) {
  if (!options_.enable_conv_operand_swap() || options_.is_layout_sensitive()) {
    return false;
  }
  if (convolution->feature_group_count() > 1 ||
      convolution->batch_group_count() > 1) {
    return false;
  }

  const auto& dnums = convolution->convolution_dimension_numbers();
  const auto& window_dims = convolution->window().dimensions();
  Window swapped_window;

  HloInstruction *input = convolution->mutable_operand(0),
                 *kernel = convolution->mutable_operand(1);
  int64_t kernel_product = 1;
  int64_t swapped_kernel_product = 1;
  DimensionVector reverse_dimensions;
  for (int64_t spatial_dim = 0;
       spatial_dim < dnums.input_spatial_dimensions_size(); ++spatial_dim) {
    const int64_t kernel_size = window_dims[spatial_dim].size();
    const bool can_be_group_or_contraction =
        !window_dims[spatial_dim].window_reversal() &&
        window_dims[spatial_dim].padding_low() == 0 &&
        window_dims[spatial_dim].padding_high() == 0 &&
        window_dims[spatial_dim].window_dilation() == 1;
    const bool is_group_dim =
        can_be_group_or_contraction &&
        window_dims[spatial_dim].base_dilation() == kernel_size &&
        window_dims[spatial_dim].stride() == kernel_size - 1;
    const int64_t input_size =
        input->shape().dimensions(dnums.input_spatial_dimensions(spatial_dim));
    const bool is_pure_contraction_dim =
        kernel_size == input_size && can_be_group_or_contraction &&
        window_dims[spatial_dim].base_dilation() == 1 &&
        window_dims[spatial_dim].stride() == 1;
    if (is_group_dim || is_pure_contraction_dim) {
      *(swapped_window.add_dimensions()) = window_dims[spatial_dim];
      continue;
    }

    const int64_t dilated_kernel_size =
        1 + (kernel_size - 1) * window_dims[spatial_dim].window_dilation();
    const int64_t dilated_input_size =
        1 + (input_size - 1) * window_dims[spatial_dim].base_dilation();

    // Don't decide to swap if the input size is one, since many convolution
    // implementations can easily hand that special case efficiently.
    kernel_product *= kernel_size;
    swapped_kernel_product *=
        input_size == 1 && window_dims[spatial_dim].stride() == 1 &&
                window_dims[spatial_dim].window_dilation() == 1 &&
                window_dims[spatial_dim].padding_high() == kernel_size - 1 &&
                window_dims[spatial_dim].padding_low() == kernel_size - 1
            ? kernel_size
            : input_size;

    auto new_dim = swapped_window.add_dimensions();
    new_dim->set_size(input_size);
    // If the kernel is not reversed, the activations must be manually reversed.
    if (!window_dims[spatial_dim].window_reversal()) {
      reverse_dimensions.push_back(
          dnums.kernel_spatial_dimensions(spatial_dim));
    }
    // The input is not originally reversed so it must be reversed to move the
    // kernel.
    new_dim->set_window_reversal(true);
    // Base dilation and window dilation switch places.
    new_dim->set_base_dilation(window_dims[spatial_dim].window_dilation());
    new_dim->set_window_dilation(window_dims[spatial_dim].base_dilation());
    new_dim->set_stride(window_dims[spatial_dim].stride());
    new_dim->set_padding_low(dilated_input_size +
                             window_dims[spatial_dim].padding_low() -
                             dilated_kernel_size);
    new_dim->set_padding_high(dilated_input_size +
                              window_dims[spatial_dim].padding_high() -
                              dilated_kernel_size);
  }

  // Don't transform if a naive convolution implementation would not have fewer
  // flops.
  if (kernel_product <= swapped_kernel_product) {
    return false;
  }
  ConvolutionDimensionNumbers swapped_dnums;
  *swapped_dnums.mutable_output_spatial_dimensions() =
      dnums.output_spatial_dimensions();
  // Swap batch and output feature of the output.
  swapped_dnums.set_output_batch_dimension(dnums.output_feature_dimension());
  swapped_dnums.set_output_feature_dimension(dnums.output_batch_dimension());

  // Swap input dnums with kernel dnums
  *swapped_dnums.mutable_input_spatial_dimensions() =
      dnums.kernel_spatial_dimensions();
  swapped_dnums.set_input_batch_dimension(
      dnums.kernel_output_feature_dimension());
  swapped_dnums.set_input_feature_dimension(
      dnums.kernel_input_feature_dimension());

  // Swap kernel dnums with input dnums
  *swapped_dnums.mutable_kernel_spatial_dimensions() =
      dnums.input_spatial_dimensions();
  swapped_dnums.set_kernel_output_feature_dimension(
      dnums.input_batch_dimension());
  swapped_dnums.set_kernel_input_feature_dimension(
      dnums.input_feature_dimension());

  PrecisionConfig precision_config;
  precision_config.add_operand_precision(
      convolution->precision_config().operand_precision(1));
  precision_config.add_operand_precision(
      convolution->precision_config().operand_precision(0));
  if (!reverse_dimensions.empty()) {
    TF_ASSIGN_OR_RETURN(kernel, MakeReverseHlo(kernel, reverse_dimensions));
  }
  TF_ASSIGN_OR_RETURN(
      HloInstruction * new_convolution,
      MakeConvolveHlo(
          kernel, input, /*feature_group_count=*/1,
          /*batch_group_count=*/1, swapped_window, swapped_dnums,
          precision_config,
          /*preferred_element_type=*/convolution->shape().element_type()));

  // If we're running on GPU we need to check that we can actually lower the
  // conv with the given reverse_dims (either none, or rank 2 and all)
  if (!options_.ConvIsLowerable(new_convolution)) {
    TF_RETURN_IF_ERROR(kernel->parent()->RemoveInstruction(new_convolution));
    return false;
  }

  convolution->SetupDerivedInstruction(new_convolution);
  TF_RETURN_IF_ERROR(ReplaceInstruction(convolution, new_convolution));

  return true;
}

absl::StatusOr<bool>
AlgebraicSimplifierVisitor::PromoteConvolutionToF32IfNotOnednnCompatible(
    HloInstruction** convolution) {
  bool can_rewrite = true;
  auto from_dtype = (*convolution)->shape().element_type();
  if (!options_.executing_on_cpu() || from_dtype != PrimitiveType::BF16) {
    return false;
  }
  if ((*convolution)->batch_group_count() != 1 ||
      (*convolution)->operand(1)->opcode() == HloOpcode::kReverse) {
    can_rewrite = false;
  }
  const Shape& inp_shape = (*convolution)->operand(0)->shape();
  const Shape& ker_shape = (*convolution)->operand(1)->shape();
  const Shape& out_shape = (*convolution)->shape();
  if (ShapeUtil::IsZeroElementArray(inp_shape) ||
      ShapeUtil::IsZeroElementArray(ker_shape) ||
      ShapeUtil::IsZeroElementArray(out_shape)) {
    can_rewrite = false;
  }

  auto dims = (*convolution)->window().dimensions().size();
  if (dims >= 4 || dims <= 0) can_rewrite = false;

  if (inp_shape.dimensions_size() != ker_shape.dimensions_size() ||
      inp_shape.dimensions_size() != out_shape.dimensions_size()) {
    can_rewrite = false;
  }

  const auto& window_dims = (*convolution)->window().dimensions();
  for (auto it = window_dims.begin(); it != window_dims.end(); ++it) {
    if (it->padding_low() < 0 || it->padding_high() < 0 || it->stride() < 0 ||
        it->base_dilation() != 1 || it->window_reversal()) {
      can_rewrite = false;
    }
  }

  if (can_rewrite) {
    return true;
  }

  // To ensure the correctness of the generated LLVM IR, we cast
  // the convolutions that are not rewritable to onednn custom calls to higher
  // precision. This does not compromise performance as lower floating point
  // precision convolutions are converted to higher precision in the regular
  // optimization pipeline.
  auto to_dtype = PrimitiveType::F32;
  std::vector<HloInstruction*> new_operands;
  auto from_dtype_operands = (*convolution)->operands();

  std::for_each(
      from_dtype_operands.begin(), from_dtype_operands.end(),
      [&new_operands, &to_dtype](HloInstruction* instr) {
        new_operands.push_back(
            instr->AddInstruction(HloInstruction::CreateConvert(
                ShapeUtil::ChangeElementType(instr->shape(), to_dtype),
                instr)));
      });

  HloInstruction* to_conv =
      (*convolution)
          ->AddInstruction(
              (*convolution)
                  ->CloneWithNewOperands(ShapeUtil::ChangeElementType(
                                             (*convolution)->shape(), to_dtype),
                                         new_operands));

  HloInstruction* from_conv =
      to_conv->AddInstruction(HloInstruction::CreateConvert(
          ShapeUtil::ChangeElementType(to_conv->shape(), from_dtype), to_conv));

  TF_RETURN_IF_ERROR(ReplaceInstruction(*convolution, from_conv));
  *convolution = to_conv;
  return false;
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::SimplifyConvToDot(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const auto& window = convolution->window();
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (!options_.enable_conv_simplification()) {
    return false;
  }

  // layout-insensitive mode, for fear of adding nontrivial reshapes.
  if (!options_.is_layout_sensitive()) {
    return false;
  }

  const Shape& input_shape = lhs->shape();
  const Shape& filter_shape = rhs->shape();
  const Shape& convolution_shape = convolution->shape();
  TF_RET_CHECK(LayoutUtil::HasLayout(input_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(filter_shape));
  TF_RET_CHECK(LayoutUtil::HasLayout(convolution_shape));

  // Require the spatial dimensions in the kernel to have a bound of one.
  for (int64_t i = 0; i < dnums.kernel_spatial_dimensions_size(); ++i) {
    if (filter_shape.dimensions(dnums.kernel_spatial_dimensions(i)) != 1) {
      return false;
    }
  }

  // Stride ignores part of the output, which matrix multiplication does not do,
  // so require no stride. Padding and base (lhs) dilation both implicitly
  // extend the data, which matrix multiplication also does not do, so require
  // no padding and no base (lhs) dilation. Window (rhs) dilation has no effect
  // for a 1x1 window, so window dilation is no problem.
  if (window_util::HasStride(window) || window_util::HasPadding(window) ||
      window_util::HasBaseDilation(window)) {
    return false;
  }

  // Also, the shapes must align for a rowmajor matmul:
  // - the input and output have the same layout.
  // - for input/output, the channel dimension must be the most minor. Other
  //   spatial dims can be in any order.
  // - for filters, the input channel dimension must be more major than the
  //   output channel dimension. The width+height don't matter because
  //   they are 1.
  //
  // These constraints are harsh. If the channel dimension is the most major
  // and/or the layout of input/output feature dimensions are reversed, we can
  // still convert Conv into more efficient Matmul with operand transposition
  // (such as the transposition flags in cuBLAS SGEMM).
  if (!LayoutUtil::Equal(input_shape.layout(), convolution_shape.layout()) ||
      LayoutUtil::Minor(input_shape.layout(), 0) !=
          dnums.input_feature_dimension() ||
      LayoutUtil::Minor(convolution_shape.layout(), 0) !=
          dnums.output_feature_dimension() ||
      // The input feature dimension should come later in the minor-to-major
      // order.
      (PositionInContainer(LayoutUtil::MinorToMajor(filter_shape),
                           dnums.kernel_input_feature_dimension()) <
       PositionInContainer(LayoutUtil::MinorToMajor(filter_shape),
                           dnums.kernel_output_feature_dimension()))) {
    return false;
  }

  if (convolution->feature_group_count() != 1 ||
      convolution->batch_group_count() != 1) {
    return false;
  }
  auto add_bitcast = [&](Shape shape, HloInstruction* operand) {
    std::vector<int64_t> dims(operand->shape().dimensions_size());
    std::iota(dims.begin(), dims.end(), 0);
    return operand->AddInstruction(
        HloInstruction::CreateBitcast(shape, operand));
  };

  // Replace it with a dot, with bitcasts around it to get the right shape.
  const int64_t input_channels =
      input_shape.dimensions(dnums.input_feature_dimension());
  const int64_t output_channels =
      filter_shape.dimensions(dnums.kernel_output_feature_dimension());

  // Computes the product of the non-feature dimensions.
  int64_t conv_width = 1;
  for (int i = 0; i < input_shape.dimensions_size(); ++i) {
    if (i != dnums.input_feature_dimension()) {
      conv_width *= input_shape.dimensions(i);
    }
  }

  // We already checked feature_dimension is most minor, so data in input_shape
  // and row-major {conv_width,input_channels} are bitwise identical.
  Shape new_input_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      input_shape.element_type(), {conv_width, input_channels});
  simplifier_->UpdateLayout(&new_input_shape);
  // We already checked input_feature_dimension is more major than
  // output_feature_dimension, so data in filter_shape and row-major
  // {input_channels,output_channels} are bitwise identical.
  Shape new_filter_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      filter_shape.element_type(), {input_channels, output_channels});
  simplifier_->UpdateLayout(&new_filter_shape);
  Shape dot_output_shape = ShapeUtil::MakeShapeWithDescendingLayout(
      convolution_shape.element_type(), {conv_width, output_channels});
  simplifier_->UpdateLayout(&dot_output_shape);

  auto new_lhs = add_bitcast(new_input_shape, lhs);
  auto new_rhs = add_bitcast(new_filter_shape, rhs);
  DotDimensionNumbers dot_dimension_numbers;
  dot_dimension_numbers.add_lhs_contracting_dimensions(1);
  dot_dimension_numbers.add_rhs_contracting_dimensions(0);
  auto dot = convolution->AddInstruction(HloInstruction::CreateDot(
      dot_output_shape, new_lhs, new_rhs, dot_dimension_numbers,
      convolution->precision_config()));

  TF_RETURN_IF_ERROR(
      ReplaceInstruction(convolution, add_bitcast(convolution_shape, dot)));
  return true;
}

absl::StatusOr<bool> AlgebraicSimplifierVisitor::SimplifyConvToMultiply(
    HloInstruction* convolution) {
  if (options_.is_layout_sensitive()) {
    return false;
  }

  auto* input = convolution->mutable_operand(0);
  auto* kernel = convolution->mutable_operand(1);
  const auto& window = convolution->window();
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  const Shape& input_shape = input->shape();
  const Shape& kernel_shape = kernel->shape();
  const Shape& convolution_shape = convolution->shape();

  // Require the spatial dimensions to be either contracted or trivial.
  for (int64_t i = 0; i < dnums.output_spatial_dimensions_size(); ++i) {
    if (kernel_shape.dimensions(dnums.kernel_spatial_dimensions(i)) != 1 &&
        convolution_shape.dimensions(dnums.output_spatial_dimensions(i)) != 1) {
      return false;
    }
  }

  // Stride ignores part of the output, which matrix multiplication does not do,
  // so require no stride. Padding and dilation both implicitly extend the data,
  // which matrix multiplication also does not do, so require no padding and no
  // dilation.
  if (window_util::HasStride(window) || window_util::HasPadding(window) ||
      window_util::HasDilation(window)) {
    return false;
  }

  // Verify feature dimensions.
  if (kernel_shape.dimensions(dnums.kernel_input_feature_dimension()) != 1 ||
      input_shape.dimensions(dnums.input_feature_dimension()) !=
          convolution->feature_group_count() ||
      convolution_shape.dimensions(dnums.output_feature_dimension()) !=
          convolution->feature_group_count()) {
    return false;
  }

  // Calculate permutations for the operand dimensions.
  DimensionVector input_permutation(input_shape.dimensions_size());
  DimensionVector kernel_permutation(kernel_shape.dimensions_size());

  input_permutation[dnums.output_batch_dimension()] =
      dnums.input_batch_dimension();
  input_permutation[dnums.output_feature_dimension()] =
      dnums.input_feature_dimension();

  kernel_permutation[dnums.output_batch_dimension()] =
      dnums.kernel_input_feature_dimension();
  kernel_permutation[dnums.output_feature_dimension()] =
      dnums.kernel_output_feature_dimension();

  // Set reduction dimensions for spatial dimensions where the kernel size is
  // not equal to one.
  DimensionVector reduction_dimensions;
  for (int64_t i = 0; i < dnums.output_spatial_dimensions_size(); ++i) {
    int64_t dim = dnums.output_spatial_dimensions(i);
    input_permutation[dim] = dnums.input_spatial_dimensions(i);
    kernel_permutation[dim] = dnums.kernel_spatial_dimensions(i);
    if (kernel_shape.dimensions(dnums.kernel_spatial_dimensions(i)) != 1) {
      reduction_dimensions.push_back(dim);
    }
  }

  // Update shapes of the operands, if necessary.
  if (!absl::c_is_sorted(input_permutation)) {
    TF_ASSIGN_OR_RETURN(input, MakeTransposeHlo(input, input_permutation));
  }
  if (!ShapeUtil::SameElementType(input_shape, convolution_shape)) {
    input = MakeConvertToHlo(input, convolution_shape.element_type());
  }

  if (!absl::c_is_sorted(kernel_permutation)) {
    TF_ASSIGN_OR_RETURN(kernel, MakeTransposeHlo(kernel, kernel_permutation));
  }
  if (!ShapeUtil::SameElementType(kernel_shape, convolution_shape)) {
    kernel = MakeConvertToHlo(kernel, convolution_shape.element_type());
  }

  // Replace convolution with reduce(input * broadcast(kernel))
  kernel = convolution->parent()->AddInstruction(
      HloInstruction::CreateBroadcastSequence(
          input->shape(), kernel, [&](std::unique_ptr<HloInstruction> added) {
            return convolution->parent()->AddInstruction(std::move(added));
          }));
  TF_ASSIGN_OR_RETURN(HloInstruction * result,
                      MakeBinaryHlo(HloOpcode::kMultiply, input, kernel));
  if (!reduction_dimensions.empty()) {
    TF_ASSIGN_OR_RETURN(
        HloInstruction * sum,
        MakeReduceHlo(
            result,
            MakeConvertToHlo(MakeR0ConstantHlo(convolution->parent(), 0),
                             convolution_shape.element_type()),
            reduction_dimensions, HloOpcode::kAdd));
    TF_ASSIGN_OR_RETURN(result, MakeReshapeHlo(convolution_shape, sum));
  }

  TF_RETURN_IF_ERROR(ReplaceInstruction(convolution, result));
  return true;
}

absl::Status AlgebraicSimplifierVisitor::HandleConvolution(
    HloInstruction* convolution) {
  if (options_.enable_scalar_multiply_reduction()) {
    TF_RETURN_IF_ERROR(ScalarMultiplyReduction(convolution));
  }

  // Zero-sized input or filter.
  if (ShapeUtil::IsZeroElementArray(convolution->operand(0)->shape()) ||
      ShapeUtil::IsZeroElementArray(convolution->operand(1)->shape())) {
    return ReplaceInstruction(convolution, MakeScalarLike(convolution, 0));
  }

  // Try to merge padding/dilation of the input with the convolution's window.
  TF_ASSIGN_OR_RETURN(bool folded_input_pad, FoldConvInputPad(convolution));
  if (folded_input_pad) {
    return absl::OkStatus();
  }

  // Try to merge dilation of the filter with the convolution's window.
  TF_ASSIGN_OR_RETURN(bool folded_filter_pad, FoldConvFilterPad(convolution));
  if (folded_filter_pad) {
    return absl::OkStatus();
  }

  // Try to swap convolution operands.
  TF_ASSIGN_OR_RETURN(bool swapped, SwapConvOperands(convolution));
  if (swapped) {
    return absl::OkStatus();
  }

  if (options_.enable_onednn_support()) {
    // Convert the data type back to F32 if we can't rewrite BF16 convolution to
    // oneDNN custom call.
    TF_ASSIGN_OR_RETURN(
        bool can_rewrite_bf16_conv_to_onednn,
        PromoteConvolutionToF32IfNotOnednnCompatible(&convolution));
    if (can_rewrite_bf16_conv_to_onednn) {
      return absl::OkStatus();
    }
  }

  // Try to replace the convolution with a kDot or a kMultiply instruction.
  TF_ASSIGN_OR_RETURN(bool replaced_with_dot, SimplifyConvToDot(convolution));
  if (replaced_with_dot) {
    return absl::OkStatus();
  }
  TF_ASSIGN_OR_RETURN(bool replaced_with_multiply,
                      SimplifyConvToMultiply(convolution));
  if (replaced_with_multiply) {
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::Status AlgebraicSimplifierVisitor::HandleMap(HloInstruction* map) {
  auto* map_computation = map->to_apply();
  auto* map_root = map_computation->root_instruction();
  if (map_root->opcode() == HloOpcode::kParameter) {
    ReplaceInstructionIfCompatible(
        map, map->mutable_operand(map_root->parameter_number()));
    return absl::OkStatus();
  }
  if (map_root->opcode() == HloOpcode::kConstant) {
    if (!ShapeUtil::IsScalar(map_root->shape())) {
      return absl::OkStatus();
    }
    auto clone = map_root->CloneWithNewOperands(map_root->shape(), {});
    if (ShapeUtil::IsScalar(map->shape())) {
      return ReplaceWithNewInstruction(map, std::move(clone));
    }
    return ReplaceWithNewInstruction(
        map, HloInstruction::CreateBroadcast(
                 map->shape(), map->AddInstruction(std::move(clone)), {}));
  }
  // Inline the map if the map computation only contains an elementwise
  // operation that can accept arbitrary shapes.
  if (map_root->opcode() == HloOpcode::kFusion || !map_root->IsElementwise()) {
    return absl::OkStatus();
  }
  std::vector<HloInstruction*> new_operands;
  for (auto* root_operand : map_root->operands()) {
    if (root_operand->opcode() != HloOpcode::kParameter) {
      return absl::OkStatus();
    }
    new_operands.push_back(
        map->mutable_operand(root_operand->parameter_number()));
  }
  auto clone = map_root->CloneWithNewOperands(map->shape(), new_operands);
  return ReplaceWithNewInstruction(map, std::move(clone));
}

absl::StatusOr<bool> AlgebraicSimplifier::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  AlgebraicSimplifierVisitor visitor(options_, this);
  for (auto* comp : module->MakeNonfusionComputations(execution_threads)) {
    if (visitor.Run(comp, options_, this)) {
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
