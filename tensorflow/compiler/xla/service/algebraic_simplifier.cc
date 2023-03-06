/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/algebraic_simplifier.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/hlo/evaluator/hlo_evaluator.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_comparison.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/overflow_util.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"

namespace xla {

namespace {

namespace m = match;

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

std::optional<double> GetConstantValue(const HloInstruction* inst) {
  if (!ShapeUtil::IsEffectiveScalar(inst->shape())) {
    return std::nullopt;
  }
  switch (inst->shape().element_type()) {
    case F16:
      return static_cast<float>(inst->literal().GetFirstElement<half>());
    case BF16:
      return static_cast<float>(inst->literal().GetFirstElement<bfloat16>());
    case F32:
      return inst->literal().GetFirstElement<float>();
    case F64:
      return inst->literal().GetFirstElement<double>();
    default:
      return std::nullopt;
  }
}

static bool IsScalarConstant(const HloInstruction* hlo,
                             const LiteralSlice& literal) {
  return hlo->opcode() == HloOpcode::kConstant &&
         ShapeUtil::IsEffectiveScalar(hlo->shape()) &&
         literal_comparison::Equal(hlo->literal(), literal).ok();
}

static bool IsScalarConstantZero(const HloInstruction* hlo) {
  return IsScalarConstant(hlo, LiteralUtil::Zero(hlo->shape().element_type()));
}

static bool IsScalarConstantNegInf(const HloInstruction* hlo) {
  return !primitive_util::IsComplexType(hlo->shape().element_type()) &&
         IsScalarConstant(hlo,
                          LiteralUtil::MinValue(hlo->shape().element_type()));
}

static bool IsScalarConstantInf(const HloInstruction* hlo) {
  return !primitive_util::IsComplexType(hlo->shape().element_type()) &&
         IsScalarConstant(hlo,
                          LiteralUtil::MaxValue(hlo->shape().element_type()));
}

bool IsNonNegative(const HloInstruction* hlo,
                   const AlgebraicSimplifierOptions& options) {
  // Utility only handles real types.
  if (IsAnyOperandComplex(hlo)) {
    return false;
  }
  switch (hlo->opcode()) {
    case HloOpcode::kMultiply: {
      return hlo->operand(0) == hlo->operand(1);
    }
    case HloOpcode::kAbs: {
      return true;
    }
    case HloOpcode::kBroadcast: {
      return IsNonNegative(hlo->operand(0), options);
    }
    case HloOpcode::kConstant: {
      if (std::optional<double> value = GetConstantValue(hlo)) {
        return *value >= 0.0;
      }
      return false;
    }
    case HloOpcode::kMaximum: {
      return IsNonNegative(hlo->operand(0), options) ||
             IsNonNegative(hlo->operand(1), options);
    }
    case HloOpcode::kSelect: {
      return IsNonNegative(hlo->operand(1), options) &&
             IsNonNegative(hlo->operand(2), options);
    }
    default:
      return IsPositive(hlo, options);
  }
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
  auto val = [&]() -> std::optional<double> {
    switch (c->shape().element_type()) {
      case BF16:
        return static_cast<double>(c->literal().GetFirstElement<bfloat16>());
      case F16:
        return static_cast<double>(c->literal().GetFirstElement<Eigen::half>());
      case F32:
        return c->literal().GetFirstElement<float>();
      case F64:
        return c->literal().GetFirstElement<double>();
      default:
        // Cowardly refuse to consider complex types.
        return std::nullopt;
    }
  }();
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

bool IsUnstridedSlice(const HloInstruction* hlo) {
  return absl::c_all_of(hlo->slice_strides(),
                        [](int64_t stride) { return stride == 1; });
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

}  // namespace

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
  switch (target->shape().element_type()) {
    case BF16:
      return HloInstruction::CreateConstant(LiteralUtil::ConvertF32ToBF16(
          LiteralUtil::CreateR0<float>(multiplier)));
      break;
    case F32:
      return HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<float>(multiplier));
      break;
    default:
      LOG(FATAL) << "Unsupported data type: " << target->shape().element_type();
  }
}

}  // namespace

Status AlgebraicSimplifierVisitor::ScalarMultiplyReduction(
    HloInstruction* dot) {
  // We only process bfloat16 and float32 for now.
  if (dot->shape().element_type() != BF16 &&
      dot->shape().element_type() != F32) {
    return OkStatus();
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
    return OkStatus();
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

Status AlgebraicSimplifierVisitor::HandleAbs(HloInstruction* abs) {
  HloInstruction* abs_operand = abs->mutable_operand(0);
  VLOG(10) << "trying transform [Abs(A) => A] " << abs->ToString()
           << " Abs operand is: " << abs_operand->ToString();
  if (IsNonNegative(abs->operand(0), options_)) {
    return ReplaceInstruction(abs, abs_operand);
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleAdd(HloInstruction* add) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(add, m::Add(m::Op(&lhs), m::Op(&rhs))));

  // A + 0 => A
  VLOG(10) << "trying transform [A + 0 => A]: " << add->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(add, lhs)) {
    return OkStatus();
  }
  // 0 + A => A
  VLOG(10) << "trying transform [0 + A => A]: " << add->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfCompatible(add, rhs)) {
    return OkStatus();
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
    return OkStatus();
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
    if (lhs_scatter_index->shape().rank() !=
        rhs_scatter_index->shape().rank()) {
      return OkStatus();
    }

    int64_t first_index_dim = lhs_scatter_index->shape().rank();
    int64_t first_update_dim = lhs_scatter_update->shape().rank();
    // Find a dimension where it is possible to concatenate the indices and
    // updates. This is the first and only non-equal dimension or the first
    // equally sized dimension.
    for (int64_t d = lhs_scatter_index->shape().rank() - 1,
                 update_dim = lhs_scatter_update->shape().rank() - 1;
         d >= 0; --d) {
      if (d == lhs_dnums.index_vector_dim()) {
        continue;
      }
      while (
          absl::c_linear_search(lhs_dnums.update_window_dims(), update_dim)) {
        --update_dim;
      }
      if (lhs_scatter_index->shape().dimensions(d) ==
          rhs_scatter_index->shape().dimensions(d)) {
        first_index_dim = d;
        first_update_dim = update_dim--;
        continue;
      }
      // More than one dimension of unequal size was found, bail out.
      if (index_concat_dimension) {
        return OkStatus();
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
    if (*index_concat_dimension == lhs_scatter_index->shape().rank()) {
      return OkStatus();
    }
    const bool update_concat_is_cheap =
        ShapeUtil::ElementsIn(rhs_scatter_update->shape()) +
            ShapeUtil::ElementsIn(lhs_scatter_update->shape()) <
        ShapeUtil::ElementsIn(lhs->shape());
    if (!update_concat_is_cheap) {
      return OkStatus();
    }
    const bool same_dimension_numbers =
        lhs_dnums.index_vector_dim() == rhs_dnums.index_vector_dim() &&
        absl::c_equal(lhs_dnums.scatter_dims_to_operand_dims(),
                      rhs_dnums.scatter_dims_to_operand_dims()) &&
        absl::c_equal(lhs_dnums.inserted_window_dims(),
                      rhs_dnums.inserted_window_dims()) &&
        absl::c_equal(lhs_dnums.update_window_dims(),
                      rhs_dnums.update_window_dims());
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
  return OkStatus();
}

StatusOr<bool> AlgebraicSimplifierVisitor::TrySimplifyTautologicalCompare(
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

Status AlgebraicSimplifierVisitor::HandleAnd(HloInstruction* logical_and) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_and, m::And(m::Op(&lhs), m::Op(&rhs))));
  // Simplify logical and
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A && True => A
    VLOG(10) << "trying transform [A && True => A]: "
             << logical_and->ToString();
    if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(logical_and, lhs)) {
      return OkStatus();
    }
    // True && A => A
    VLOG(10) << "trying transform [True && A => A]: "
             << logical_and->ToString();
    if (IsAll(lhs, 1) && ReplaceInstructionIfCompatible(logical_and, rhs)) {
      return OkStatus();
    }
  }

  // A && False => False or A & 0 => 0
  VLOG(10) << "trying transform [A && False => False]: "
           << logical_and->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(logical_and, rhs)) {
    return OkStatus();
  }

  // False && A => False or A & 0 => 0
  VLOG(10) << "trying transform [False && A => False]: "
           << logical_and->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfCompatible(logical_and, lhs)) {
    return OkStatus();
  }

  // Simplify tautological conjunctions.
  TF_ASSIGN_OR_RETURN(bool found_tautological_compare,
                      TrySimplifyTautologicalCompare(logical_and));
  if (found_tautological_compare) {
    return OkStatus();
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleBitcast(HloInstruction* bitcast) {
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
  // All bitcasts can be eliminated (assuming layout constraints are satisfied).
  ReplaceInstructionIfCompatible(bitcast, bitcast->mutable_operand(0));
  return OkStatus();
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
      if (result_shape.dimensions(bitcast_dim) == 1) {
        // Postpone all degenerated dimensions (those with size 1) to the end.
        continue;
      }
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
  for (int i = 0; i < result_shape.rank(); ++i) {
    if (result_shape.dimensions(i) == 1) {
      bitcast_pos++;
      // Since there is a possiblity of over-incrementing bitcast_pos
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
  CHECK_EQ(bitcast_pos + 1, result_shape.rank());
  return new_shape;
}

std::vector<std::vector<int64_t>>
AlgebraicSimplifierVisitor::InvertBitcastDimMap(
    const Shape& original_shape, const Shape& bitcast_shape,
    const std::vector<std::vector<int64_t>>& original_map) {
  std::vector<std::vector<int64_t>> result_map(bitcast_shape.dimensions_size());
  // Invert the operand map into result map.
  for (auto i = 0; i < original_shape.rank(); ++i) {
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

Status AlgebraicSimplifierVisitor::HandleBitcastConvert(
    HloInstruction* bitcast) {
  TF_ASSIGN_OR_RETURN(bool replaced,
                      TrySimplifyTautologicalBitcastConvert(bitcast));
  if (replaced) {
    return OkStatus();
  }
  // Eliminate bitcast converts between same shape.
  ReplaceInstructionIfCompatible(bitcast, bitcast->mutable_operand(0));
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleCopy(HloInstruction* copy) {
  if (SwapCopyBitcastCopy(copy)) {
    return OkStatus();
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
    return OkStatus();
  }

  if (HloInstruction* bitcast_operand =
          BitcastingOperandOfReshapeOrCopyChain(copy, options_)) {
    ReplaceWithBitcast(copy, bitcast_operand);
    return OkStatus();
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleConcatenate(
    HloInstruction* concatenate) {
  absl::Span<HloInstruction* const> operands(concatenate->operands());
  if (operands.size() == 1) {
    // Unary concatenates are useless.
    ReplaceInstructionIfCompatible(concatenate, operands[0]);
    return OkStatus();
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
    return OkStatus();
  }

  if (options_.is_layout_sensitive()) {
    return OkStatus();
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
        !IsUnstridedSlice(operands[i])) {
      new_operands.push_back(operands[i]);
      ++i;
      continue;
    }
    int64_t slice_end = operands[i]->slice_limits(concatenate_dimension);
    HloInstruction* slice_operand = operands[i]->mutable_operand(0);
    int64_t j = i + 1;
    while (j < operands.size()) {
      if (operands[j]->opcode() != HloOpcode::kSlice ||
          !IsUnstridedSlice(operands[j]) ||
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
    return OkStatus();
  }

  if (operands.size() == 2) {
    // A binary concat with a broadcasted scalar as an operand can be converted
    // into a pad which is simpler to fold into other operations.
    bool is_effective_low_pad = Match(
        operands[0], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    bool is_effective_high_pad = Match(
        operands[1], m::Broadcast(m::Op().WithShape(m::Shape().IsScalar())));
    if (!is_effective_low_pad && !is_effective_high_pad) {
      return OkStatus();
    }
    PaddingConfig padding_config;
    for (int64_t dim = 0; dim < operands[0]->shape().rank(); ++dim) {
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
    for (int64_t i = 0; i < new_shape.rank(); ++i) {
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
  return OkStatus();
}

StatusOr<bool>
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

Status AlgebraicSimplifierVisitor::HandleConstant(HloInstruction* constant) {
  // Tuple constants aren't directly supported by any backend. Expand them into
  // explicit Tuple instructions.
  if (constant->shape().IsTuple()) {
    return ReplaceInstruction(
        constant,
        BuildTupleConstant(computation_, constant->literal(), simplifier_));
  }

  if (constant->shape().element_type() == TOKEN) {
    return OkStatus();
  }

  // If a literal is all the same element replace it with a scalar broadcast.
  if (ShapeUtil::ElementsIn(constant->shape()) > 1 &&
      constant->literal().IsAllFirst()) {
    Literal unique_scalar(
        LiteralUtil::GetFirstScalarLiteral(constant->literal()));
    HloInstruction* scalar = constant->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(std::move(unique_scalar)));
    return ReplaceWithNewInstruction(
        constant,
        HloInstruction::CreateBroadcast(constant->shape(), scalar, {}));
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleSubtract(HloInstruction* sub) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(sub, m::Subtract(m::Op(&lhs), m::Op(&rhs))));
  // A - 0 => A
  VLOG(10) << "trying transform [A - 0 => A]: " << sub->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(sub, lhs)) {
    return OkStatus();
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

  return OkStatus();
}
namespace {
template <typename T>
Status InvertConstant(const HloInstruction& constant, Literal* result) {
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
    int64_t b_value = c->literal().GetFirstElement<T>();
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
    uint64_t b_value = c->literal().GetFirstElement<T>();
    if (absl::has_single_bit(b_value)) {
      return HloInstruction::CreateBinary(
          divide->shape(), HloOpcode::kShiftRightLogical, a,
          MakeScalarLike(a, Log2Floor(b_value)));
    }
  }

  return nullptr;
}
}  // namespace

Status AlgebraicSimplifierVisitor::HandleDivide(HloInstruction* divide) {
  HloInstruction *a, *b, *c, *d;
  CHECK(Match(divide, m::Divide(m::Op(&a), m::Op(&b))));
  // A/1 => A
  VLOG(10) << "trying transform [A/1 => A]: " << divide->ToString();
  if (IsAll(b, 1) && ReplaceInstructionIfCompatible(divide, a)) {
    return OkStatus();
  }

  // A / B => A >> log2(B) if B is a power of 2.
  switch (divide->shape().element_type()) {
    case S8:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int8_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S16:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int16_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S32:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int32_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case S64:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<int64_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U8:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint8_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U16:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint16_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U32:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint32_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    case U64:
      if (std::unique_ptr<HloInstruction> shift =
              TryDivideToShift<uint64_t>(divide, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(divide, std::move(shift));
      }
      break;
    default:
      break;
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

  // A/sqrt(B) => A*rsqrt(X).
  if (Match(divide, m::Divide(m::Op(&a), m::Sqrt(m::Op(&b))))) {
    auto* rsqrt = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kRsqrt, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(rsqrt->shape(),
                                             HloOpcode::kMultiply, a, rsqrt));
  }

  // A/rsqrt(B) => A*sqrt(B).
  if (Match(divide, m::Divide(m::Op(&a), m::Rsqrt(m::Op(&b))))) {
    auto* sqrt = divide->mutable_operand(1)->AddInstruction(
        HloInstruction::CreateUnary(divide->shape(), HloOpcode::kSqrt, b));
    return ReplaceWithNewInstruction(
        divide, HloInstruction::CreateBinary(sqrt->shape(),
                                             HloOpcode::kMultiply, a, sqrt));
  }

  // Simplifying integral division would produce unexpected results.
  if (ShapeUtil::ElementIsIntegral(divide->shape())) {
    return OkStatus();
  }

  // A / Const => A * (1 / Const)
  //
  // (Backends can do this transformation, but generally only if the constant is
  // a scalar.)
  if (Match(divide, m::Divide(m::NonConstant(&a), m::Op(&b))) &&
      (Match(b, m::Constant(&c)) || Match(b, m::Broadcast(m::Constant(&c))))) {
    Shape result_shape = c->literal().shape();
    Literal new_literal(result_shape);
    switch (result_shape.element_type()) {
      case F16:
        TF_RETURN_IF_ERROR(InvertConstant<half>(*c, &new_literal));
        break;
      case F32:
        TF_RETURN_IF_ERROR(InvertConstant<float>(*c, &new_literal));
        break;
      case BF16:
        TF_RETURN_IF_ERROR(InvertConstant<bfloat16>(*c, &new_literal));
        break;
      case F64:
        TF_RETURN_IF_ERROR(InvertConstant<double>(*c, &new_literal));
        break;
      case C64:
        TF_RETURN_IF_ERROR(InvertConstant<complex64>(*c, &new_literal));
        break;
      case C128:
        TF_RETURN_IF_ERROR(InvertConstant<complex128>(*c, &new_literal));
        break;
      default:
        return OkStatus();
    }
    auto inverse = c->AddInstruction(
        simplifier_->CreateConstantWithLayoutUpdated(new_literal.Clone()));
    if (b != c) {
      inverse = b->AddInstruction(HloInstruction::CreateBroadcast(
          b->shape(), inverse, b->dimensions()));
    }
    TF_ASSIGN_OR_RETURN(auto new_divide,
                        MakeBinaryHlo(HloOpcode::kMultiply, a, inverse));
    return ReplaceInstruction(divide, new_divide);
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

  return OkStatus();
}

StatusOr<bool> AlgebraicSimplifierVisitor::RemoveDegenerateDimensionFromDot(
    HloInstruction* dot) {
  const Shape& lhs_shape = dot->operand(0)->shape();
  int64_t num_degenerate_lhs_dims = 0;
  std::vector<int64_t> lhs_dimension_map(lhs_shape.rank(), -1);
  for (int64_t i = 0; i < lhs_shape.rank(); ++i) {
    if (lhs_shape.dimensions(i) == 1) {
      ++num_degenerate_lhs_dims;
    } else {
      lhs_dimension_map[i] = i - num_degenerate_lhs_dims;
    }
  }

  const Shape& rhs_shape = dot->operand(1)->shape();
  int64_t num_degenerate_rhs_dims = 0;
  std::vector<int64_t> rhs_dimension_map(rhs_shape.rank(), -1);
  for (int64_t i = 0; i < rhs_shape.rank(); ++i) {
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
                 /*preferred_element_type=*/dot->shape().element_type()));
  dot->SetupDerivedInstruction(new_dot);
  if (ShapeUtil::Compatible(dot->shape(), new_dot->shape())) {
    TF_RETURN_IF_ERROR(ReplaceInstruction(dot, new_dot));
  } else {
    TF_RETURN_IF_ERROR(ReplaceWithNewInstruction(
        dot, HloInstruction::CreateReshape(dot->shape(), new_dot)));
  }
  return true;
}

StatusOr<bool> AlgebraicSimplifierVisitor::RemoveTransposesFromDotOperands(
    HloInstruction* dot) {
  const int64_t rank = dot->shape().rank();
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

StatusOr<HloInstruction*>
AlgebraicSimplifierVisitor::NormalizeDotOperandToBatchMajorAndContractingMinor(
    HloInstruction* dot_operand, absl::Span<const int64_t> batch_dimensions,
    absl::Span<const int64_t> contracting_dimensions) {
  std::vector<int64_t> transpose_dimensions(batch_dimensions.begin(),
                                            batch_dimensions.end());
  for (int64_t i = 0; i < dot_operand->shape().rank(); ++i) {
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

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfConcat(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
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

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfConcatHelper(
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

    // TODO(b/69062148): We can get rid of `swapped` once all backends support
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

StatusOr<HloInstruction*> AlgebraicSimplifierVisitor::OptimizeDotOfGather(
    HloInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1 ||
      dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
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
StatusOr<HloInstruction*>
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
      !Match(rhs, m::Constant(&constant))) {
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
  if ((unmodified_dims.size() != reshape->shape().rank() - 1) ||
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
  std::vector<int64_t> rhs_transpose_dims(rhs_reshape->shape().rank());
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

Status AlgebraicSimplifierVisitor::HandleDot(HloInstruction* dot) {
  CHECK(computation_ == dot->parent());
  const auto& dnums = dot->dot_dimension_numbers();

  HloInstruction *lhs, *rhs;
  CHECK(Match(dot, m::Dot(m::Op(&lhs), m::Op(&rhs))));
  if (options_.is_layout_sensitive()) {
    return OkStatus();
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

  const bool is_packed_nibble =
      absl::c_linear_search(dot->precision_config().operand_precision(),
                            PrecisionConfig::PACKED_NIBBLE);
  // If there are no contracting dimensions, a dot can be rewritten as
  // mul(broadcast(transpose(x)),broadcast(transpose(y)))
  if (!is_packed_nibble && options_.enable_dot_to_multiply_rewrite() &&
      dnums.lhs_contracting_dimensions_size() == 0) {
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
    if (dot->shape().rank() != lhs->shape().rank()) {
      std::vector<int64_t> lhs_broadcast_dims(lhs->shape().rank());
      absl::c_iota(lhs_broadcast_dims, 0);
      new_lhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
          dot->shape(), new_lhs, lhs_broadcast_dims));
    }
    if (dot->shape().rank() != rhs->shape().rank()) {
      std::vector<int64_t> rhs_broadcast_dims(
          dnums.lhs_batch_dimensions_size());
      absl::c_iota(rhs_broadcast_dims, 0);
      for (int64_t i = lhs->shape().rank(); i < dot->shape().rank(); ++i) {
        rhs_broadcast_dims.push_back(i);
      }
      new_rhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
          dot->shape(), new_rhs, rhs_broadcast_dims));
    }
    return ReplaceWithNewInstruction(
        dot, HloInstruction::CreateBinary(dot->shape(), HloOpcode::kMultiply,
                                          new_lhs, new_rhs));
  }

  // If the lhs or rhs have only batch and contracting dimensions, a dot can be
  // rewritten as reduce(mul(broadcast(transpose(x)),broadcast(transpose(y))))
  if (!is_packed_nibble && options_.enable_dot_strength_reduction() &&
      ((dnums.lhs_batch_dimensions_size() +
            dnums.lhs_contracting_dimensions_size() ==
        lhs->shape().rank()) ||
       (dnums.rhs_contracting_dimensions_size() +
            dnums.rhs_batch_dimensions_size() ==
        rhs->shape().rank())) &&
      ShouldStrengthReduceDotToReduce(dot)) {
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

    int64_t lhs_outer_dims =
        lhs->shape().rank() - (dnums.lhs_batch_dimensions_size() +
                               dnums.lhs_contracting_dimensions_size());
    int64_t rhs_outer_dims =
        rhs->shape().rank() - (dnums.rhs_batch_dimensions_size() +
                               dnums.rhs_contracting_dimensions_size());
    CHECK(lhs_outer_dims == 0 || rhs_outer_dims == 0);
    if (rhs_outer_dims > 0) {
      std::vector<int64_t> lhs_broadcast_dims(
          dnums.lhs_batch_dimensions_size());
      absl::c_iota(lhs_broadcast_dims, 0);
      lhs_broadcast_dims.resize(lhs->shape().rank());
      std::iota(lhs_broadcast_dims.begin() + dnums.lhs_batch_dimensions_size(),
                lhs_broadcast_dims.end(),
                dnums.lhs_batch_dimensions_size() + rhs_outer_dims);
      new_lhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
          new_rhs->shape(), new_lhs, lhs_broadcast_dims));
    } else if (lhs_outer_dims > 0) {
      std::vector<int64_t> rhs_broadcast_dims(
          dnums.rhs_batch_dimensions_size());
      absl::c_iota(rhs_broadcast_dims, 0);
      rhs_broadcast_dims.resize(rhs->shape().rank());
      std::iota(rhs_broadcast_dims.begin() + dnums.rhs_batch_dimensions_size(),
                rhs_broadcast_dims.end(),
                dnums.rhs_batch_dimensions_size() + lhs_outer_dims);
      new_rhs = dot->AddInstruction(HloInstruction::CreateBroadcast(
          new_lhs->shape(), new_rhs, rhs_broadcast_dims));
    }

    TF_ASSIGN_OR_RETURN(HloInstruction * new_dot,
                        MakeBinaryHlo(HloOpcode::kMultiply, new_lhs, new_rhs));
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
                      RemoveDegenerateDimensionFromDot(dot));
  if (removed_degenerate_dimensions) {
    return OkStatus();
  }

  TF_ASSIGN_OR_RETURN(bool removed_transposes,
                      RemoveTransposesFromDotOperands(dot));
  if (removed_transposes) {
    return OkStatus();
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleGather(HloInstruction* gather) {
  const Shape& operand_shape = gather->operand(0)->shape();
  if (ShapeUtil::IsZeroElementArray(operand_shape)) {
    return ReplaceInstruction(gather, MakeScalarLike(gather, 0));
  }

  // Gathering from a scalar operand is simply a broadcast of that scalar
  if (ShapeUtil::IsEffectiveScalar(operand_shape)) {
    HloInstruction* new_operand = gather->mutable_operand(0);
    if (operand_shape.rank()) {
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
  if (operand_shape.rank() == 1 &&
      operand_shape.dimensions(0) <= options_.very_small_gather_size() &&
      gather->gather_dimension_numbers().index_vector_dim() ==
          index_shape.rank() &&
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
  return OkStatus();
}

namespace {
StatusOr<std::unique_ptr<HloInstruction>> MinMaxToClamp(
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

Status AlgebraicSimplifierVisitor::HandleMaximum(HloInstruction* maximum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(maximum, m::Maximum(m::Op(&lhs), m::Op(&rhs))));

  // max(x, -inf) -> x
  PrimitiveType ty = maximum->shape().element_type();
  if (primitive_util::IsIntegralType(ty) ||
      (primitive_util::IsFloatingPointType(ty) &&
       options_.minmax_propagate_nan())) {
    Literal min_val = LiteralUtil::MinValue(ty);
    if (IsAll(lhs, min_val)) {
      return ReplaceInstruction(maximum, rhs);
    }
    if (IsAll(rhs, min_val)) {
      return ReplaceInstruction(maximum, lhs);
    }
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
      return OkStatus();
    }
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleMinimum(HloInstruction* minimum) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(minimum, m::Minimum(m::Op(&lhs), m::Op(&rhs))));

  // min(x, inf) -> x
  PrimitiveType ty = minimum->shape().element_type();
  if (primitive_util::IsIntegralType(ty) ||
      (primitive_util::IsFloatingPointType(ty) &&
       options_.minmax_propagate_nan())) {
    Literal max_val = LiteralUtil::MaxValue(ty);
    if (IsAll(lhs, max_val)) {
      return ReplaceInstruction(minimum, rhs);
    }
    if (IsAll(rhs, max_val)) {
      return ReplaceInstruction(minimum, lhs);
    }
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleClamp(HloInstruction* clamp) {
  HloInstruction* clamp_lower_bound;
  HloInstruction* clamp_upper_bound;
  HloInstruction* to_clamp;
  CHECK(Match(clamp, m::Clamp(m::Op(&clamp_lower_bound), m::Op(&to_clamp),
                              m::Op(&clamp_upper_bound))));

  // clamp(a, clamp(a, x, b), b) -> clamp(a, x, b)
  if (Match(to_clamp, m::Clamp(m::Op().Is(clamp_lower_bound), m::Op(),
                               m::Op().Is(clamp_upper_bound))) &&
      ReplaceInstructionIfCompatible(clamp, to_clamp)) {
    return OkStatus();
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleMultiply(HloInstruction* multiply) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(multiply, m::Multiply(m::Op(&lhs), m::Op(&rhs))));
  // LHS*1 => LHS
  VLOG(10) << "trying transform [LHS*1 => LHS]: " << multiply->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(multiply, lhs)) {
    return OkStatus();
  }
  // 1*RHS => RHS
  VLOG(10) << "trying transform [1*RHS => RHS]: " << multiply->ToString();
  if (IsAll(lhs, 1) && ReplaceInstructionIfCompatible(multiply, rhs)) {
    return OkStatus();
  }

  // 0*RHS => 0. Only applies for integral types for correct NaN-handling.
  if (IsAll(lhs, 0) &&
      primitive_util::IsIntegralType(multiply->shape().element_type()) &&
      ReplaceInstructionIfCompatible(multiply, lhs)) {
    return OkStatus();
  }
  // LHS*0 => 0
  if (IsAll(rhs, 0) &&
      primitive_util::IsIntegralType(multiply->shape().element_type()) &&
      ReplaceInstructionIfCompatible(multiply, rhs)) {
    return OkStatus();
  }

  {
    HloInstruction* abs_operand;
    if (lhs == rhs && Match(lhs, m::Abs(m::Op(&abs_operand))) &&
        !ShapeUtil::ElementIsComplex(abs_operand->shape())) {
      TF_RETURN_IF_ERROR(multiply->ReplaceOperandWith(0, abs_operand));
      TF_RETURN_IF_ERROR(multiply->ReplaceOperandWith(1, abs_operand));
      MarkAsChanged();
      return OkStatus();
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

  VLOG(10) << "trying transform [rsqrt(B) * rsqrt(B) => 1/B] "
           << multiply->ToString();
  HloInstruction* b;
  if (Match(multiply, m::Multiply(m::Rsqrt(m::Op(&b)), m::Rsqrt(m::Op(&b)))) &&
      IsPositive(b, options_)) {
    return ReplaceWithNewInstruction(
        multiply,
        HloInstruction::CreateBinary(multiply->shape(), HloOpcode::kDivide,
                                     MakeScalarLike(b, 1), b));
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleNegate(HloInstruction* negate) {
  // negate(negate(x)) => x
  HloInstruction* x;
  if (Match(negate, m::Negate(m::Negate(m::Op(&x)))) &&
      ReplaceInstructionIfCompatible(negate, x)) {
    return OkStatus();
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleNot(HloInstruction* logical_not) {
  // not(not(x)) => x
  HloInstruction* x;
  if (Match(logical_not, m::Not(m::Not(m::Op(&x)))) &&
      ReplaceInstructionIfCompatible(logical_not, x)) {
    return OkStatus();
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleOr(HloInstruction* logical_or) {
  HloInstruction *lhs, *rhs;
  CHECK(Match(logical_or, m::Or(m::Op(&lhs), m::Op(&rhs))));

  // Simplify logical or
  if (ShapeUtil::HasPrimitiveType(lhs->shape(), xla::PRED) &&
      ShapeUtil::HasPrimitiveType(rhs->shape(), xla::PRED)) {
    // A || True => True
    VLOG(10) << "trying transform [A || True => True]: "
             << logical_or->ToString();
    if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(logical_or, rhs)) {
      return OkStatus();
    }
    // True || A => True
    VLOG(10) << "trying transform [True || A => True]: "
             << logical_or->ToString();
    if (IsAll(lhs, 1) && ReplaceInstructionIfCompatible(logical_or, lhs)) {
      return OkStatus();
    }
  }

  // A || False => A and A | 0 => A
  VLOG(10) << "trying transform [A || False => A]: " << logical_or->ToString();
  if (IsAll(rhs, 0) && ReplaceInstructionIfCompatible(logical_or, lhs)) {
    return OkStatus();
  }

  // False || A => A and 0 | A => A
  VLOG(10) << "trying transform [False || A => A]: " << logical_or->ToString();
  if (IsAll(lhs, 0) && ReplaceInstructionIfCompatible(logical_or, rhs)) {
    return OkStatus();
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleLog(HloInstruction* log) {
  // ln(exp(A)) => A
  VLOG(10) << "trying transform [ln(exp(A)) => A]: " << log->ToString();
  HloInstruction *a, *b;
  if (Match(log, m::Log(m::Exp(m::Op(&a)))) &&
      ReplaceInstructionIfCompatible(log, a)) {
    return OkStatus();
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleGetTupleElement(
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
      return OkStatus();
    }
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleOptimizationBarrier(
    HloInstruction* barrier) {
  if (!barrier->shape().IsTuple() ||
      barrier == computation_->root_instruction()) {
    return OkStatus();
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
    return OkStatus();
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
  return OkStatus();
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

Status AlgebraicSimplifierVisitor::HandleBroadcast(HloInstruction* broadcast) {
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

  // A degenerate broadcast that has the same input and output rank can be
  // converted into a transpose.
  if (broadcast->shape().rank() == operand->shape().rank() &&
      ShapeUtil::ElementsIn(broadcast->shape()) ==
          ShapeUtil::ElementsIn(operand->shape())) {
    VLOG(10) << "transform broadcast(X) -> transpose(X) where "
                "n(broadcast(X)) == n(X)";
    return ReplaceWithNewInstruction(
        broadcast,
        HloInstruction::CreateTranspose(broadcast->shape(), operand, dims));
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
    // A Broadcast that feeds a unary element-wise operation can sink the
    // broadcast after the unary element-wise operation.
    TF_ASSIGN_OR_RETURN(
        bool sink_succeeded,
        TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(broadcast));
    if (sink_succeeded) {
      MarkAsChanged();
      return OkStatus();
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
        VLOG(10) << "transform permuting/subset  of a scalar broadcast into "
                 << "a single broadcast";
        HloInstruction* new_broadcast = user->AddInstruction(
            HloInstruction::CreateBroadcast(user->shape(), operand, {}));
        // Use HloInstruction::ReplaceAllUsesWith instead of
        // HloComputation::ReplaceWithNewInstruction because we are replacing an
        // instruction other than the visited instruction.
        MarkAsChanged();
        return user->ReplaceAllUsesWith(new_broadcast);
      }
    }
    return OkStatus();
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
    return OkStatus();
  }
  if (options_.enable_normalize_broadcast_operand() &&
      ShapeUtil::HasDegenerateDimensions(operand->shape())) {
    auto new_operand =
        operand->parent()->AddInstruction(HloInstruction::CreateReshape(
            ShapeUtil::DropDegenerateDimensions(operand->shape()), operand));
    std::vector<int64_t> new_dims;
    new_dims.reserve(new_operand->shape().rank());
    for (int64_t i = 0; i < operand->shape().rank(); ++i) {
      if (operand->shape().dimensions(i) != 1) {
        new_dims.push_back(dims[i]);
      }
    }
    return ReplaceWithNewInstruction(
        broadcast, HloInstruction::CreateBroadcast(broadcast->shape(),
                                                   new_operand, new_dims));
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleCompare(HloInstruction* compare) {
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
      lhs->opcode() == HloOpcode::kIota && IsAll(rhs, 0)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, false));
  } else if (compare->comparison_direction() == ComparisonDirection::kGt &&
             IsAll(lhs, 0) && rhs->opcode() == HloOpcode::kIota) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, false));
  } else if (compare->comparison_direction() == ComparisonDirection::kGe &&
             lhs->opcode() == HloOpcode::kIota && IsAll(rhs, 0)) {
    return ReplaceInstruction(compare, MakeScalarLike(compare, true));
  } else if (compare->comparison_direction() == ComparisonDirection::kLe &&
             IsAll(lhs, 0) && rhs->opcode() == HloOpcode::kIota) {
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
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleConvert(HloInstruction* convert) {
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
  return OkStatus();
}

// Complex(Real(c), Imag(c)) -> c
Status AlgebraicSimplifierVisitor::HandleComplex(HloInstruction* complex) {
  HloInstruction *c0, *c1;
  if (Match(complex, m::Complex(m::Real(m::Op(&c0)), m::Imag(m::Op(&c1)))) &&
      c0 == c1) {
    return ReplaceInstruction(complex, c0);
  }
  return OkStatus();
}

// Real(Complex(r, i)) -> r
Status AlgebraicSimplifierVisitor::HandleReal(HloInstruction* real) {
  HloInstruction* op;
  if (Match(real, m::Real(m::Complex(m::Op(&op), m::Op())))) {
    return ReplaceInstruction(real, op);
  }
  return OkStatus();
}

// Imag(Complex(r, i)) -> i
Status AlgebraicSimplifierVisitor::HandleImag(HloInstruction* imag) {
  HloInstruction* op;
  if (Match(imag, m::Imag(m::Complex(m::Op(), m::Op(&op))))) {
    return ReplaceInstruction(imag, op);
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleIota(HloInstruction* instruction) {
  // iota -> zero if the iota dimension never produces an element other than
  // zero.
  auto* iota = Cast<HloIotaInstruction>(instruction);
  if (iota->shape().dimensions(iota->iota_dimension()) <= 1) {
    return ReplaceInstruction(iota, MakeScalarLike(iota, 0));
  }
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandlePad(HloInstruction* pad) {
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
    for (int64_t i = 0; i < pad->shape().rank(); ++i) {
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
      return OkStatus();
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
      pad->operand(0)->operand(0)->shape().rank() <= pad_dims) {
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
    // TODO(b/34628603): Add support for negative padding in the backends, or
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandlePower(HloInstruction* power) {
  VLOG(10) << "trying transform [pow(A, 0) => 1]: " << power->ToString();
  HloInstruction *lhs, *rhs;
  CHECK(Match(power, m::Power(m::Op(&lhs), m::Op(&rhs))));
  if (IsAll(rhs, 0)) {
    return ReplaceInstruction(power, MakeScalarLike(power, 1));
  }

  VLOG(10) << "trying transform [pow(A, 1) => A]: " << power->ToString();
  if (IsAll(rhs, 1) && ReplaceInstructionIfCompatible(power, lhs)) {
    return OkStatus();
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

  return OkStatus();
}

StatusOr<bool>
AlgebraicSimplifierVisitor::TryToSinkBroadcastAfterOpWithUniqueNonScalarOperand(
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

    // Check if all the operands of the user are compatible broadcasts for
    // sinking. (They are either scalar broadcasts or broadcasts casting
    // from/to the same shape/dimensions)
    int64_t compatible_broadcast_count = 0;
    int64_t broadcast_use_count = 0;
    for (HloInstruction* user_operand : user->operands()) {
      if (is_compatible_broadcast(user_operand)) {
        ++compatible_broadcast_count;
      } else if (broadcast == user_operand) {
        ++broadcast_use_count;
      }
    }
    if (compatible_broadcast_count + broadcast_use_count !=
        user->operand_count()) {
      continue;
    }
    std::vector<HloInstruction*> new_operands;
    new_operands.reserve(user->operand_count());

    Shape changed_shape;
    for (HloInstruction* user_operand : user->operands()) {
      // If this is a broadcast operand that is not our original broadcast input
      // to this function then we might need to change the input.
      if (is_compatible_broadcast(user_operand)) {
        // If this is a broadcast from a scalar value rewrite a broadcast from
        // the scalar to the new shape enforced from the other broadcast
        // operands.
        if (is_scalar_broadcast(user_operand)) {
          changed_shape = ShapeUtil::ChangeElementType(
              operand->shape(), user_operand->shape().element_type());
          simplifier_->UpdateLayout(&changed_shape);
          new_operands.push_back(
              user_operand->AddInstruction(HloInstruction::CreateBroadcast(
                  changed_shape, user_operand->mutable_operand(0), {})));
        } else {
          // For the non-scalar broadcasts we guarantee that the shape of the
          // operand of the broadcast needs to be already a compatible shape.
          new_operands.push_back(user_operand->mutable_operand(0));
        }
      } else {
        CHECK_EQ(broadcast, user_operand);
        new_operands.push_back(operand);
      }
    }
    VLOG(4) << "Sinking broadcast after user:";
    VLOG(4) << "  old broadcast: " << broadcast->ToString();
    VLOG(4) << "  old user: " << user->ToString();
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
    int64_t b_value = c->literal().GetFirstElement<T>();
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
    uint64_t b_value = c->literal().GetFirstElement<T>();
    if (absl::has_single_bit(b_value)) {
      HloInstruction* mask_amount =
          remainder->AddInstruction(simplifier->CreateConstantWithLayoutUpdated(
              LiteralUtil::CreateR0<T>(b_value - 1)));
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

Status AlgebraicSimplifierVisitor::HandleRemainder(HloInstruction* remainder) {
  HloInstruction *a, *b;
  CHECK(Match(remainder, m::Remainder(m::Op(&a), m::Op(&b))));

  // (A % B) % B == A % B.
  if (Match(a, m::Remainder(m::Op(), m::Op().Is(b)))) {
    return ReplaceInstruction(remainder, a);
  }

  // A % B => A & (B - 1) if B is a power of 2.
  switch (remainder->shape().element_type()) {
    case S8:
      if (std::unique_ptr<HloInstruction> shift =
              TryRemainderToAnd<int8_t>(remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S16:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<int16_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S32:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<int32_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case S64:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<int64_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U8:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<uint8_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U16:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<uint16_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U32:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<uint32_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    case U64:
      if (std::unique_ptr<HloInstruction> shift = TryRemainderToAnd<uint64_t>(
              remainder, computation_, simplifier_)) {
        return ReplaceWithNewInstruction(remainder, std::move(shift));
      }
      break;
    default:
      break;
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
      if (max_val.has_value() &&
          FitsInIntegralType(*max_val, iota->shape().element_type())) {
        return ReplaceWithNewInstruction(
            remainder,
            HloInstruction::CreateBinary(remainder->shape(),
                                         HloOpcode::kRemainder, iota, bcast));
      }
    }
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleReshape(HloInstruction* reshape) {
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
      for (int64_t i = 0; i <= old_slice_shape.rank(); ++i) {
        if (absl::c_linear_search(trivial_reshape->deleted_dimensions, i)) {
          continue;
        }
        while (absl::c_linear_search(trivial_reshape->inserted_dimensions,
                                     new_slice_shape.size())) {
          new_slice_shape.push_back(1);
          new_dus_operands.push_back(zero);
        }
        if (i < old_slice_shape.rank()) {
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
  return OkStatus();
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

Status AlgebraicSimplifierVisitor::HandleReverse(HloInstruction* reverse) {
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
        return OkStatus();
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
        return OkStatus();
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
  return OkStatus();
}

StatusOr<bool> AlgebraicSimplifierVisitor::TrySimplifyScalarSlice(
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
    if (slice->shape().rank() != 1) {
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

StatusOr<bool> AlgebraicSimplifierVisitor::TryToReorderSliceAndReshape(
    HloInstruction* slice) {
  CHECK_EQ(slice->opcode(), HloOpcode::kSlice);
  if (!IsUnstridedSlice(slice)) {
    return false;
  }
  HloInstruction* reshape = slice->mutable_operand(0);
  if (reshape->opcode() != HloOpcode::kReshape) {
    return false;
  }
  HloInstruction* new_slice_operand = reshape->mutable_operand(0);
  int64_t slice_rank = slice->shape().rank();
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
    const int64_t rank = new_slice_shape.rank();
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
StatusOr<bool> AlgebraicSimplifierVisitor::TryToReorderSliceAndReverse(
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

Status AlgebraicSimplifierVisitor::HandleSlice(HloInstruction* slice) {
  // Delete no-op slices, i.e. where shape = operand shape.
  if (ReplaceInstructionIfCompatible(slice, slice->mutable_operand(0))) {
    return OkStatus();
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
    for (int64_t i = 0; i < slice->shape().rank(); ++i) {
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
      return OkStatus();
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
      IsUnstridedSlice(slice) && IsUnstridedSlice(slice->operand(0))) {
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
    return OkStatus();
  }

  HloInstruction* broadcast;
  HloInstruction* broadcast_operand;
  if (Match(slice,
            m::Slice(m::Broadcast(&broadcast, m::Op(&broadcast_operand))))) {
    std::vector<int64_t> new_slice_starts;
    std::vector<int64_t> new_slice_strides;
    std::vector<int64_t> new_slice_limits;
    new_slice_starts.reserve(broadcast_operand->shape().rank());
    new_slice_strides.reserve(broadcast_operand->shape().rank());
    new_slice_limits.reserve(broadcast_operand->shape().rank());
    for (int64_t dim : broadcast->dimensions()) {
      new_slice_starts.push_back(slice->slice_starts(dim));
      new_slice_strides.push_back(slice->slice_strides(dim));
      new_slice_limits.push_back(slice->slice_limits(dim));
    }
    VLOG(3) << "Sink broadcast through slice";
    VLOG(3) << "Original slice: " << slice->ToString();
    VLOG(3) << "Original broadcast: " << broadcast->ToString();
    auto new_slice_shape = broadcast_operand->shape();
    for (int64_t i = 0; i < broadcast_operand->shape().rank(); ++i) {
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

  // Try to simplify concat -> slice to an operand of concat.
  if (slice->operand(0)->opcode() == HloOpcode::kConcatenate &&
      IsUnstridedSlice(slice)) {
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

  // Do not try to reorder slices and reshapes after layout assignment as it may
  // be invalid.
  if (!options_.is_layout_sensitive()) {
    TF_ASSIGN_OR_RETURN(replaced, TryToReorderSliceAndReshape(slice));
  }
  if (replaced) {
    return OkStatus();
  }

  bool reversed = false;
  if (Match(slice, m::Slice(m::Reverse(m::Op())))) {
    TF_ASSIGN_OR_RETURN(reversed, TryToReorderSliceAndReverse(slice));
  }
  if (reversed) {
    return OkStatus();
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleRsqrt(HloInstruction* rsqrt) {
  VLOG(10) << "trying transform [rsqrt(Pow(A, -2)) => |A|] "
           << rsqrt->ToString();
  HloInstruction* rsqrt_operand = rsqrt->mutable_operand(0);
  if (rsqrt_operand->opcode() == HloOpcode::kPower &&
      IsAll(rsqrt_operand->operand(1), -2) &&
      IsPositive(rsqrt_operand, options_)) {
    return ReplaceWithNewInstruction(
        rsqrt, HloInstruction::CreateUnary(rsqrt->shape(), HloOpcode::kAbs,
                                           rsqrt_operand->mutable_operand(0)));
  }

  VLOG(10) << "trying transform [rsqrt(Divide(1, A)) => sqrt(A)] "
           << rsqrt->ToString();
  if (rsqrt_operand->opcode() == HloOpcode::kDivide &&
      IsAll(rsqrt_operand->operand(0), 1) &&
      IsPositive(rsqrt_operand->operand(1), options_)) {
    return ReplaceWithNewInstruction(
        rsqrt, HloInstruction::CreateUnary(rsqrt->shape(), HloOpcode::kSqrt,
                                           rsqrt_operand->mutable_operand(1)));
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleDynamicSlice(
    HloInstruction* dynamic_slice) {
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
  for (int64_t dim = 0; dim < operand->shape().rank(); ++dim) {
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
    new_indices.reserve(broadcast_operand->shape().rank());
    std::vector<int64_t> new_slice_sizes;
    new_slice_sizes.reserve(broadcast_operand->shape().rank());

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
      for (int64_t i = 0; i < broadcast_operand->shape().rank(); ++i) {
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
    starts.reserve(reshape_operand->shape().rank());
    std::vector<int64_t> slice_sizes;
    slice_sizes.reserve(reshape_operand->shape().rank());
    for (int64_t dim = 0; dim < reshape_operand->shape().rank(); ++dim) {
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

  // Convert a dynamic slice into a slice if all offsets are constant and the
  // operand is not constant.
  if (operand->opcode() != HloOpcode::kConstant &&
      absl::c_all_of(absl::MakeSpan(dynamic_slice->operands().begin() + 1,
                                    dynamic_slice->operands().end()),
                     [](HloInstruction* operand) {
                       return operand->opcode() == HloOpcode::kConstant &&
                              ShapeUtil::ElementIsIntegral(operand->shape());
                     })) {
    const int64_t rank = operand->shape().rank();
    std::vector<int64_t> slice_starts(rank);
    std::vector<int64_t> slice_limits(rank);
    std::vector<int64_t> slice_strides(rank, 1);

    for (int64_t i = 0; i < rank; ++i) {
      std::optional<int64_t> offset =
          dynamic_slice->operand(i + 1)->literal().GetFirstInteger();
      if (!offset || *offset < 0) {
        return OkStatus();
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
  if (operand->shape().rank() == 1 && dynamic_slice->shape().rank() == 1 &&
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
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
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
    PaddingConfig padding_config;
    if (compatible) {
      for (int64_t dim = 0; dim < updated_shape.rank(); ++dim) {
        auto padding_config_dim = padding_config.add_dimensions();
        auto slice_dim_start = update_start_indx->operand(dim + offset);
        if (!Match(slice_dim_start, m::ConstantScalar())) {
          compatible = false;
          break;
        }
        VLOG(2) << "slice: " << slice_dim_start->ToString();
        std::optional<int64_t> beg =
            slice_dim_start->literal().GetFirstInteger();
        if (!beg) {
          compatible = false;
          break;
        }
        VLOG(2) << "beg value: " << *beg;
        auto update_width = ShapeUtil::GetDimension(update_shape, dim);
        auto bcast_width = ShapeUtil::GetDimension(updated_shape, dim);
        // Clamp beg so that it is non-negative.
        *beg = std::max<int64_t>(0, *beg);
        // Clamp beg so that it is in-bounds.
        *beg = std::min<int64_t>(bcast_width - update_width, *beg);
        VLOG(2) << "adjusted beg value: " << *beg;
        padding_config_dim->set_edge_padding_low(*beg);
        padding_config_dim->set_edge_padding_high(bcast_width -
                                                  (*beg + update_width));
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
  for (int64_t dim = 0; dim < dus_update->shape().rank(); ++dim) {
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
    return OkStatus();
  }
  return OkStatus();
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

Status AlgebraicSimplifierVisitor::HandleReduce(HloInstruction* hlo) {
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
    return OkStatus();
  }

  // TODO(b/131122694): Most of those optimizations below can be done for
  // multi-output reduces.
  if (multi_output_reduce) {
    return OkStatus();
  }

  // A Transpose feeding a reduce can simply permute the reduction dimensions
  // field if the output of the reduce is a vector or scalar. Higher ranked
  // result may require a transpose of the output.
  if (arg->opcode() == HloOpcode::kTranspose &&
      (reduce->shape().rank() < 2 || arg->user_count() == 1 ||
       absl::c_all_of(arg->users(), [](HloInstruction* use) {
         return use->opcode() == HloOpcode::kReduce;
       }))) {
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

  // A reshape that collapses multiple dimensions into a dimension being
  // reduced can just reduce all of those dimensions instead of doing a
  // collapsing reshape before a reduction.
  if (options_.enable_reduce_of_reshape() &&
      arg->opcode() == HloOpcode::kReshape) {
    std::vector<std::pair<int64_t, int64_t>> unmodified_dims =
        ShapeUtil::DimensionsUnmodifiedByReshape(arg->operand(0)->shape(),
                                                 arg->shape());
    std::vector<bool> arg_dim_in_output(arg->shape().rank(), true);
    std::vector<bool> arg_dim_unmodified(arg->shape().rank(), false);
    for (auto dim : dimensions) {
      arg_dim_in_output[dim] = false;
    }
    for (auto dim_pair : unmodified_dims) {
      arg_dim_unmodified[dim_pair.second] = true;
    }
    // The goal is to verify that all dimensions that are not removed in the
    // reduce are unmodified by the reshape. For example:
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
      for (int64_t i = 0; i < arg->operand(0)->shape().rank(); ++i) {
        if (!dimensions_not_to_reduce.contains(i)) {
          new_reduce_dimensions.push_back(i);
        }
      }
      return ReplaceWithNewInstruction(
          reduce, HloInstruction::CreateReduce(
                      reduce_result_shape, arg->mutable_operand(0), init_value,
                      new_reduce_dimensions, function));
    }
  }
  // Convert Reduce(concat({a,b,...})) to
  //  map(reduce(a),map(reduce(b),...,))
  //
  // This should make fusion easier or use less memory bandwidth in the unfused
  // case.
  if (arg->opcode() == HloOpcode::kConcatenate &&
      absl::c_linear_search(reduce->dimensions(),
                            arg->concatenate_dimension())) {
    HloInstruction* old_reduce = nullptr;
    for (HloInstruction* operand : arg->operands()) {
      HloInstruction* new_reduce = reduce->AddInstruction(
          HloInstruction::CreateReduce(reduce_result_shape, operand, init_value,
                                       reduce->dimensions(), function));
      if (old_reduce != nullptr) {
        new_reduce = reduce->AddInstruction(HloInstruction::CreateMap(
            reduce_result_shape, {old_reduce, new_reduce}, function));
      }
      old_reduce = new_reduce;
    }
    return ReplaceInstruction(reduce, old_reduce);
  }

  HloInstruction *dot, *lhs, *rhs;
  // Convert Reduce(Dot(X,Y)) to Dot(X,Y) if any of the dimensions reduced were
  // batch dimensions of the dot. The transformation supports reducing other
  // dimensions as well.
  if (options_.enable_dot_strength_reduction() &&
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
    TF_ASSIGN_OR_RETURN(
        auto new_dot,
        MakeDotHlo(lhs, rhs, new_dnums, dot->precision_config(),
                   /*preferred_element_type=*/dot->shape().element_type()));
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
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleReduceWindow(HloInstruction* hlo) {
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
  // TODO(b/73062247) Variadic reduce window is not yet supported in simplifier.
  if (multi_output_reduce_window) {
    return OkStatus();
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
    return OkStatus();
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
    return OkStatus();
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
      return OkStatus();
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
  const int64_t rank = reduce_window->shape().rank();
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

Status AlgebraicSimplifierVisitor::HandleSelect(HloInstruction* select) {
  // select(x, y, y) -> y.
  if (select->operand(1) == select->operand(2)) {
    return ReplaceInstruction(select, select->mutable_operand(1));
  }
  // select(true, x, y) -> x.
  if (IsAll(select->operand(0), true)) {
    return ReplaceInstruction(select, select->mutable_operand(1));
  }
  // select(false, x, y) -> y.
  if (IsAll(select->operand(0), false)) {
    return ReplaceInstruction(select, select->mutable_operand(2));
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

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleScatter(HloInstruction* hlo) {
  auto* scatter = Cast<HloScatterInstruction>(hlo);

  if (absl::c_all_of(scatter->scatter_updates(),
                     [](const HloInstruction* updates) {
                       return ShapeUtil::IsZeroElementArray(updates->shape());
                     }) &&
      ReplaceInstructionIfCompatible(scatter, scatter->scatter_operands())) {
    return OkStatus();
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
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleSort(HloInstruction* sort) {
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
  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleSqrt(HloInstruction* sqrt) {
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
  return OkStatus();
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

Status AlgebraicSimplifierVisitor::HandleTranspose(HloInstruction* transpose) {
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

  // Convert transpose(dot(a,b)) to dot(b,a).
  auto do_transpose_of_dot = [&]() -> StatusOr<bool> {
    if (options_.supports_non_canonical_dots() ||
        operand->opcode() != HloOpcode::kDot || operand->user_count() != 1) {
      return false;
    }
    HloInstruction* dot = operand;
    HloInstruction* lhs = dot->mutable_operand(0);
    HloInstruction* rhs = dot->mutable_operand(1);

    const int64_t rank = dot->shape().rank();
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
            lhs->shape().rank() ||
        dnums.rhs_contracting_dimensions_size() == 0 ||
        dnums.rhs_contracting_dimensions_size() +
                dnums.rhs_batch_dimensions_size() + 1 !=
            rhs->shape().rank()) {
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
    return OkStatus();
  }

  // Transpose(dot(a,b))->dot(b,a) for any dot.
  HloInstruction *lhs, *rhs, *dot;
  if (options_.supports_non_canonical_dots() &&
      Match(operand, m::Dot(&dot, m::Op(&lhs), m::Op(&rhs))) &&
      dot->user_count() == 1) {
    TF_ASSIGN_OR_RETURN(bool did_transform, [&]() -> StatusOr<bool> {
      const auto& dnums = dot->dot_dimension_numbers();
      const int64_t num_batch_dims = dnums.lhs_batch_dimensions_size();
      for (int64_t i = 0; i < num_batch_dims; ++i) {
        if (transpose->dimensions(i) >= num_batch_dims) {
          return false;
        }
      }
      const int64_t num_rhs_outer_dims =
          rhs->shape().rank() - (dnums.rhs_contracting_dimensions_size() +
                                 dnums.rhs_batch_dimensions_size());
      const int64_t num_lhs_outer_dims =
          lhs->shape().rank() - (dnums.lhs_contracting_dimensions_size() +
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
      dot->SetupDerivedInstruction(new_dot);
      TF_CHECK_OK(ReplaceInstruction(transpose, new_dot));
      return true;
    }());
    if (did_transform) {
      return OkStatus();
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
    return OkStatus();
  }

  // Replace reshape of a transpose of a reshape with concatenated slicing if
  // the reshape/transpose combination can be interpreted as a space-to-depth
  // transformation.
  if (operand->opcode() == HloOpcode::kReshape &&
      transpose->user_count() == 1 &&
      HloOpcode::kReshape == transpose->users()[0]->opcode()) {
    VLOG(2) << "trying depth-to-space transform";
    HloInstruction* reshape_operand = operand->mutable_operand(0);
    HloInstruction* outer_reshape = transpose->users()[0];
    TF_ASSIGN_OR_RETURN(
        bool did_transform, ([&]() -> StatusOr<bool> {
          if (operand->shape().dimensions_size() !=
              reshape_operand->shape().dimensions_size() + 1) {
            return false;
          }

          // Check that the reshape is splitting a single dimension into two.
          int64_t split_dim = 0;
          bool found_split_dims = false;
          for (int64_t dim = 0; dim < reshape_operand->shape().rank(); dim++) {
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
               dim < reshape_operand->shape().rank(); dim++) {
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
          for (int64_t dim = 0; dim < operand->shape().rank(); dim++) {
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
          for (int64_t dim = 0; dim < operand->shape().rank(); dim++) {
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
          for (int64_t dim = 0; dim < reshape_operand->shape().rank(); dim++) {
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
            const auto rank = reshape_operand->shape().rank();
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
      return OkStatus();
    }
  }

  return OkStatus();
}

StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvInputPad(
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

StatusOr<bool> AlgebraicSimplifierVisitor::FoldConvFilterPad(
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

StatusOr<bool> AlgebraicSimplifierVisitor::SwapConvOperands(
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

StatusOr<bool> AlgebraicSimplifierVisitor::SimplifyConvToDot(
    HloInstruction* convolution) {
  auto* lhs = convolution->mutable_operand(0);
  auto* rhs = convolution->mutable_operand(1);
  const auto& window = convolution->window();
  const ConvolutionDimensionNumbers& dnums =
      convolution->convolution_dimension_numbers();

  if (!options_.enable_conv_simplification()) {
    return false;
  }

  // TODO(b/31337498): For now, we cowardly refuse to do this optimization in
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

StatusOr<bool> AlgebraicSimplifierVisitor::SimplifyConvToMultiply(
    HloInstruction* convolution) {
  if (options_.is_layout_sensitive() ||
      absl::c_linear_search(convolution->precision_config().operand_precision(),
                            PrecisionConfig::PACKED_NIBBLE)) {
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
  DimensionVector input_permutation(input_shape.rank());
  DimensionVector kernel_permutation(kernel_shape.rank());

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

Status AlgebraicSimplifierVisitor::HandleConvolution(
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
    return OkStatus();
  }

  // Try to merge dilation of the filter with the convolution's window.
  TF_ASSIGN_OR_RETURN(bool folded_filter_pad, FoldConvFilterPad(convolution));
  if (folded_filter_pad) {
    return OkStatus();
  }

  // Try to swap convolution operands.
  TF_ASSIGN_OR_RETURN(bool swapped, SwapConvOperands(convolution));
  if (swapped) {
    return OkStatus();
  }
  // Try to replace the convolution with a kDot or a kMultiply instruction.
  TF_ASSIGN_OR_RETURN(bool replaced_with_dot, SimplifyConvToDot(convolution));
  if (replaced_with_dot) {
    return OkStatus();
  }
  TF_ASSIGN_OR_RETURN(bool replaced_with_multiply,
                      SimplifyConvToMultiply(convolution));
  if (replaced_with_multiply) {
    return OkStatus();
  }

  return OkStatus();
}

Status AlgebraicSimplifierVisitor::HandleMap(HloInstruction* map) {
  auto* map_computation = map->to_apply();
  auto* map_root = map_computation->root_instruction();
  if (map_root->opcode() == HloOpcode::kParameter) {
    ReplaceInstructionIfCompatible(
        map, map->mutable_operand(map_root->parameter_number()));
    return OkStatus();
  }
  if (map_root->opcode() == HloOpcode::kConstant) {
    if (!ShapeUtil::IsScalar(map_root->shape())) {
      return OkStatus();
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
    return OkStatus();
  }
  std::vector<HloInstruction*> new_operands;
  for (auto* root_operand : map_root->operands()) {
    if (root_operand->opcode() != HloOpcode::kParameter) {
      return OkStatus();
    }
    new_operands.push_back(
        map->mutable_operand(root_operand->parameter_number()));
  }
  auto clone = map_root->CloneWithNewOperands(map->shape(), new_operands);
  return ReplaceWithNewInstruction(map, std::move(clone));
}

StatusOr<bool> AlgebraicSimplifier::Run(
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
