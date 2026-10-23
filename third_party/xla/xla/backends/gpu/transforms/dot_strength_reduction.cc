/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dot_strength_reduction.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/codegen/triton/support_legacy.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/matmul_indexing_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

namespace {

HloInstruction* ConvertTo(HloInstruction* instruction, PrimitiveType type,
                          const OpMetadata* metadata) {
  if (instruction->shape().element_type() == type) {
    return instruction;
  }
  Shape new_shape = instruction->shape();
  new_shape.set_element_type(type);
  return instruction->parent()->AddInstruction(
      HloInstruction::CreateConvert(new_shape, instruction), metadata);
}

// Transposes the dot operand to make dimensions in "batch, non-contracting,
// contracting" order, sorted by index within each category.
HloInstruction* PermuteDotOperandDimensions(HloInstruction* operand,
                                            DotOperandDims* dims,
                                            const OpMetadata* metadata) {
  std::vector<int64_t> permutation;
  for (auto kind : {DotOperandDims::kBatch, DotOperandDims::kNonContracting,
                    DotOperandDims::kContracting}) {
    for (auto index : dims->Indices(kind)) {
      permutation.push_back(index);
    }
  }
  if (absl::c_is_sorted(permutation)) {
    return operand;
  }
  Shape new_shape = ShapeUtil::PermuteDimensions(permutation, operand->shape());
  operand = operand->parent()->AddInstruction(
      HloInstruction::CreateTranspose(new_shape, operand, permutation),
      metadata);
  dims->ApplyPermutation(permutation);
  return operand;
}

HloInstruction* BroadcastDimensions(HloInstruction* operand,
                                    int64_t insert_before,
                                    absl::Span<const int64_t> bounds_to_insert,
                                    const OpMetadata* metadata) {
  if (bounds_to_insert.empty()) {
    return operand;
  }

  std::vector<int64_t> broadcast_dimensions(
      operand->shape().dimensions().size());
  std::iota(broadcast_dimensions.begin(),
            broadcast_dimensions.begin() + insert_before, 0);
  std::iota(broadcast_dimensions.begin() + insert_before,
            broadcast_dimensions.end(),
            insert_before + bounds_to_insert.size());

  Shape new_shape = ShapeUtil::InsertDimensionsAtIndex(
      operand->shape(), insert_before, bounds_to_insert);
  return operand->parent()->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape, operand, broadcast_dimensions),
      metadata);
}

HloComputation* CreateScalarAddComputation(HloModule* module,
                                           PrimitiveType type) {
  HloComputation::Builder b("scalar_add_computation");
  Shape shape = ShapeUtil::MakeShape(type, {});
  auto scalar_lhs =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
  auto scalar_rhs =
      b.AddInstruction(HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
  auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
  return module->AddEmbeddedComputation(b.Build(scalar_op));
}

// Reduces the last `num_dims_to_reduce` dimensions of `instruction`.
HloInstruction* ReduceDimensions(HloInstruction* instruction,
                                 size_t num_dims_to_reduce,
                                 PrimitiveType accumulator_type,
                                 const OpMetadata* metadata) {
  if (num_dims_to_reduce == 0) {
    return instruction;
  }
  HloComputation* computation = instruction->parent();

  std::vector<int64_t> reduce_dims(num_dims_to_reduce);
  absl::c_iota(reduce_dims,
               instruction->shape().dimensions().size() - num_dims_to_reduce);

  Shape reduce_shape =
      ShapeUtil::DeleteDimensions(reduce_dims, instruction->shape());
  reduce_shape.set_element_type(accumulator_type);

  Literal zero_literal = LiteralUtil::Zero(accumulator_type);
  HloInstruction* zero = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(zero_literal)));

  HloComputation* add_computation =
      CreateScalarAddComputation(computation->parent(), accumulator_type);

  return computation->AddInstruction(
      HloInstruction::CreateReduce(
          reduce_shape, ConvertTo(instruction, accumulator_type, metadata),
          zero, reduce_dims, add_computation),
      metadata);
}

struct OneHotDotMatch {
  HloInstruction* indices;
  HloInstruction* one_hot;
  HloInstruction* weights;
  int64_t weights_contract_dim;
  bool lhs_is_one_hot;
};

// Matches a dot where one operand is a one-hot tensor built by comparing
// integer `indices` against an `iota` along the contracting dimension.
//
// Input (4 tokens, 128 classes, 512 features):
//   indices = s32[4] parameter(0)
//   bcast   = s32[4,128] broadcast(indices), dimensions={0}
//   classes = s32[4,128] iota(), iota_dimension=1
//   eq      = pred[4,128] compare(bcast, classes), direction=EQ
//   one_hot = bf16[4,128] convert(eq)
//   weights = bf16[128,512] parameter(1)
//   dot     = bf16[4,512] dot(one_hot, weights), lhs_contracting_dims={1},
//                                                rhs_contracting_dims={0}
//
// Output: `OneHotDotMatch{indices, one_hot, weights,
// /*weights_contract_dim=*/0,
//                         /*lhs_is_one_hot=*/true}`
std::optional<OneHotDotMatch> MatchOneHotDot(HloDotInstruction* dot) {
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();
  if (dnums.lhs_batch_dimensions_size() != 0 ||
      dnums.rhs_batch_dimensions_size() != 0 ||
      dnums.lhs_contracting_dimensions_size() != 1 ||
      dnums.rhs_contracting_dimensions_size() != 1) {
    return std::nullopt;
  }

  // TODO(b/485881552): Default-precision F32 dots run in TF32 on GPU, while
  // gather keeps full F32 precision. Only rewrite F32 dots when full F32
  // precision is requested.
  const PrecisionConfig::Algorithm alg = dot->precision_config().algorithm();
  const auto& precisions = dot->precision_config().operand_precision();
  const bool full_f32_precision =
      alg == PrecisionConfig::ALG_DOT_F32_F32_F32 ||
      (alg == PrecisionConfig::ALG_UNSET && precisions.size() == 2 &&
       absl::c_all_of(precisions,
                      [](int p) { return p == PrecisionConfig::HIGHEST; }));

  for (int side = 0; side < 2; ++side) {
    HloInstruction* one_hot = dot->mutable_operand(side);
    HloInstruction* weights = dot->mutable_operand(1 - side);
    const int64_t one_hot_contract_dim =
        (side == 0) ? dnums.lhs_contracting_dimensions(0)
                    : dnums.rhs_contracting_dimensions(0);
    const int64_t weights_contract_dim =
        (side == 0) ? dnums.rhs_contracting_dimensions(0)
                    : dnums.lhs_contracting_dimensions(0);

    if (weights->shape().element_type() == F32 && !full_f32_precision) {
      continue;
    }
    const int64_t one_hot_rank = one_hot->shape().dimensions().size();
    if (one_hot_rank < 2 || one_hot_contract_dim != one_hot_rank - 1 ||
        weights->shape().dimensions().empty()) {
      continue;
    }

    const int64_t num_classes = one_hot->shape().dimensions().back();
    if (num_classes <= 1 ||
        weights->shape().dimensions(weights_contract_dim) != num_classes) {
      continue;
    }

    HloInstruction* curr = one_hot;
    while (curr->opcode() == HloOpcode::kConvert ||
           curr->opcode() == HloOpcode::kReshape ||
           curr->opcode() == HloOpcode::kBitcast) {
      HloInstruction* op = curr->mutable_operand(0);
      if (op->shape().dimensions().empty() ||
          op->shape().dimensions().back() != num_classes) {
        break;
      }
      curr = op;
    }

    if (curr->opcode() != HloOpcode::kCompare ||
        curr->comparison_direction() != ComparisonDirection::kEq ||
        curr->shape().dimensions().size() < 2 ||
        curr->shape().dimensions().back() != num_classes) {
      continue;
    }

    const int64_t cmp_rank = curr->shape().dimensions().size();
    auto is_iota_along_last_dim = [&](const HloInstruction* op) {
      if (op->opcode() == HloOpcode::kIota) {
        return Cast<HloIotaInstruction>(op)->iota_dimension() == cmp_rank - 1;
      }
      if (op->opcode() == HloOpcode::kBroadcast &&
          op->dimensions().size() == 1 && op->dimensions(0) == cmp_rank - 1 &&
          op->operand(0)->opcode() == HloOpcode::kIota) {
        return Cast<HloIotaInstruction>(op->operand(0))->iota_dimension() == 0;
      }
      return false;
    };

    auto get_broadcasted_indices = [&](HloInstruction* op) -> HloInstruction* {
      if (op->opcode() != HloOpcode::kBroadcast ||
          op->dimensions().size() != cmp_rank - 1) {
        return nullptr;
      }
      for (int64_t i = 0; i < cmp_rank - 1; ++i) {
        if (op->dimensions(i) != i) {
          return nullptr;
        }
      }
      HloInstruction* indices = op->mutable_operand(0);
      if (!ShapeUtil::ElementIsIntegral(indices->shape())) {
        return nullptr;
      }
      return indices;
    };

    HloInstruction* indices = nullptr;
    if (is_iota_along_last_dim(curr->operand(1))) {
      indices = get_broadcasted_indices(curr->mutable_operand(0));
    } else if (is_iota_along_last_dim(curr->operand(0))) {
      indices = get_broadcasted_indices(curr->mutable_operand(1));
    }
    if (indices == nullptr) {
      continue;
    }

    const int bit_width =
        primitive_util::BitWidth(indices->shape().element_type());
    const int value_bits = ShapeUtil::ElementIsSigned(indices->shape())
                               ? bit_width - 1
                               : bit_width;
    if (value_bits < 63 && (num_classes - 1) >= (int64_t{1} << value_bits)) {
      continue;
    }

    absl::Span<const int64_t> one_hot_nc_dims =
        one_hot->shape().dimensions().first(one_hot_rank - 1);
    int64_t one_hot_nc_elements = 1;
    for (int64_t d : one_hot_nc_dims) {
      one_hot_nc_elements *= d;
    }
    if (one_hot_nc_elements != ShapeUtil::ElementsIn(indices->shape())) {
      continue;
    }

    return OneHotDotMatch{indices, one_hot, weights, weights_contract_dim,
                          /*lhs_is_one_hot=*/side == 0};
  }

  return std::nullopt;
}

struct SinkableBroadcast {
  HloInstruction* sunk_bcast = nullptr;
  HloInstruction* gather_source = nullptr;
};

// If `weights` is a broadcast of a smaller table (possibly followed by
// reshapes), returns the unbroadcasted table so we can gather from it first
// and broadcast the result afterward.
//
// Input:
//   table   = bf16[128,64] parameter(1)
//   bcast   = bf16[1,128,8,64] broadcast(table), dimensions={1,3}
//   weights = bf16[128,512] reshape(bcast)
//
// Output: `sunk_bcast = bcast`, `gather_source = table` (so we gather
// `bf16[4,64]` from `table` instead of `bf16[4,512]` from `weights`).
SinkableBroadcast MatchSinkableBroadcast(HloInstruction* weights,
                                         int64_t weights_contract_dim,
                                         int64_t num_classes) {
  if (weights_contract_dim != 0) {
    return {/*sunk_bcast=*/nullptr, /*gather_source=*/weights};
  }
  HloInstruction* weights_curr = weights;
  while (weights_curr->opcode() == HloOpcode::kReshape ||
         weights_curr->opcode() == HloOpcode::kBitcast) {
    weights_curr = weights_curr->mutable_operand(0);
  }
  if (weights_curr->opcode() != HloOpcode::kBroadcast) {
    return {/*sunk_bcast=*/nullptr, /*gather_source=*/weights};
  }
  HloInstruction* bcast_operand = weights_curr->mutable_operand(0);
  if (bcast_operand->shape().dimensions().empty() ||
      bcast_operand->shape().dimensions(0) != num_classes ||
      weights_curr->dimensions().size() !=
          bcast_operand->shape().dimensions().size() ||
      !absl::c_is_sorted(weights_curr->dimensions())) {
    return {/*sunk_bcast=*/nullptr, /*gather_source=*/weights};
  }
  for (int64_t d = 0; d < weights_curr->dimensions(0); ++d) {
    if (weights_curr->shape().dimensions(d) != 1) {
      return {/*sunk_bcast=*/nullptr, /*gather_source=*/weights};
    }
  }
  return {/*sunk_bcast=*/weights_curr, /*gather_source=*/bcast_operand};
}

// Builds an in-bounds mask `0 <= indices <= num_classes - 1` (or
// `indices <= num_classes - 1` for unsigned types).
//
// Input:  reshaped_indices = s32[4], num_classes = 128
// Output: in_bounds = pred[4] and(compare(reshaped_indices, 0, GE),
//                                 compare(reshaped_indices, 127, LE))
absl::StatusOr<HloInstruction*> CreateInBoundsMask(
    HloInstruction* reshaped_indices, int64_t num_classes,
    const OpMetadata* metadata) {
  HloComputation* computation = reshaped_indices->parent();
  const PrimitiveType indices_type = reshaped_indices->shape().element_type();
  Shape pred_shape =
      ShapeUtil::ChangeElementType(reshaped_indices->shape(), PRED);
  ABSL_ASSIGN_OR_RETURN(
      Literal max_class_lit,
      LiteralUtil::CreateR0<int64_t>(num_classes - 1).Convert(indices_type));
  HloInstruction* max_class_const = computation->AddInstruction(
      HloInstruction::CreateConstant(std::move(max_class_lit)));
  HloInstruction* max_class_bcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(reshaped_indices->shape(),
                                      max_class_const, {}),
      metadata);
  HloInstruction* in_bounds = computation->AddInstruction(
      HloInstruction::CreateCompare(pred_shape, reshaped_indices,
                                    max_class_bcast, ComparisonDirection::kLe),
      metadata);
  if (ShapeUtil::ElementIsSigned(reshaped_indices->shape())) {
    HloInstruction* zero_idx = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::Zero(indices_type)));
    HloInstruction* zero_idx_bcast = computation->AddInstruction(
        HloInstruction::CreateBroadcast(reshaped_indices->shape(), zero_idx,
                                        {}),
        metadata);
    HloInstruction* ge_zero = computation->AddInstruction(
        HloInstruction::CreateCompare(pred_shape, reshaped_indices,
                                      zero_idx_bcast, ComparisonDirection::kGe),
        metadata);
    in_bounds = computation->AddInstruction(
        HloInstruction::CreateBinary(pred_shape, HloOpcode::kAnd, ge_zero,
                                     in_bounds),
        metadata);
  }
  return in_bounds;
}

// Re-applies the sunk broadcast and reshape after `select`.
//
// Input:
//   selected   = bf16[4,64] select(...)
//   sunk_bcast = bf16[1,128,8,64] broadcast(bf16[128,64]), dimensions={1,3}
//   dot_shape  = bf16[4,512]
//
// Output:
//   bcast  = bf16[4,8,64] broadcast(selected), dimensions={0,2}
//   result = bf16[4,512] reshape(bcast)
HloInstruction* SinkBroadcastAfterSelect(
    HloInstruction* selected, const HloInstruction* sunk_bcast,
    absl::Span<const int64_t> one_hot_nc_dims, bool lhs_is_one_hot,
    const Shape& dot_shape, const OpMetadata* metadata) {
  HloComputation* computation = selected->parent();
  const int64_t k = one_hot_nc_dims.size();
  const int64_t bcast_dim0 = sunk_bcast->dimensions(0);
  absl::Span<const int64_t> bcast_suffix =
      sunk_bcast->shape().dimensions().subspan(bcast_dim0 + 1);
  std::vector<int64_t> new_bcast_shape_dims;
  new_bcast_shape_dims.reserve(k + bcast_suffix.size());
  std::vector<int64_t> new_bcast_dims;
  new_bcast_dims.reserve(k + sunk_bcast->dimensions().size() - 1);
  if (lhs_is_one_hot) {
    new_bcast_shape_dims.insert(new_bcast_shape_dims.end(),
                                one_hot_nc_dims.begin(), one_hot_nc_dims.end());
    new_bcast_shape_dims.insert(new_bcast_shape_dims.end(),
                                bcast_suffix.begin(), bcast_suffix.end());
    for (int64_t i = 0; i < k; ++i) {
      new_bcast_dims.push_back(i);
    }
    for (size_t i = 1; i < sunk_bcast->dimensions().size(); ++i) {
      new_bcast_dims.push_back(sunk_bcast->dimensions(i) - bcast_dim0 - 1 + k);
    }
  } else {
    new_bcast_shape_dims.insert(new_bcast_shape_dims.end(),
                                bcast_suffix.begin(), bcast_suffix.end());
    new_bcast_shape_dims.insert(new_bcast_shape_dims.end(),
                                one_hot_nc_dims.begin(), one_hot_nc_dims.end());
    for (size_t i = 1; i < sunk_bcast->dimensions().size(); ++i) {
      new_bcast_dims.push_back(sunk_bcast->dimensions(i) - bcast_dim0 - 1);
    }
    for (int64_t i = 0; i < k; ++i) {
      new_bcast_dims.push_back(bcast_suffix.size() + i);
    }
  }

  Shape new_bcast_shape =
      ShapeUtil::MakeShape(dot_shape.element_type(), new_bcast_shape_dims);
  selected = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_bcast_shape, selected,
                                      new_bcast_dims),
      metadata);
  if (selected->shape().dimensions() != dot_shape.dimensions()) {
    selected = computation->AddInstruction(
        HloInstruction::CreateReshape(dot_shape, selected), metadata);
  }
  return selected;
}

// Rewrites a one-hot dot into a gather and an in-bounds select.
//
// Input:
//   dot = bf16[4,512] dot(one_hot /*bf16[4,128]*/, weights /*bf16[128,512]*/)
//
// Output:
//   start_indices = s32[4,1] reshape(indices)
//   gather        = bf16[4,512] gather(weights, start_indices)
//   in_bounds     = pred[4] and(compare(indices, 0, GE),
//                               compare(indices, 127, LE))
//   mask          = pred[4,512] broadcast(in_bounds), dimensions={0}
//   ROOT select   = bf16[4,512] select(mask, gather, zeros)
absl::StatusOr<HloInstruction*> ExpandOneHotDot(HloDotInstruction* dot,
                                                const OneHotDotMatch& match) {
  HloComputation* computation = dot->parent();
  const OpMetadata* metadata = &dot->metadata();
  HloInstruction* indices = match.indices;
  HloInstruction* one_hot = match.one_hot;
  HloInstruction* weights = match.weights;
  const int64_t weights_contract_dim = match.weights_contract_dim;
  const bool lhs_is_one_hot = match.lhs_is_one_hot;

  const int64_t num_classes = one_hot->shape().dimensions().back();
  absl::Span<const int64_t> one_hot_nc_dims =
      one_hot->shape().dimensions().first(one_hot->shape().dimensions().size() -
                                          1);
  const int64_t k = one_hot_nc_dims.size();
  const PrimitiveType indices_type = indices->shape().element_type();
  const PrimitiveType out_type = dot->shape().element_type();

  HloInstruction* reshaped_indices = indices;
  if (indices->shape().dimensions() != one_hot_nc_dims) {
    reshaped_indices = computation->AddInstruction(
        HloInstruction::CreateReshape(
            ShapeUtil::MakeShape(indices_type, one_hot_nc_dims), indices),
        metadata);
  }

  std::vector<int64_t> start_indices_dims(one_hot_nc_dims.begin(),
                                          one_hot_nc_dims.end());
  start_indices_dims.push_back(1);
  HloInstruction* start_indices = computation->AddInstruction(
      HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(indices_type, start_indices_dims),
          reshaped_indices),
      metadata);

  const auto [sunk_bcast, gather_source] =
      MatchSinkableBroadcast(weights, weights_contract_dim, num_classes);

  absl::Span<const int64_t> source_dims = gather_source->shape().dimensions();
  const int64_t m = source_dims.size() - 1;

  std::vector<int64_t> offset_dims(m);
  absl::c_iota(offset_dims, lhs_is_one_hot ? k : 0);

  std::vector<int64_t> slice_sizes;
  slice_sizes.reserve(source_dims.size());
  std::vector<int64_t> source_slice_dims;
  source_slice_dims.reserve(m);
  for (int64_t d = 0; d < source_dims.size(); ++d) {
    if (d == weights_contract_dim) {
      slice_sizes.push_back(1);
    } else {
      slice_sizes.push_back(source_dims[d]);
      source_slice_dims.push_back(source_dims[d]);
    }
  }

  std::vector<int64_t> gather_dims;
  gather_dims.reserve(k + m);
  if (lhs_is_one_hot) {
    gather_dims.insert(gather_dims.end(), one_hot_nc_dims.begin(),
                       one_hot_nc_dims.end());
    gather_dims.insert(gather_dims.end(), source_slice_dims.begin(),
                       source_slice_dims.end());
  } else {
    gather_dims.insert(gather_dims.end(), source_slice_dims.begin(),
                       source_slice_dims.end());
    gather_dims.insert(gather_dims.end(), one_hot_nc_dims.begin(),
                       one_hot_nc_dims.end());
  }
  Shape gather_shape = (sunk_bcast == nullptr)
                           ? dot->shape()
                           : ShapeUtil::MakeShape(out_type, gather_dims);
  Shape raw_gather_shape = ShapeUtil::ChangeElementType(
      gather_shape, gather_source->shape().element_type());

  GatherDimensionNumbers gather_dnums =
      HloGatherInstruction::MakeGatherDimNumbers(
          /*offset_dims=*/offset_dims,
          /*collapsed_slice_dims=*/{weights_contract_dim},
          /*start_index_map=*/{weights_contract_dim},
          /*index_vector_dim=*/k);
  // Note: GPU gather emitters clamp out-of-bounds indices to [0, num_classes-1]
  // in IR, and the `select` below masks out-of-bounds entries to zero.
  HloInstruction* gather = computation->AddInstruction(
      HloInstruction::CreateGather(raw_gather_shape, gather_source,
                                   start_indices, gather_dnums, slice_sizes,
                                   /*indices_are_sorted=*/false),
      metadata);
  gather = ConvertTo(gather, out_type, metadata);

  ABSL_ASSIGN_OR_RETURN(HloInstruction * in_bounds,
                   CreateInBoundsMask(reshaped_indices, num_classes, metadata));

  HloInstruction* in_bounds_bcast = in_bounds;
  if (in_bounds->shape().dimensions() != gather_shape.dimensions()) {
    std::vector<int64_t> bcast_dims(k);
    absl::c_iota(bcast_dims, lhs_is_one_hot ? 0 : m);
    in_bounds_bcast = computation->AddInstruction(
        HloInstruction::CreateBroadcast(
            ShapeUtil::ChangeElementType(gather_shape, PRED), in_bounds,
            bcast_dims),
        metadata);
  }

  HloInstruction* zero_out = computation->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(out_type)));
  HloInstruction* zero_gather = computation->AddInstruction(
      HloInstruction::CreateBroadcast(gather_shape, zero_out, {}), metadata);
  HloInstruction* selected = computation->AddInstruction(
      HloInstruction::CreateTernary(gather_shape, HloOpcode::kSelect,
                                    in_bounds_bcast, gather, zero_gather),
      metadata);

  if (sunk_bcast != nullptr) {
    selected = SinkBroadcastAfterSelect(selected, sunk_bcast, one_hot_nc_dims,
                                        lhs_is_one_hot, dot->shape(), metadata);
  }

  return selected;
}

}  // namespace

absl::StatusOr<HloInstruction*> DotStrengthReduction::ExpandInstruction(
    HloInstruction* instruction) {
  HloDotInstruction* dot = Cast<HloDotInstruction>(instruction);
  if (enable_one_hot_dot_to_gather_) {
    if (std::optional<OneHotDotMatch> match = MatchOneHotDot(dot)) {
      return ExpandOneHotDot(dot, *match);
    }
  }
  const OpMetadata* metadata = &dot->metadata();
  ABSL_ASSIGN_OR_RETURN(auto dot_dims, DotOperandDims::FromDot(dot));

  std::array<HloInstruction*, 2> operands = {dot->mutable_operand(0),
                                             dot->mutable_operand(1)};
  for (int i = 0; i < 2; ++i) {
    DotOperandDims& our_dims = dot_dims[i];
    DotOperandDims& other_dims = dot_dims[1 - i];
    // Convert operands to the dot resulting type.
    operands[i] = ConvertTo(operands[i], dot->shape().element_type(), metadata);
    // Ensure dimensions are in "batch, non-contracting, contracting" order.
    operands[i] = PermuteDotOperandDimensions(operands[i], &our_dims, metadata);

    // Both lhs and rhs will have [batch, lhs non-contracting, rhs
    // non-contracting, contracting] dimensions.
    // Therefore, we insert other side's non-contracting dimensions before or
    // after our contracting depending on the operand.
    int insert_before = our_dims.Rank(DotOperandDims::kBatch);
    if (i == 0) {
      insert_before += our_dims.Rank(DotOperandDims::kNonContracting);
    }

    operands[i] = BroadcastDimensions(
        operands[i], insert_before,
        other_dims.Sizes(DotOperandDims::kNonContracting), metadata);
  }

  // At this point, both operands have the same shape. Elementwise multiply.
  CHECK(operands[0]->shape().dimensions() == operands[1]->shape().dimensions());
  ABSL_ASSIGN_OR_RETURN(
      HloInstruction * flow,
      MakeMultiplyForDotPrecisionAlgorithm(
          operands[0], operands[1], dot->precision_config().algorithm()));
  flow->set_metadata(*metadata);

  // If there were any contracting dims, we need to reduce them.
  flow = ReduceDimensions(flow, dot_dims[0].Rank(DotOperandDims::kContracting),
                          GetGemmAccumulatorType(dot), metadata);

  // If the output type is different from what it was before (either because
  // reduction used a different accumulator type, or because types of operand
  // differed the output type for multiply), convert to the output type.
  flow = ConvertTo(flow, dot->shape().element_type(), metadata);
  return flow;
}

bool DotStrengthReduction::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kDot) {
    return false;
  }
  if (enable_one_hot_dot_to_gather_ &&
      MatchOneHotDot(Cast<HloDotInstruction>(instruction)).has_value()) {
    return true;
  }

  const HloDotInstruction* dot = Cast<HloDotInstruction>(instruction);
  const HloInstruction* lhs = dot->operand(0);
  const HloInstruction* rhs = dot->operand(1);
  const DotDimensionNumbers& dnums = dot->dot_dimension_numbers();

  const bool lhs_is_vector = (dnums.lhs_batch_dimensions_size() +
                                  dnums.lhs_contracting_dimensions_size() ==
                              lhs->shape().dimensions().size());
  const bool rhs_is_vector = (dnums.rhs_batch_dimensions_size() +
                                  dnums.rhs_contracting_dimensions_size() ==
                              rhs->shape().dimensions().size());
  // For s32xs32->s32 dots, with RHS contracting dimension 1,
  // the loop emitter is slow and other backends don't support it.
  // Rewriting it as a faster alternative.
  const bool is_favourable_s32_dot = [&]() {
    if (lhs->shape().element_type() != S32 ||
        rhs->shape().element_type() != S32 ||
        dot->shape().element_type() != S32) {
      return false;
    }
    if (dnums.rhs_contracting_dimensions().size() != 1 ||
        dnums.rhs_contracting_dimensions()[0] != 1) {
      return false;
    }
    return true;
  }();
  if (!lhs_is_vector && !rhs_is_vector && !is_favourable_s32_dot) {
    return false;
  }
  // Strength-reduce vector-vector dots since they are not supported by
  // GemmFusion.
  if (lhs_is_vector && rhs_is_vector) {
    return true;
  }

  absl::StatusOr<bool> is_too_small =
      IsMatrixMultiplicationTooSmallForRewriting(*instruction,
                                                 /*threshold=*/10000000);
  CHECK_OK(is_too_small.status());
  if (is_too_small.value()) {
    return true;
  }

  // If GemmFusion cannot handle this dot, we should strength-reduce it so that
  // it can be handled by the fusion pipeline.
  return !legacy_triton::CanTritonHandleGEMM(*dot, compute_capability_);
}

}  // namespace gpu
}  // namespace xla
