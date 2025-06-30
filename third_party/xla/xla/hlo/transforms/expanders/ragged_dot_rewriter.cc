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

#include "xla/hlo/transforms/expanders/ragged_dot_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {

namespace {

std::unique_ptr<HloComputation> CreateScalarAddComputation(PrimitiveType type) {
  auto embedded_builder = HloComputation::Builder("add");
  auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(type, {}), "lhs"));
  auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(type, {}), "rhs"));
  embedded_builder.AddInstruction(
      HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
  return embedded_builder.Build();
}

std::unique_ptr<HloInstruction> Zero(PrimitiveType type) {
  return HloInstruction::CreateConstant(LiteralUtil::Zero(type));
}

// Takes an array of shape [batch_dims..., num_groups] and returns an array of
// the same shape with the elements of the array along the last dimension
// now representing the cumulative sum of all elements in the input array up to
// the current group.
std::unique_ptr<HloInstruction> CreateCumulativeSum(
    HloInstruction* group_sizes) {
  int64_t batch_dims = group_sizes->shape().dimensions().size() - 1;
  int64_t num_groups = group_sizes->shape().dimensions(batch_dims);

  Window cumsum_window;
  // Add batch dimensions.
  for (int i = 0; i < batch_dims; ++i) {
    WindowDimension* dim = cumsum_window.add_dimensions();
    dim->set_size(1);
    dim->set_padding_low(0);
    dim->set_padding_high(0);
    dim->set_stride(1);
    dim->set_window_dilation(1);
    dim->set_base_dilation(1);
  }
  // Add group dimension.
  WindowDimension* dim = cumsum_window.add_dimensions();
  dim->set_size(num_groups);
  dim->set_padding_low(num_groups - 1);
  dim->set_padding_high(0);
  dim->set_stride(1);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);

  auto type = group_sizes->shape().element_type();
  HloComputation* add = group_sizes->GetModule()->AddEmbeddedComputation(
      CreateScalarAddComputation(type));
  auto zero = group_sizes->parent()->AddInstruction(Zero(type));
  return HloInstruction::CreateReduceWindow(group_sizes->shape(), group_sizes,
                                            zero, cumsum_window, add);
}

// Expands ragged_op by one dimension with each row in the new dimension
// representing each group. It then zeros out the elements that don't belong to
// that group.
std::unique_ptr<HloInstruction> RaggedToDense(HloInstruction* ragged_operand,
                                              HloInstruction* group_sizes,
                                              int new_dim_index,
                                              int ragged_dim) {
  auto computation = ragged_operand->parent();
  HloInstruction* cumulative_sum =
      computation->AddInstruction(CreateCumulativeSum(group_sizes));
  // Group dimension is always the last dimension.
  int group_dim = group_sizes->shape().dimensions().size() - 1;
  int num_groups = group_sizes->shape().dimensions(group_dim);

  // Create the mask to zero out the top half. First slice off the last element
  // of the cumulative sum array and append 0 to the beginning.
  auto slice_shape = cumulative_sum->shape();
  slice_shape.set_dimensions(group_dim, num_groups - 1);
  llvm::SmallVector<int64_t> slice_starts(slice_shape.dimensions().size(), 0);
  llvm::SmallVector<int64_t> slice_limits(slice_shape.dimensions().size());
  for (int i = 0; i < slice_shape.dimensions().size(); ++i) {
    slice_limits[i] = slice_shape.dimensions(i);
  }
  llvm::SmallVector<int64_t> slice_strides(slice_shape.dimensions().size(), 1);
  auto slice = computation->AddInstruction(HloInstruction::CreateSlice(
      slice_shape, cumulative_sum, slice_starts, slice_limits, slice_strides));

  // Concat a zero to the beginning of the cumulative sum array (represents how
  // many elements to zero out on top for the first group (zero)).
  auto zero_slice_shape = slice_shape;
  zero_slice_shape.set_dimensions(group_dim, 1);
  auto group_type = group_sizes->shape().element_type();
  auto zero_group_type = computation->AddInstruction(Zero(group_type));
  auto zero_arr = computation->AddInstruction(
      HloInstruction::CreateBroadcast(zero_slice_shape, zero_group_type, {}));
  auto concat = computation->AddInstruction(HloInstruction::CreateConcatenate(
      group_sizes->shape(), {zero_arr, slice}, group_dim));

  // Broadcast cumulative sum array to operand shape + group dimension.
  auto old_shape = ragged_operand->shape();
  llvm::SmallVector<int64_t> new_shape_dims;
  new_shape_dims.reserve(old_shape.dimensions().size() + 1);
  new_shape_dims.append(old_shape.dimensions().begin(),
                        old_shape.dimensions().end());
  new_shape_dims.insert(new_shape_dims.begin() + new_dim_index, num_groups);
  auto new_shape_gs = ShapeUtil::MakeShape(group_type, new_shape_dims);
  llvm::SmallVector<int64_t> broadcast_dims(
      concat->shape().dimensions().size());
  for (int i = 0; i < concat->shape().dimensions().size(); ++i) {
    broadcast_dims[i] = i;
  }
  auto broadcast = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape_gs, concat, broadcast_dims));

  auto iota = computation->AddInstruction(HloInstruction::CreateIota(
      ShapeUtil::MakeShape(group_type, {old_shape.dimensions(ragged_dim)}), 0));
  auto broadcast_iota = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape_gs, iota, {ragged_dim + 1}));

  // Zero out the top if row < cum_group_size.
  auto new_shape_pred = ShapeUtil::MakeShape(PRED, new_shape_gs.dimensions());
  auto compare_top = computation->AddInstruction(HloInstruction::CreateCompare(
      new_shape_pred, broadcast, broadcast_iota, ComparisonDirection::kLe));

  // Put zeros on the bottom if row >= cum_group_size.
  auto broadcast_groups =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          new_shape_gs, cumulative_sum, broadcast_dims));
  auto compare_bottom =
      computation->AddInstruction(HloInstruction::CreateCompare(
          new_shape_pred, broadcast_iota, broadcast_groups,
          ComparisonDirection::kLt));
  // Combine top+bottom half masks.
  auto mask = computation->AddInstruction(HloInstruction::CreateBinary(
      new_shape_pred, HloOpcode::kAnd, compare_top, compare_bottom));

  // Load LHS & broadcast it for each group.
  llvm::SmallVector<int64_t> old_dims_new_shape(old_shape.dimensions().size());
  for (int i = 0; i < new_dim_index; ++i) {
    old_dims_new_shape[i] = i;
  }
  for (int i = new_dim_index; i < old_shape.dimensions().size(); ++i) {
    old_dims_new_shape[i] = i + 1;
  }
  auto element_type = old_shape.element_type();
  auto new_shape = ShapeUtil::MakeShape(element_type, new_shape_dims);
  auto broadcast_ragged =
      computation->AddInstruction(HloInstruction::CreateBroadcast(
          new_shape, ragged_operand, old_dims_new_shape));

  auto zero_operand_type = computation->AddInstruction(Zero(element_type));
  auto zero_new_shape = computation->AddInstruction(
      HloInstruction::CreateBroadcast(new_shape, zero_operand_type, {}));
  // Apply mask to operand.
  return HloInstruction::CreateTernary(new_shape, HloOpcode::kSelect, mask,
                                       broadcast_ragged, zero_new_shape);
}

enum class RaggedDotMode {
  kRaggedNonContracting,
  kRaggedContracting,
  kRaggedBatch,
};

RaggedDotMode GetRaggedDotMode(int lhs_ragged_dim,
                               const DotDimensionNumbers& dnums) {
  if (std::find(dnums.lhs_contracting_dimensions().begin(),
                dnums.lhs_contracting_dimensions().end(),
                lhs_ragged_dim) != dnums.lhs_contracting_dimensions().end()) {
    return RaggedDotMode::kRaggedContracting;
  }
  if (std::find(dnums.lhs_batch_dimensions().begin(),
                dnums.lhs_batch_dimensions().end(),
                lhs_ragged_dim) != dnums.lhs_batch_dimensions().end()) {
    return RaggedDotMode::kRaggedBatch;
  }
  return RaggedDotMode::kRaggedNonContracting;
}

int FindRhsRaggedDim(const DotDimensionNumbers& dot_dims, int lhs_ragged_dim) {
  const auto& lhs_contracting_dims = dot_dims.lhs_contracting_dimensions();
  int ragged_contracting_index =
      std::distance(std::find(lhs_contracting_dims.begin(),
                              lhs_contracting_dims.end(), lhs_ragged_dim),
                    lhs_contracting_dims.begin());
  return dot_dims.rhs_contracting_dimensions(ragged_contracting_index);
}

DotDimensionNumbers CreateRaggedNonContractingDotDims(
    const DotDimensionNumbers& old_dims, int new_dim_index, int rhs_group_dim) {
  DotDimensionNumbers new_dims;
  new_dims.add_lhs_contracting_dimensions(new_dim_index);
  for (auto dim : old_dims.lhs_contracting_dimensions()) {
    new_dims.add_lhs_contracting_dimensions(dim + 1);
  }
  for (auto dim : old_dims.lhs_batch_dimensions()) {
    new_dims.add_lhs_batch_dimensions(dim);
  }
  new_dims.add_rhs_contracting_dimensions(rhs_group_dim);
  for (auto dim : old_dims.rhs_contracting_dimensions()) {
    new_dims.add_rhs_contracting_dimensions(dim);
  }
  for (auto dim : old_dims.rhs_batch_dimensions()) {
    new_dims.add_rhs_batch_dimensions(dim);
  }
  return new_dims;
}

DotDimensionNumbers CreateRaggedContractingDotDims(
    const DotDimensionNumbers& old_dims) {
  DotDimensionNumbers new_dims;
  new_dims.add_lhs_batch_dimensions(0);
  for (auto dim : old_dims.lhs_batch_dimensions()) {
    new_dims.add_lhs_batch_dimensions(dim);
  }
  new_dims.add_rhs_batch_dimensions(0);
  for (auto dim : old_dims.rhs_batch_dimensions()) {
    new_dims.add_rhs_batch_dimensions(dim);
  }
  for (auto dim : old_dims.rhs_contracting_dimensions()) {
    new_dims.add_rhs_contracting_dimensions(dim + 1);
  }
  for (auto dim : old_dims.lhs_contracting_dimensions()) {
    new_dims.add_lhs_contracting_dimensions(dim + 1);
  }
  return new_dims;
}

absl::StatusOr<std::unique_ptr<HloInstruction>> RaggedToGeneral(
    HloRaggedDotInstruction* ragged_dot) {
  const auto& ragged_dims = ragged_dot->ragged_dot_dimension_numbers();
  const auto& dot_dims = ragged_dims.dot_dimension_numbers();
  if (ragged_dims.lhs_ragged_dimensions().size() != 1) {
    return absl::UnimplementedError("lhs_ragged_dimensions must have size 1");
  }
  int lhs_ragged_dim = ragged_dims.lhs_ragged_dimensions(0);
  // Unsure about this new_dim_index. It's similar to the way jax does it, but
  // it comes with an assumption that batch dimensions always come first. They
  // then also do a transpose to move the group dimension to the front, which
  // I haven't implemented here.
  int new_dim_index = dot_dims.rhs_batch_dimensions().size();

  auto* computation = ragged_dot->parent();
  auto lhs = ragged_dot->mutable_operand(0);
  auto rhs = ragged_dot->mutable_operand(1);
  auto group_sizes = ragged_dot->mutable_operand(2);
  DotDimensionNumbers new_dot_dims;

  RaggedDotMode mode =
      GetRaggedDotMode(lhs_ragged_dim, ragged_dims.dot_dimension_numbers());
  switch (mode) {
    case RaggedDotMode::kRaggedNonContracting: {
      if (ragged_dims.rhs_group_dimensions().size() != 1) {
        return absl::UnimplementedError(
            "rhs_group_dimensions must have size equal to 1 when lhs ragged "
            "dimension is a non-contracting dimension");
      }
      int rhs_group_dim = ragged_dims.rhs_group_dimensions(0);
      lhs = computation->AddInstruction(
          RaggedToDense(lhs, group_sizes, new_dim_index, lhs_ragged_dim));
      new_dot_dims = CreateRaggedNonContractingDotDims(dot_dims, new_dim_index,
                                                       rhs_group_dim);
      break;
    }
    case RaggedDotMode::kRaggedContracting: {
      lhs = computation->AddInstruction(
          RaggedToDense(lhs, group_sizes, new_dim_index, lhs_ragged_dim));
      int rhs_ragged_dim = FindRhsRaggedDim(dot_dims, lhs_ragged_dim);
      rhs = computation->AddInstruction(
          RaggedToDense(rhs, group_sizes, new_dim_index, rhs_ragged_dim));
      new_dot_dims = CreateRaggedContractingDotDims(dot_dims);
      break;
    }
    case RaggedDotMode::kRaggedBatch: {
      new_dot_dims = dot_dims;
      break;
    }
  }

  return HloInstruction::CreateDot(ragged_dot->shape(), lhs, rhs, new_dot_dims,
                                   ragged_dot->precision_config());
}

}  // namespace

absl::StatusOr<bool> RaggedDotRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Gather all Ragged Dot operations.
  std::vector<HloRaggedDotInstruction*> ragged_dots;
  for (auto* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kRaggedDot) {
        ragged_dots.push_back(Cast<HloRaggedDotInstruction>(instruction));
      }
    }
  }

  for (auto* ragged_dot : ragged_dots) {
    TF_ASSIGN_OR_RETURN(auto general_dot, RaggedToGeneral(ragged_dot));
    general_dot->set_metadata(ragged_dot->metadata());
    TF_RETURN_IF_ERROR(ragged_dot->parent()->ReplaceWithNewInstruction(
        ragged_dot, std::move(general_dot)));
  }

  return !ragged_dots.empty();
}

}  // namespace xla
