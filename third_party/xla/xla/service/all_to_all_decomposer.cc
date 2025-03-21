/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/all_to_all_decomposer.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"

namespace xla {
bool AllToAllDecomposer::InstructionMatchesPattern(
    HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kRaggedAllToAll) {
    auto* ragged_all_to_all =
        DynCast<HloRaggedAllToAllInstruction>(instruction);
    if (ragged_all_to_all == nullptr) {
      return false;
    }
    // Do not attempt to change layout constrained collectives.
    if (ragged_all_to_all->constrain_layout()) {
      return false;
    }
    return ragged_all_to_all->shape().dimensions_size() < min_array_rank_;
  }

  auto* all_to_all = DynCast<HloAllToAllInstruction>(instruction);
  if (all_to_all == nullptr) {
    return false;
  }
  // Do not attempt to change layout constrained collectives.
  if (all_to_all->constrain_layout()) {
    return false;
  }
  if (all_to_all->shape().IsTuple()) {
    return false;
  }
  if (decompose_to_tuple_) {
    return true;
  }
  return all_to_all->shape().dimensions_size() < min_array_rank_;
}

absl::StatusOr<HloInstruction*> AllToAllDecomposer::ExpandRaggedAllToAll(
    HloInstruction* instruction) {
  Shape input_shape = instruction->operand(0)->shape();
  Shape aliased_output_shape = instruction->operand(1)->shape();
  Shape output_shape = instruction->shape();
  CHECK_EQ(instruction->operand_count(), 6);
  CHECK_EQ(input_shape.dimensions_size(), output_shape.dimensions_size());
  CHECK_EQ(output_shape, aliased_output_shape)
      << "Output shape must match shape of operand 1 shape (which is aliased "
         "to output).";

  Shape new_input_shape;
  Shape new_output_shape;
  new_input_shape.set_element_type(input_shape.element_type());
  new_output_shape.set_element_type(output_shape.element_type());

  // New input and output shape are the same as original shape but dimensions
  // are padded with 1s until min_array_rank_.
  for (int64_t i = 0; i < input_shape.dimensions_size(); ++i) {
    new_input_shape.add_dimensions(input_shape.dimensions(i));
    new_output_shape.add_dimensions(output_shape.dimensions(i));
  }
  while (new_input_shape.dimensions_size() < min_array_rank_) {
    new_input_shape.add_dimensions(1);
    new_output_shape.add_dimensions(1);
  }
  *(new_input_shape.mutable_layout()) =
      LayoutUtil::GetDefaultLayoutForRank(min_array_rank_);
  *(new_output_shape.mutable_layout()) =
      LayoutUtil::GetDefaultLayoutForRank(min_array_rank_);

  // Reshape operands
  HloInstruction* operand_0_reshape =
      instruction->parent()->AddInstruction(HloInstruction::CreateReshape(
          new_input_shape, instruction->mutable_operand(0)));
  instruction->SetupDerivedInstruction(operand_0_reshape);
  HloInstruction* operand_1_reshape =
      instruction->parent()->AddInstruction(HloInstruction::CreateReshape(
          new_output_shape, instruction->mutable_operand(1)));
  instruction->SetupDerivedInstruction(operand_1_reshape);
  HloInstruction* ragged_all_to_all =
      instruction->parent()->AddInstruction(instruction->CloneWithNewOperands(
          new_output_shape,
          {operand_0_reshape, operand_1_reshape,
           instruction->mutable_operand(2), instruction->mutable_operand(3),
           instruction->mutable_operand(4), instruction->mutable_operand(5)}));
  HloInstruction* output_reshape = instruction->parent()->AddInstruction(
      HloInstruction::CreateReshape(instruction->shape(), ragged_all_to_all));
  instruction->SetupDerivedInstruction(output_reshape);
  return output_reshape;
}

absl::StatusOr<HloInstruction*> AllToAllDecomposer::ExpandInstruction(
    HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kRaggedAllToAll) {
    return ExpandRaggedAllToAll(instruction);
  }

  auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
  int64_t split_dim = *all_to_all->split_dimension();
  int64_t all_to_all_group_size =
      all_to_all->replica_groups().empty()
          ? instruction->GetModule()->config().replica_count()
          : all_to_all->replica_groups()[0].replica_ids_size();
  int64_t split_size =
      all_to_all->shape().dimensions(split_dim) / all_to_all_group_size;
  if (!decompose_to_tuple_) {
    Shape new_all_to_all_shape;
    new_all_to_all_shape.set_element_type(
        instruction->operand(0)->shape().element_type());
    for (int64_t i = 0; i < instruction->shape().dimensions_size(); ++i) {
      if (i != split_dim) {
        new_all_to_all_shape.add_dimensions(all_to_all->shape().dimensions(i));
        continue;
      }
      new_all_to_all_shape.add_dimensions(all_to_all_group_size);
      new_all_to_all_shape.add_dimensions(split_size);
      for (int64_t j = all_to_all->shape().dimensions_size() + 1;
           j < min_array_rank_; ++j) {
        new_all_to_all_shape.add_dimensions(1);
      }
    }
    *(new_all_to_all_shape.mutable_layout()) =
        LayoutUtil::GetDefaultLayoutForRank(min_array_rank_);
    HloInstruction* operand_reshape =
        instruction->parent()->AddInstruction(HloInstruction::CreateReshape(
            new_all_to_all_shape, instruction->mutable_operand(0)));
    instruction->SetupDerivedInstruction(operand_reshape);
    HloInstruction* all_to_all =
        instruction->parent()->AddInstruction(instruction->CloneWithNewOperands(
            new_all_to_all_shape, {operand_reshape}));
    HloInstruction* output_reshape = instruction->parent()->AddInstruction(
        HloInstruction::CreateReshape(instruction->shape(), all_to_all));
    instruction->SetupDerivedInstruction(output_reshape);
    return output_reshape;
  }
  DimensionVector slice_starts(all_to_all->shape().dimensions_size(), 0);
  DimensionVector slice_strides(all_to_all->shape().dimensions_size(), 1);
  DimensionVector slice_limits(all_to_all->shape().dimensions().begin(),
                               all_to_all->shape().dimensions().end());
  slice_limits[split_dim] = split_size;
  Shape slice_shape = all_to_all->shape();
  slice_shape.set_dimensions(split_dim, split_size);
  std::vector<HloInstruction*> slices;
  slices.reserve(all_to_all_group_size);
  HloInstruction* operand = all_to_all->mutable_operand(0);
  for (int64_t i = 0; i < all_to_all_group_size; ++i) {
    slices.push_back(
        all_to_all->parent()->AddInstruction(HloInstruction::CreateSlice(
            slice_shape, operand, slice_starts, slice_limits, slice_strides)));
    all_to_all->SetupDerivedInstruction(slices.back());
    slice_starts[split_dim] = slice_limits[split_dim];
    slice_limits[split_dim] += split_size;
  }
  Shape all_to_all_shape = ShapeUtil::MakeTupleShapeWithPtrs(
      std::vector<const Shape*>(all_to_all_group_size, &slice_shape));
  HloInstruction* new_all_to_all =
      all_to_all->parent()->AddInstruction(HloInstruction::CreateAllToAll(
          all_to_all_shape, slices, all_to_all->device_list(), false,
          all_to_all->channel_id(), std::nullopt));
  std::vector<HloInstruction*> gtes;
  gtes.reserve(all_to_all_group_size);
  for (int64_t i = 0; i < all_to_all_group_size; ++i) {
    gtes.push_back(all_to_all->parent()->AddInstruction(
        HloInstruction::CreateGetTupleElement(slice_shape, new_all_to_all, i)));
    all_to_all->SetupDerivedInstruction(new_all_to_all);
  }
  HloInstruction* concat = all_to_all->parent()->AddInstruction(
      HloInstruction::CreateConcatenate(all_to_all->shape(), gtes, split_dim));
  all_to_all->SetupDerivedInstruction(concat);
  return concat;
}

}  // namespace xla
