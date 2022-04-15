/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_to_all_decomposer.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {
bool AllToAllDecomposer::InstructionMatchesPattern(
    HloInstruction* instruction) {
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
  return all_to_all->shape().rank() < min_array_rank_;
}
StatusOr<HloInstruction*> AllToAllDecomposer::ExpandInstruction(
    HloInstruction* instruction) {
  auto* all_to_all = Cast<HloAllToAllInstruction>(instruction);
  int64_t split_dim = *all_to_all->split_dimension();
  int64_t all_to_all_group_size =
      all_to_all->replica_groups().empty()
          ? instruction->parent()->parent()->config().replica_count()
          : all_to_all->replica_groups()[0].replica_ids_size();
  int64_t split_size =
      all_to_all->shape().dimensions(split_dim) / all_to_all_group_size;
  if (!decompose_to_tuple_) {
    Shape new_all_to_all_shape;
    new_all_to_all_shape.set_element_type(
        instruction->operand(0)->shape().element_type());
    for (int64_t i = 0; i < instruction->shape().rank(); ++i) {
      if (i != split_dim) {
        new_all_to_all_shape.add_dimensions(all_to_all->shape().dimensions(i));
        continue;
      }
      new_all_to_all_shape.add_dimensions(all_to_all_group_size);
      new_all_to_all_shape.add_dimensions(split_size);
      for (int64_t j = all_to_all->shape().rank() + 1; j < min_array_rank_;
           ++j) {
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
  DimensionVector slice_starts(all_to_all->shape().rank(), 0);
  DimensionVector slice_strides(all_to_all->shape().rank(), 1);
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
  Shape all_to_all_shape = ShapeUtil::MakeTupleShape(
      std::vector<Shape>(all_to_all_group_size, slice_shape));
  HloInstruction* new_all_to_all =
      all_to_all->parent()->AddInstruction(HloInstruction::CreateAllToAll(
          all_to_all_shape, slices, all_to_all->replica_groups(), false,
          all_to_all->channel_id(), absl::nullopt));
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
