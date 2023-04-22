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

#include "tensorflow/compiler/xla/service/spmd/canonicalize_all_gather_for_cse.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"

namespace xla {

namespace {

// Returns if an instructions adds only degenerate dimensions to the shape of
// the input, like going from [X,Y] to [1,X,Y,1].
bool IsAddingOnlyDegenerateDimensions(const HloInstruction* inst) {
  if (inst->opcode() != HloOpcode::kBitcast &&
      inst->opcode() != HloOpcode::kReshape) {
    return false;
  }
  const Shape& in_shape = inst->operand(0)->shape();
  const Shape& out_shape = inst->shape();
  return ShapeUtil::ElementsIn(in_shape) == ShapeUtil::ElementsIn(out_shape) &&
         ShapeUtil::DimensionsUnmodifiedByReshape(in_shape, out_shape).size() ==
             in_shape.rank();
}

}  // namespace

StatusOr<bool> CanonicalizeAllGatherForCSE::RunOnComputation(
    HloComputation* comp) {
  bool changed = false;
  // Helper to find the respective shape input dimension of an shape output
  // dimension of a reshape.
  std::vector<HloInstruction*> ordered_hlos = comp->MakeInstructionPostOrder();
  for (HloInstruction* hlo : ordered_hlos) {
    HloAllGatherInstruction* ag = DynCast<HloAllGatherInstruction>(hlo);
    // Only supporting AllGather on dimension 0 as it's the only case currently
    // happening and additional cases needs more complexity.
    // TODO(cjfj): Support all-gathers with more than one operand.
    if (!ag || ag->all_gather_dimension() != 0 || ag->operand_count() > 1) {
      continue;
    }
    HloInstruction* real_data = ag->mutable_operand(0);
    const int64 ag_dim = ag->all_gather_dimension();
    const Shape& out_shape = ag->shape();
    const Shape& in_shape = ag->operand(0)->shape();
    CHECK_EQ(out_shape.dimensions(ag_dim) % in_shape.dimensions(ag_dim), 0);
    const int64 all_gather_participants =
        out_shape.dimensions(ag_dim) / in_shape.dimensions(ag_dim);
    // Look through bitcast/bitcast-like reshapes, keeping track of the position
    // of the all-gather dimension through the reshapes (should stay 0 or become
    // -1 if the dimension has been added from a reshape we have passed through)
    while (IsAddingOnlyDegenerateDimensions(real_data)) {
      real_data = real_data->mutable_operand(0);
    }
    // If we looked through some reshapes and there's more than just one reshape
    // adding the dimension the all-gather is operating on then perform the
    // canonicalization.
    if (real_data != ag->operand(0)) {
      std::vector<int64> new_dimensions;
      new_dimensions.reserve(real_data->shape().dimensions_size() + 1);
      new_dimensions.push_back(1);
      new_dimensions.insert(new_dimensions.end(),
                            real_data->shape().dimensions().begin(),
                            real_data->shape().dimensions().end());
      // Adding specialized all-gather dimension.
      HloInstruction* ag_input =
          comp->AddInstruction(HloInstruction::CreateReshape(
              ShapeUtil::MakeShape(real_data->shape().element_type(),
                                   new_dimensions),
              real_data));
      new_dimensions[0] = all_gather_participants;
      absl::optional<int64> new_channel_id =
          ag->channel_id() ? absl::make_optional(this->NextChannelId())
                           : absl::nullopt;
      HloInstruction* new_ag =
          comp->AddInstruction(HloInstruction::CreateAllGather(
              ShapeUtil::MakeShape(real_data->shape().element_type(),
                                   new_dimensions),
              {ag_input}, /*all_gather_dimension=*/0, ag->replica_groups(),
              ag->constrain_layout(), new_channel_id,
              ag->use_global_device_ids()));
      HloInstruction* new_formatting = comp->AddInstruction(
          HloInstruction::CreateReshape(ag->shape(), new_ag));
      TF_RETURN_IF_ERROR(ag->ReplaceAllUsesWith(new_formatting));
      TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(ag));
      changed = true;
    }
  }
  return changed;
}

StatusOr<bool> CanonicalizeAllGatherForCSE::Run(HloModule* module) {
  bool changed = false;
  next_channel_id_ = hlo_query::NextChannelId(*module);
  for (HloComputation* comp : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
