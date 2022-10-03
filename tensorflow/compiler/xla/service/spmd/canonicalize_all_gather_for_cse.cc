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

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"

namespace xla {

StatusOr<bool> CanonicalizeAllGatherForCSE::RunOnComputation(
    HloComputation* comp) {
  bool changed = false;
  // Helper to find the respective shape input dimension of an shape output
  // dimension of a reshape.
  std::vector<HloInstruction*> ordered_hlos = comp->MakeInstructionPostOrder();
  for (HloInstruction* hlo : ordered_hlos) {
    HloAllGatherInstruction* ag = DynCast<HloAllGatherInstruction>(hlo);

    // TODO(cjfj): Support all-gathers with more than one operand.
    if (!ag || ag->operand_count() > 1) {
      continue;
    }

    // Also only do this for degenerate dimension sizes as the additional
    // reshaping may not be worth the potential for CSE.
    HloInstruction* real_data = ag->mutable_operand(0);
    while (real_data->ReshapeMerelyInsertsOrDeletes1SizedDimensions()
               .has_value()) {
      real_data = real_data->mutable_operand(0);
    }

    if (real_data == ag->operand(0)) {
      continue;
    }

    const int64_t ag_dim = ag->all_gather_dimension();
    int64_t new_ag_dim;
    if (auto dims = ShapeUtil::ReshapeLeavesDimensionsUnmodified(
            ag->operand(0)->shape(), real_data->shape(), {ag_dim})) {
      new_ag_dim = dims->at(0);
    } else {
      int64_t major_elements =
          Product(absl::MakeConstSpan(ag->operand(0)->shape().dimensions())
                      .subspan(0, ag_dim));
      new_ag_dim = 0;
      while (major_elements > 1) {
        major_elements /= real_data->shape().dimensions(new_ag_dim++);
      }
    }
    if (new_ag_dim == real_data->shape().rank()) {
      continue;
    }

    const int64_t all_gather_participants =
        ShapeUtil::ElementsIn(ag->shape()) /
        ShapeUtil::ElementsIn(ag->operand(0)->shape());
    Shape new_ag_shape = real_data->shape();
    new_ag_shape.set_dimensions(
        new_ag_dim,
        all_gather_participants * new_ag_shape.dimensions(new_ag_dim));
    std::optional<int64_t> new_channel_id =
        ag->channel_id() ? std::make_optional(this->NextChannelId())
                         : std::nullopt;
    HloInstruction* new_ag =
        comp->AddInstruction(HloInstruction::CreateAllGather(
            new_ag_shape, {real_data}, /*all_gather_dimension=*/new_ag_dim,
            ag->replica_groups(), ag->constrain_layout(), new_channel_id,
            ag->use_global_device_ids()));
    ag->SetupDerivedInstruction(new_ag);
    HloInstruction* new_formatting = comp->AddInstruction(
        HloInstruction::CreateReshape(ag->shape(), new_ag));
    TF_RETURN_IF_ERROR(comp->ReplaceInstruction(ag, new_formatting));
    changed = true;
  }
  return changed;
}

StatusOr<bool> CanonicalizeAllGatherForCSE::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  next_channel_id_ = hlo_query::NextChannelId(*module);
  for (HloComputation* comp : module->computations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool comp_changed, RunOnComputation(comp));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla
