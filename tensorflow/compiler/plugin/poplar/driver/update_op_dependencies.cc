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

#include "tensorflow/compiler/plugin/poplar/driver/update_op_dependencies.h"
#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/driver/inplace_instructions.h"

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/bcast.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> UpdateOpDependenctOrdering::Run(HloModule* module) {
  auto entry_computation = module->entry_computation();
  std::unique_ptr<HloReachabilityMap> reachability_map =
      entry_computation->ComputeReachability();

  bool changed = false;

  for (const auto priority :
       annotations_.inplace_instructions.GetPriorityOrder()) {
    std::vector<const HloInstruction*> to_remove;
    std::vector<const HloInstruction*> to_inplace;
    for (auto* inst :
         annotations_.inplace_instructions.GetPrioritySet(priority)) {
      // We only currently support inplace ops inside the entry computation
      if (inst->parent() != module->entry_computation()) {
        to_remove.push_back(inst);
        continue;
      }
      // We do not allow parameter variables (non-streaming) which are not in
      // High priority to be in-place, otherwise we would have to reload data
      // which would perform poorly during inference
      if (inst->operand(0)->opcode() == HloOpcode::kParameter) {
        // Work out whether this parameter is streamed
        auto num_streaming =
            inst->parent()->num_parameters() - annotations_.num_resource_inputs;
        if (inst->operand(0)->parameter_number() >= num_streaming &&
            priority != InplaceInstructions::Priority::HIGH) {
          to_remove.push_back(inst);
          continue;
        }
      }

      bool add_to_inplace = true;
      std::vector<HloInstruction*> dependencies;
      HloInstruction* laundered_inst;
      TF_ASSIGN_OR_RETURN(laundered_inst,
                          module->LaunderConstInstructionFromModule(inst));
      for (auto* peer : inst->operand(0)->users()) {
        if (peer == inst) {
          continue;
        }

        // If peer is a depenency of inst, this can't be inplace
        if (reachability_map->IsReachable(inst, peer)) {
          add_to_inplace = false;
          break;
        } else {
          // Add ctrl dep, and remove peer from list
          HloInstruction* from;
          TF_ASSIGN_OR_RETURN(from,
                              module->LaunderConstInstructionFromModule(peer));

          from->AddControlDependencyTo(laundered_inst);
          dependencies.push_back(from);
          entry_computation->UpdateReachabilityThroughInstruction(
              inst, reachability_map.get());
        }
      }

      if (add_to_inplace) {
        to_inplace.push_back(inst);
      } else {
        // otherwise remove it and undo any dependencies added
        to_remove.push_back(inst);

        for (auto* depenency : dependencies) {
          depenency->RemoveControlDependencyTo(laundered_inst);
        }
        entry_computation->UpdateReachabilityThroughInstruction(
            inst, reachability_map.get());
      }

      changed |= add_to_inplace;
    }

    // move instructions which are to be in place to the top priority set
    for (auto* inst : to_inplace) {
      annotations_.inplace_instructions.MovePriority(
          priority, InplaceInstructions::Priority::HIGH, inst);
    }

    // remove instructions which are not actually in place anymore
    for (auto* inst : to_remove) {
      annotations_.inplace_instructions.RemoveFrom(priority, inst);
    }
  }

  return changed;
}  // namespace poplarplugin

}  // namespace poplarplugin
}  // namespace xla
