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

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/util/bcast.h"

namespace xla {
namespace poplarplugin {

StatusOr<bool> UpdateOpDependenctOrdering::Run(HloModule* module) {
  std::vector<const HloInstruction*> to_remove;
  auto entry_computation = module->entry_computation();
  std::unique_ptr<HloReachabilityMap> reachability_map =
      entry_computation->ComputeReachability();

  for (auto* inst : annotations_.inplace_instructions) {
    // We only currently support inplace ops inside the entry computation
    if (inst->parent() != module->entry_computation()) {
      to_remove.push_back(inst);
      continue;
    }

    for (auto* peer : inst->operand(0)->users()) {
      if (peer == inst) {
        continue;
      }

      // If peer is a depenency of inst, remove inst from inplace list
      if (reachability_map->IsReachable(inst, peer)) {
        to_remove.push_back(inst);
        break;
      } else {
        // Add ctrl dep, and remove peer from list
        HloInstruction* from;
        TF_ASSIGN_OR_RETURN(from,
                            module->LaunderConstInstructionFromModule(peer));
        HloInstruction* to;
        TF_ASSIGN_OR_RETURN(to,
                            module->LaunderConstInstructionFromModule(inst));
        from->AddControlDependencyTo(to);
        entry_computation->UpdateReachabilityThroughInstruction(
            inst, reachability_map.get());
      }
    }
  }

  // remove instructions which are not actually in place anymore
  for (auto* inst : to_remove) {
    annotations_.inplace_instructions.erase(inst);
  }

  return true;
}  // namespace poplarplugin

}  // namespace poplarplugin
}  // namespace xla
