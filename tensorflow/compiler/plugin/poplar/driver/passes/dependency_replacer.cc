/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/passes/dependency_replacer.h"
#include "tensorflow/compiler/plugin/poplar/driver/tools/util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {

/*
 * For non-fusion computations, take all add-dependency ops and introduce a
 * control dependency from the user of the add-dependency to the op which is
 * the feed for the after-all.  Then remove the add-dependency.
 *
 * This is repeated until there are no add-dependency ops remaining in the
 * computation.
 *
 * GetAllDeps finds all instructions which are control dependency predecessors
 * of the argument, by following add-dependency and after-all instructions.
 */
StatusOr<bool> DependencyReplacer::Run(HloModule* module) {
  for (auto* comp : module->computations()) {
    if (IsPopOpsFusion(comp)) {
      continue;
    }

    std::vector<HloInstruction*> deps;
    do {
      deps.clear();

      for (HloInstruction* inst : comp->instructions()) {
        if (inst->opcode() == HloOpcode::kAddDependency &&
            inst->operand(0)->opcode() != HloOpcode::kAddDependency) {
          deps.push_back(inst);
        }
      }

      for (auto* inst : deps) {
        std::vector<HloInstruction*> dep_srcs;
        GetAllDeps(inst->operand(1), dep_srcs);

        auto users = inst->users();
        for (auto* user : users) {
          for (auto* dep_src : dep_srcs) {
            if (add_control_deps_) {
              TF_RETURN_IF_ERROR(dep_src->AddControlDependencyTo(user));
              TF_RETURN_IF_ERROR(user->CopyAllControlDepsFrom(inst));
            }
            TF_RETURN_IF_ERROR(inst->DropAllControlDeps());
          }
          TF_RETURN_IF_ERROR(
              inst->ReplaceUseWith(user, inst->mutable_operand(0)));
        }
      }

      for (HloInstruction* inst : deps) {
        TF_RETURN_IF_ERROR(comp->RemoveInstructionAndUnusedOperands(inst));
      }
    } while (deps.size() > 0);
  }
  return true;
}

DependencyReplacer::DependencyReplacer(bool add_control_deps)
    : add_control_deps_(add_control_deps) {}

}  // namespace poplarplugin
}  // namespace xla
