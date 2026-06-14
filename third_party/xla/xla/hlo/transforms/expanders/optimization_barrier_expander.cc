/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/hlo/transforms/expanders/optimization_barrier_expander.h"

#include <iterator>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"

namespace xla {

absl::StatusOr<bool> OptimizationBarrierExpander::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  std::vector<HloInstruction*> barriers;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    bool modified = false;
    for (HloInstruction* inst : computation->instructions()) {
      if (inst->opcode() != HloOpcode::kOptimizationBarrier) {
        continue;
      }
      if (only_remove_singleton_opt_barriers_ && inst->operand_count() == 1 &&
          inst->operand(0)->opcode() == HloOpcode::kTuple &&
          inst->operand(0)->operand_count() > 1) {
        continue;
      }
      barriers.push_back(inst);
      modified = true;
    }

    if (modified && module->has_schedule()) {
      const auto& sequences = module->schedule().sequences();
      auto it = sequences.find(computation->unique_id());
      if (it != sequences.end()) {
        std::vector<HloInstruction*> sequence;
        sequence.reserve(it->second.instructions().size());
        absl::c_copy_if(it->second.instructions(), std::back_inserter(sequence),
                        [](HloInstruction* inst) {
                          return inst->opcode() !=
                                 HloOpcode::kOptimizationBarrier;
                        });
        module->schedule().set_sequence(computation, sequence);
      }
    }
  }

  for (HloInstruction* inst : barriers) {
    HloInstruction* arg = inst->mutable_operand(0);
    RETURN_IF_ERROR(arg->CopyAllControlDepsFrom(inst));

    RETURN_IF_ERROR(inst->ReplaceAllUsesWith(arg));
    RETURN_IF_ERROR(inst->DropAllControlDeps());

    RETURN_IF_ERROR(inst->parent()->RemoveInstruction(inst));
  }

  return !barriers.empty();
}

}  // namespace xla
