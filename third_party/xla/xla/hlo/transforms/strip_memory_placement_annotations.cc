/* Copyright 2026 The OpenXLA Authors.

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
#include "xla/hlo/transforms/strip_memory_placement_annotations.h"

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> StripMemoryPlacementAnnotations::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          (instruction->custom_call_target() == "annotate_device_placement" ||
           instruction->custom_call_target() == "MoveToHost" ||
           instruction->custom_call_target() == "MoveToDevice")) {
        // Drop control dependencies if any exist (not expected but safe).
        TF_RETURN_IF_ERROR(instruction->DropAllControlDeps());
        // Ensure the instruction has exactly one operand.
        TF_RET_CHECK(instruction->operand_count() == 1)
            << "Memory placement custom call should have exactly one operand: "
            << instruction->ToString();
        HloInstruction* operand = instruction->mutable_operand(0);
        if (instruction->has_sharding() && !operand->has_sharding()) {
          operand->set_sharding(instruction->sharding());
        }
        TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(operand));
        // If the instruction is the root, update the computation's root.
        if (computation->root_instruction() == instruction) {
          computation->set_root_instruction(operand);
        }
        // Remove the custom call instruction.
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        changed = true;
      }
    }
  }
  return changed;
}

}  // namespace xla
