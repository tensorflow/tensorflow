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

#include "xla/service/control_dep_rewriter.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> ControlDepRewriter::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    std::vector<HloInstruction*> to_remove;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCustomCall &&
          instruction->custom_call_target() == "control_dep") {
        changed = true;
        CHECK_EQ(instruction->operand_count(), 2);
        HloInstruction* src = instruction->mutable_operand(0);
        HloInstruction* dst = instruction->mutable_operand(1);
        RETURN_IF_ERROR(src->AddControlDependencyTo(dst));
        to_remove.push_back(instruction);
      }
    }
    for (HloInstruction* instruction : to_remove) {
      RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
    }
  }
  return changed;
}

}  // namespace xla
