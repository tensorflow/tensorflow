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

#include "xla/service/nullary_function_wrap_inliner.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/call_inliner.h"

namespace xla {

absl::StatusOr<bool> NullaryFunctionWrapInliner::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation : module->computations(execution_threads)) {
    std::vector<HloInstruction*> calls_to_inline;
    for (HloInstruction* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCall &&
          instruction->operand_count() == 1 &&
          instruction->operand(0)->shape().IsToken()) {
        calls_to_inline.push_back(instruction);
      }
    }
    for (HloInstruction* call : calls_to_inline) {
      HloInstruction* token_op = call->mutable_operand(0);
      ABSL_ASSIGN_OR_RETURN(CallInliner::InlinedInstructionMap inlined_map,
                       CallInliner::Inline(call));
      for (const auto& [orig_inst, inlined_inst] : inlined_map) {
        if (orig_inst->opcode() != HloOpcode::kParameter) {
          ABSL_RETURN_IF_ERROR(token_op->AddControlDependencyTo(inlined_inst));
        }
      }
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
