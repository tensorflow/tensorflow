/* Copyright 2024 The OpenXLA Authors.
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

#include "xla/hlo/transforms/hlo_broadcast_splitter.h"

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/errors.h"

namespace xla {

absl::StatusOr<bool> HloBroadcastSplitter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  HloInstructionSet seen_broadcasts;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      auto operands = instruction->unique_operands();
      for (int64_t i = 0; i < instruction->operand_count(); ++i) {
        HloInstruction* op = instruction->mutable_operand(i);
        if (op->opcode() != HloOpcode::kBroadcast) {
          continue;
        }
        if (seen_broadcasts.insert(op).second) {
          continue;
        }
        HloInstruction* clone = op->parent()->AddInstruction(op->Clone());
        changed = true;
        TF_RETURN_IF_ERROR(instruction->ReplaceOperandWith(i, clone));
      }
    }
  }

  return changed;
}

}  // namespace xla
