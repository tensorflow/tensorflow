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

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
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
  absl::flat_hash_map<HloInstruction*, HloComputation*> multi_user_broadcasts;
  absl::flat_hash_map<HloInstruction*, HloComputation*> multi_usage_broadcasts;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      if (instruction->opcode() == HloOpcode::kBroadcast) {
        if (instruction->user_count() > 1) {
          multi_user_broadcasts[instruction] = computation;
        } else if (instruction->user_count() == 1) {
          HloInstruction* user = instruction->users()[0];
          if (absl::c_count(user->operands(), instruction) > 1) {
            multi_usage_broadcasts[instruction] = computation;
          }
        }
      }
    }
  }

  for (auto& [broadcast, computation] : multi_user_broadcasts) {
    for (HloInstruction* user : broadcast->users()) {
      HloInstruction* cloned_broadcast =
          user->AddInstruction(broadcast->Clone());
      TF_RETURN_IF_ERROR(broadcast->ReplaceUseWith(user, cloned_broadcast));
      changed = true;
    }
  }

  for (auto& [broadcast, computation] : multi_usage_broadcasts) {
    HloInstruction* user = broadcast->users()[0];
    for (int64_t i = 0; i != user->operand_count(); ++i) {
      const HloInstruction* operand = user->operand(i);
      if (operand == broadcast) {
        HloInstruction* cloned_broadcast =
            user->AddInstruction(broadcast->Clone());
        TF_RETURN_IF_ERROR(
            broadcast->ReplaceUseWith(user, i, cloned_broadcast));
        changed = true;
      }
    }
  }

  return changed;
}

}  // namespace xla
