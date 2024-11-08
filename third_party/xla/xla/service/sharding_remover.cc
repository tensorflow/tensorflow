/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/service/sharding_remover.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/spmd/shardy/constants.h"
#include "tsl/platform/errors.h"

namespace xla {

// Remove Sharding custom-call instruction by assigning its users to
// to its operand.
absl::StatusOr<bool> ShardingRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  const absl::flat_hash_set<absl::string_view> to_remove_sharding_ops = {
      "Sharding", "SPMDShardToFullShape", "SPMDFullToShardShape",
      sdy::kShardingGroupCustomCallTargetName,
      sdy::kFuncResultShardingTargetName};

  for (HloComputation* computation : module->computations(execution_threads)) {
    auto instructions = computation->MakeInstructionPostOrder();
    std::reverse(instructions.begin(), instructions.end());
    for (HloInstruction* instruction : instructions) {
      if (instruction->opcode() != HloOpcode::kCustomCall) {
        continue;
      }

      if (!to_remove_sharding_ops.contains(instruction->custom_call_target())) {
        continue;
      }
      CHECK(instruction->operand_count() == 1)
          << "Sharding instruction must have exactly one operand";

      // ShardingGroupOp is dangling so we just remove it.
      if (instruction->custom_call_target() ==
          sdy::kShardingGroupCustomCallTargetName) {
        TF_RETURN_IF_ERROR(computation->RemoveInstruction(instruction));
        continue;
      }

      TF_RETURN_IF_ERROR(instruction->ReplaceAllUsesWith(
          instruction->mutable_operand(0), name()));
      changed = true;

      // We do not DCE sharding custom-call, so replace sharding custom-call
      // with a copy instead, so that it can be DCE-ed in later passes.
      if (instruction->custom_call_target() == "Sharding" ||
          instruction->custom_call_target() ==
              sdy::kFuncResultShardingTargetName) {
        auto copy = computation->AddInstruction(
            HloInstruction::CreateUnary(instruction->shape(), HloOpcode::kCopy,
                                        instruction->mutable_operand(0)));
        TF_RETURN_IF_ERROR(computation->ReplaceInstruction(instruction, copy));
        instruction = copy;
      }
    }
  }

  return changed;
}

}  // namespace xla
