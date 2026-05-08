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

#include "xla/hlo/transforms/collectives/async_collective_replacer.h"

#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"
#include "xla/tsl/platform/errors.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

bool ShouldBeReplaced(const AsyncCollectiveReplacer::Config& config,
                      const HloInstruction* instruction) {
  if (instruction->opcode() == HloOpcode::kAllReduceStart) {
    return config.convert_all_reduce(instruction);
  }
  if (instruction->opcode() == HloOpcode::kAllGatherStart) {
    return config.convert_all_gather(instruction);
  }
  if (instruction->opcode() == HloOpcode::kCollectivePermuteStart) {
    return config.convert_collective_permute(instruction);
  }
  if (instruction->opcode() == HloOpcode::kAsyncStart &&
      instruction->async_wrapped_opcode() == HloOpcode::kCollectiveBroadcast) {
    return config.convert_collective_broadcast(instruction);
  }
  if (instruction->opcode() == HloOpcode::kAsyncStart &&
      instruction->async_wrapped_opcode() == HloOpcode::kAllToAll) {
    return config.convert_all_to_all(instruction);
  }
  if (instruction->opcode() == HloOpcode::kAsyncStart &&
      instruction->async_wrapped_opcode() == HloOpcode::kReduceScatter) {
    return config.convert_reduce_scatter(instruction);
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> AsyncCollectiveReplacer::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    // Gather the list of all async pairs that need to be replaced.
    std::vector<std::pair<HloInstruction*, HloInstruction*>> async_pairs;
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (!ShouldBeReplaced(config_, inst)) {
        continue;
      }
      if (inst->users().size() != 1) {
        return Internal(
            "Expected exactly one user for async start op %s, but found %d",
            inst->name(), inst->users().size());
      }
      HloInstruction* done = inst->users()[0];
      VLOG(1) << "Replacing async start/done ops with synchronous counterpart";
      VLOG(1) << "async start = " << inst->ToString();
      VLOG(1) << "async done = " << done->ToString();
      async_pairs.push_back({inst, done});
      changed = true;
    }

    // Replace them.
    TF_RETURN_IF_ERROR(
        ConvertAsyncCollectivesToSync::ReplaceAsyncInstructionsWithSync(
            computation, async_pairs));
  }
  return changed;
}

}  // namespace xla
