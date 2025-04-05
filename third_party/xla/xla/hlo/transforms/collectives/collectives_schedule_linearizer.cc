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

#include "xla/hlo/transforms/collectives/collectives_schedule_linearizer.h"

#include <memory>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "tsl/platform/errors.h"

namespace xla {

// TODO(b/181653482): Fix for interprocedural collectives as well.
absl::StatusOr<bool> CollectivesScheduleLinearizer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (is_enabled_ && !is_enabled_(module)) {
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    std::unique_ptr<HloReachabilityMap> reachability;
    HloInstruction* prev_done = nullptr;
    auto post_order = computation->MakeInstructionPostOrder();
    for (HloInstruction* inst : post_order) {
      auto* next = DynCast<HloCollectiveInstruction>(inst);
      if (!next) {
        continue;
      }
      // Build reachability map on demand if we actually see collectives.
      if (!reachability) {
        reachability = HloReachabilityMap::Build(computation, post_order);
      }
      // Derive the 'start' and 'done' peers of this instruction. For non-async
      // variants of collectives, they are the same as this instruction. For
      // async variants, the start is this instruction and the 'done' is the
      // matching async-done instruction.
      HloInstruction* start = next;
      HloInstruction* done = next;
      switch (next->opcode()) {
        case HloOpcode::kAllReduceStart:
        case HloOpcode::kAllGatherStart:
        case HloOpcode::kCollectivePermuteStart:
        case HloOpcode::kAsyncStart:
          // Find the async-done corresponding to this async start instruction.
          CHECK_EQ(start->user_count(), 1);
          done = start->users()[0];
          break;
        default:
          break;
      }

      if (prev_done && !reachability->IsConnected(start, prev_done)) {
        // If prev_done and start are independent, enforce ordering.
        TF_RETURN_IF_ERROR(prev_done->AddControlDependencyTo(next));
        // Adding control dependency does not update the reachability map.
        reachability->UpdateReachabilityThroughInstruction(start);

        VLOG(1) << "Adding control dependency from " << prev_done->ToString()
                << " to " << start->ToString();
        changed = true;
      }
      prev_done = done;
    }
  }
  return changed;
}

}  // namespace xla
