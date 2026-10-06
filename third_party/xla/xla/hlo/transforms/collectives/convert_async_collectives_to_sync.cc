/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/convert_async_collectives_to_sync.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/scheduling_annotations_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla {

absl::StatusOr<bool> ConvertAsyncCollectivesToSync::RunOnComputation(
    HloComputation* computation) {
  HloModule* module = computation->parent();
  std::vector<std::pair<HloInstruction*, HloInstruction*>> async_pairs;

  const HloInstructionSequence& sequence =
      module->schedule().sequence(computation);

  // Set of async-start ops that are currently in flight, i.e., their done not
  // yet seen.
  absl::flat_hash_set<HloInstruction*> in_flight_ops;

  for (HloInstruction* instruction : sequence.instructions()) {
    if (hlo_query::IsAsyncCollectiveStartOp(instruction)) {
      in_flight_ops.insert(instruction);
      VLOG(3) << "Found async start " << instruction->ToString();
    } else if (hlo_query::IsAsyncCollectiveDoneOp(instruction)) {
      // If this done is matching with the previous start and all intervening
      // ops are nops (i.e., prev_async_start was not reset to null), then we
      // were unable to schedule an independent op to overlap with this async
      // collective, so convert it to sync.
      VLOG(3) << "Found async done " << instruction->ToString();

      // All async-done ops are unary ops.
      TF_RET_CHECK(instruction->operand_count() == 1);
      HloInstruction* matching_async_start =
          hlo_instruction_utils::async::FindAsyncStart(
              instruction->mutable_operand(0));

      // Find if corresponding async-start is in the set of in-flight ops and
      // erase it (since it cannot be paired with any other async-done).
      if (matching_async_start != nullptr &&
          in_flight_ops.erase(matching_async_start) == 1) {
        async_pairs.push_back({matching_async_start, instruction});
        VLOG(3) << "Added pair: {" << matching_async_start->name() << ", "
                << instruction->name();
      }
    } else if (!in_flight_ops.empty() && (!is_nop_ || !is_nop_(instruction))) {
      VLOG(3) << "Found intervening non-NOP instruction "
              << instruction->ToString();
      in_flight_ops.clear();
    }
  }

  if (async_pairs.empty()) {
    return false;
  }

  for (auto& [async_start, async_done] : async_pairs) {
    ABSL_ASSIGN_OR_RETURN(std::optional<int64_t> group_id,
                     GetSchedulingAnnotationGroupId(async_done));
    if (group_id) {
      LOG(WARNING) << "Async collective pair (" << async_start->name() << ", "
                   << async_done->name() << ") with scheduling group id "
                   << *group_id
                   << " is not overlapped after scheduling and is converted "
                      "to a synchronous collective.";
    }
  }

  ABSL_RETURN_IF_ERROR(ConvertAsyncInstructionsToSync(computation, async_pairs));
  return true;
}

absl::StatusOr<bool> ConvertAsyncCollectivesToSync::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (!module->has_schedule()) {
    VLOG(3) << "Skipping as module is not scheduled";
    return false;
  }
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (!module->schedule().is_computation_scheduled(computation)) {
      VLOG(3) << "Skipping computation" << computation->name()
              << " as it is not scheduled";
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(bool computation_changed, RunOnComputation(computation));
    changed |= computation_changed;
  }
  return changed;
}

}  // namespace xla
