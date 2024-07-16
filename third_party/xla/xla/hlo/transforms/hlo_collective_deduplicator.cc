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

#include "xla/hlo/transforms/hlo_collective_deduplicator.h"

#include <cstdint>
#include <set>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"

namespace xla {

absl::StatusOr<bool> HloCollectiveDeduplicator::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  int64_t unique_channel_id = hlo_query::NextChannelId(*module);
  std::set<int64_t> seen_ids;
  for (HloComputation* computation : module->computations(execution_threads)) {
    for (HloInstruction* instruction :
         computation->MakeInstructionPostOrder()) {
      HloInstruction* collective = IsOrHasCollectiveWithChannelId(instruction);
      // Skip send/recv for now, as these allow collisions under certain
      // circumstances. We should keep track of these in the future, but
      // priority in the interim is to unblock AllReduce/AllGather ops.
      if (DynCast<HloSendRecvInstruction>(instruction)) continue;
      if (collective) {
        int64_t channel_id = collective->channel_id().value();
        if (seen_ids.find(channel_id) != seen_ids.end()) {
          collective->set_channel_id(unique_channel_id++);
          changed = true;
        } else {
          seen_ids.emplace(channel_id);
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
