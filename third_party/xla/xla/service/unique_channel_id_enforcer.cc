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

#include "xla/service/unique_channel_id_enforcer.h"

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/utils/hlo_query.h"

namespace xla {

absl::StatusOr<bool> UniqueChannelIdEnforcer::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  absl::flat_hash_set<std::optional<int64_t>> used_channel_ids;
  auto next_channel_id = hlo_query::NextChannelId(*module);
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      if (!hlo_query::IsCollectiveCommunicationOp(instruction->opcode()))
        continue;
      auto channel_id = instruction->channel_id();
      if (used_channel_ids.contains(channel_id)) {
        if (assert_unique_channel_ids_) {
          LOG(ERROR) << "Duplicate channel ID " << channel_id.value_or(-1)
                     << " found on instruction: " << instruction->ToString();
          return absl::InternalError(absl::StrFormat(
              "Duplicate channel ID %d found on instruction: %s",
              channel_id.value_or(-1), instruction->ToString()));
        }
        instruction->set_channel_id(next_channel_id);
        used_channel_ids.insert(next_channel_id);
        next_channel_id++;
        changed = true;
      } else {
        used_channel_ids.insert(channel_id);
      }
    }
  }

  return changed;
}

}  // namespace xla
