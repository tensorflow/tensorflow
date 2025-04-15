/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/service/collective_permute_key.h"

#include <optional>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Returns a key that will be equal for collective-permute instructions that are
// compatible with each other, and hence might be combined, or different if not.
std::optional<CollectivePermuteKey> GetCollectivePermuteKey(
    const HloInstruction* instruction) {
  if (instruction->opcode() != HloOpcode::kCollectivePermute) {
    return std::nullopt;
  }

  const auto* cp = Cast<HloCollectivePermuteInstruction>(instruction);
  return CollectivePermuteKey{cp->source_target_pairs()};
}

}  // namespace xla
