/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/all_reduce_key.h"

#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

// Returns a key that will be equal for all-reduce instructions that are
// compatible with each other, and hence might be combined, or different if not.
absl::optional<AllReduceKey> GetAllReduceKey(const HloInstruction* instruction,
                                             const HloDomainMap* domain_map) {
  if (instruction->opcode() != HloOpcode::kAllReduce &&
      instruction->opcode() != HloOpcode::kReduceScatter) {
    return absl::nullopt;
  }

  if (instruction->to_apply()->instruction_count() != 3 ||
      instruction->to_apply()->num_parameters() != 2) {
    VLOG(1) << "Skipping due to non-trivial reduction function.";
    return absl::nullopt;
  }

  const auto* ar = Cast<HloAllReduceInstructionBase>(instruction);

  std::vector<std::vector<int64_t>> replica_groups;
  replica_groups.reserve(ar->replica_groups().size());
  for (const ReplicaGroup& replica_group : ar->replica_groups()) {
    replica_groups.push_back(
        std::vector<int64_t>(replica_group.replica_ids().begin(),
                             replica_group.replica_ids().end()));
  }

  const HloInstruction* to_apply_root = ar->to_apply()->root_instruction();
  // Domain metadata id returned by `GetDomainMetadataId` is guaranteed to be >=
  // 0, so use -1 when we don't need to track domain metadata id.
  int64 domain_metadata_id =
      domain_map ? domain_map->GetDomainMetadataId(ar) : -1;
  return AllReduceKey{
      to_apply_root->opcode(),     to_apply_root->shape().element_type(),
      domain_metadata_id,          ar->channel_id().has_value(),
      ar->use_global_device_ids(), replica_groups};
}

}  // namespace xla
