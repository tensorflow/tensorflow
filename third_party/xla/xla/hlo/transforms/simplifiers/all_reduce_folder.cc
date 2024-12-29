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

#include "xla/hlo/transforms/simplifiers/all_reduce_folder.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/all_reduce_key.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace {
// Folds the given two sets of non-empty replica groups into a single set if
// possible.
std::optional<std::vector<ReplicaGroup>> FoldReplicaGroups(
    absl::Span<const ReplicaGroup> replica_groups0,
    absl::Span<const ReplicaGroup> replica_groups1) {
  // For a valid all-reduce with non-empty replica groups, the groups should
  // list each replica exactly once.
  int64_t num_replicas = 0;
  for (const ReplicaGroup &rg : replica_groups0) {
    for (int64_t id : rg.replica_ids()) {
      num_replicas = std::max(num_replicas, id);
    }
  }
  num_replicas++;

  // We will build, for each replica, the effective set of replicas which
  // contribute to the output of that replica by essentially tracing through
  // the 2 sets of replica groups.

  // For each replica, remember its replica group # from replica_group0
  std::vector<int> replica_group_no(num_replicas, -1);
  for (int group_no = 0; group_no < replica_groups0.size(); ++group_no) {
    for (int64_t id : replica_groups0[group_no].replica_ids()) {
      replica_group_no[id] = group_no;
    }
  }

  // For each replica, trace through the 2 replica groups to build the set of
  // contributing replicas for each replica's output. In an all-reduce, each
  // contributor can contribute only once, so if see a contributing replica more
  // than once, such replica groups cannot be folded.

  // Note: Using std::vector<bool> instead of flat_hash_set for contributor sets
  // since flat_hash_set cannot be used as a flat_hash_map key.
  // Map to a contributor set to its unique id.
  absl::flat_hash_map<std::vector<bool>, int64_t> contributor_set_id;

  // Map each replica to the unique id for the set of its contributors.
  std::vector<int64_t> contributing_replicas_set_id(num_replicas, 0);

  int64_t next_id = 1;
  for (const ReplicaGroup &rg : replica_groups1) {
    std::vector<bool> contributors(num_replicas, false);
    for (int64_t id : rg.replica_ids()) {
      int64_t group_no = replica_group_no[id];
      for (int64_t contrib : replica_groups0[group_no].replica_ids()) {
        // If the contributor already preset in the set, fail. As an example
        // rg0 = {0, 1}
        // rg1 = {0, 1}
        // In such a case, when processing id = 1 from rg0, replica #0 will
        // already be present, so the groups cannot be merged.
        if (contributors[contrib]) {
          return std::nullopt;
        }
        contributors[contrib] = true;
      }
    }

    // Uniquefy the contributor sets by assigning a unique id to each unique
    // set.
    int64_t set_id;
    auto it = contributor_set_id.find(contributors);
    if (it != contributor_set_id.end()) {
      set_id = it->second;
    } else {
      set_id = next_id++;
      contributor_set_id[contributors] = set_id;
    }

    // All replica id in the group have the same set of contributors.
    for (int64_t id : rg.replica_ids()) {
      contributing_replicas_set_id[id] = set_id;
    }
  }

  // Now verify, for each unique set of contributors, whether for all of the
  // associated replicas have the same contributors. These unique sets now
  // become the folded replica groups.
  std::vector<ReplicaGroup> new_replica_groups;
  new_replica_groups.reserve(contributor_set_id.size());

  for (const auto &it : contributor_set_id) {
    const std::vector<bool> &contributors = it.first;
    const int64_t set_id = it.second;
    new_replica_groups.emplace_back();
    ReplicaGroup &group = new_replica_groups.back();
    for (int64_t replica = 0; replica < num_replicas; ++replica) {
      if (contributors[replica]) {
        if (contributing_replicas_set_id[replica] != set_id) {
          return std::nullopt;
        }
        group.add_replica_ids(replica);
      }
    }
  }

  // Sort the replica groups by the first id for stable behavior. Otherwise,
  // groups are formed according to the order in the contributor_set_id map,
  // which is not stable.
  absl::c_sort(new_replica_groups,
               [](const ReplicaGroup &a, const ReplicaGroup &b) {
                 return a.replica_ids(0) < b.replica_ids(0);
               });
  return new_replica_groups;
}

}  // namespace

absl::StatusOr<bool> AllReduceFolder::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  if (hlo_query::ContainsLayoutConstrainedAllReduce(*module)) {
    VLOG(1) << "Skip AllReduceFolder because the module contains all-reduce "
               "with constrained layouts";
    return false;
  }

  int64_t next_channel_id = hlo_query::NextChannelId(*module);

  bool changed = false;
  for (auto computation : module->computations(execution_threads)) {
    for (HloInstruction *inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() != HloOpcode::kAllReduce ||
          inst->operand(0)->opcode() != HloOpcode::kAllReduce) {
        continue;
      }

      auto *ar0 = Cast<HloAllReduceInstruction>(inst->mutable_operand(0));
      auto *ar1 = Cast<HloAllReduceInstruction>(inst);

      if (ar0->user_count() != 1) {
        continue;
      }

      // Check if the 2 all-reduce instructions are compatible with the
      // exception of the replica groups.
      std::optional<AllReduceKey> key0 = GetAllReduceKey(
          ar0, /*domain_map=*/nullptr, /*ignore_replica_groups=*/true);
      std::optional<AllReduceKey> key1 = GetAllReduceKey(
          ar1, /*domain_map=*/nullptr, /*ignore_replica_groups=*/true);
      if (!key0 || !key1 || *key0 != *key1 || ar0->replica_groups().empty() ||
          ar1->replica_groups().empty()) {
        continue;
      }

      // Since both all-reduces have non-empty replica groups, they list all the
      // participants. We essentially build, for each participant, which replica
      // contributes to the result of second all-reduce for that participant.
      // For example, for the below sequence:
      //   ar0 = all-reduce(x)   replica_groups={{0,1},{2,3},{4,5},{6,7}}
      //   ar1 = all-reduce(ar0) replica_groups={{0,2},{1,3},{4,6},{5,7}}

      // ar1 output for replica 0 contains { x0, x1, x2, x3}, where x_i is the
      // value of x in replica i.
      // r1 = { x0, x1, x2, x3} as well.
      // After we have these sets, we check if these sets are compatible for
      // forming a new all-reduce.

      std::optional<std::vector<ReplicaGroup>> new_replica_groups =
          FoldReplicaGroups(ar0->replica_groups(), ar1->replica_groups());
      if (!new_replica_groups) {
        continue;
      }
      std::optional<int64_t> channel_id;
      if (ar0->channel_id()) {
        channel_id = next_channel_id++;
      }

      // Create new all-reduce and delete the 2 existing ones.
      HloInstruction *new_ar =
          computation->AddInstruction(HloInstruction::CreateAllReduce(
              ar0->shape(), ar0->operands(), ar0->to_apply(),
              CollectiveDeviceList(*new_replica_groups),
              /*constrain_layout=*/false, channel_id,
              ar0->use_global_device_ids()));
      TF_RETURN_IF_ERROR(ar1->ReplaceAllUsesWith(new_ar));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar1));
      TF_RETURN_IF_ERROR(computation->RemoveInstruction(ar0));
      changed = true;
    }
  }
  return changed;
}

}  // namespace xla
