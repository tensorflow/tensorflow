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

#include "xla/service/collective_conflict_analysis.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/xla_data.pb.h"

namespace xla {

void AbstractReplicaGroups::merge_groups(int64_t replica_id,
                                         int64_t other_replica_id) {
  if (get_index(replica_id) == -1 && get_index(other_replica_id) == -1) {
    set_index(replica_id, groups.size());
    set_index(other_replica_id, groups.size());
    groups.push_back({replica_id, other_replica_id});
    return;
  }
  if (get_index(replica_id) == get_index(other_replica_id)) return;
  if (get_index(replica_id) == -1) {
    std::swap(replica_id, other_replica_id);
  }
  CHECK_NE(get_index(replica_id), -1);
  if (get_index(other_replica_id) == -1) {
    set_index(other_replica_id, get_index(replica_id));
    groups[get_index(replica_id)].insert(other_replica_id);
    return;
  }
  CHECK(get_index(replica_id) != -1 && get_index(other_replica_id) != -1 &&
        get_index(replica_id) != get_index(other_replica_id));
  auto& other_set = groups[get_index(other_replica_id)];
  for (int64_t replica_id_in_other_set : other_set) {
    groups[get_index(replica_id)].insert(replica_id_in_other_set);
    set_index(replica_id_in_other_set, get_index(replica_id));
  }
  other_set.clear();
}

bool IsConflictingAbstractReplicaGroups(AbstractReplicaGroups& lhs,
                                        AbstractReplicaGroups& rhs) {
  std::vector<int64_t> frequency(lhs.groups.size(), 0);
  for (auto& rhs_group : rhs.groups) {
    std::fill(frequency.begin(), frequency.end(), 0);
    for (int64_t rhs_replica_id : rhs_group) {
      int64_t i = lhs.get_index(rhs_replica_id);
      if (i == -1) continue;
      if (++frequency[i] >= 2) return true;
    }
  }
  return false;
}

void GetAbstractReplicaGroups(HloInstruction* instr,
                              AbstractReplicaGroups& groups) {
  // Abstract from source-target pairs of collective-permute to abstract replica
  // groups.
  if (instr->opcode() == HloOpcode::kCollectivePermute) {
    auto* cp = Cast<HloCollectivePermuteInstruction>(instr);
    for (auto& [i, j] : cp->source_target_pairs()) {
      groups.merge_groups(i, j);
    }
    return;
  }

  // Abstract from source-target pairs of send/recv to abstract replica groups.
  auto add_replica_group = [&groups](const ReplicaGroup& replica_group) {
    auto& ids = replica_group.replica_ids();
    if (ids.empty()) return;
    int64_t leader_id = ids[0];
    for (int64_t i = 1; i < ids.size(); ++i) {
      groups.merge_groups(leader_id, ids[i]);
    }
  };
  if (instr->opcode() == HloOpcode::kSend ||
      instr->opcode() == HloOpcode::kRecv) {
    auto* sr = Cast<HloSendRecvInstruction>(instr);
    CHECK(!sr->is_host_transfer());
    std::optional<std::string> source_target_pairs_str =
        sr->frontend_attributes().map().at(kSendRecvSourceTargetPairsAttr);
    CHECK(source_target_pairs_str.has_value());
    absl::StatusOr<std::vector<ReplicaGroup>> source_target_pairs =
        ParseReplicaGroupsOnly(*source_target_pairs_str);
    CHECK(source_target_pairs.ok() && "Expect valid source_target_pairs");
    for (auto& replica_group : *source_target_pairs) {
      add_replica_group(replica_group);
    }
    return;
  }

  // Convert normal replica groups to abstract replica groups.
  for (auto& replica_group : GetCollectiveReplicaGroups(instr)) {
    add_replica_group(replica_group);
  }
}

std::vector<HloInstruction*> FindAllConflictingCollectives(
    const HloComputation* computation,
    const std::vector<HloInstruction*>& seed_collectives) {
  absl::flat_hash_set<HloInstruction*> seen;

  // Get the supremum of all abstract replica groups of the seed collectives
  // we're starting with.
  AbstractReplicaGroups abstract_replica_groups_supremum;
  for (HloInstruction* instr : seed_collectives) {
    GetAbstractReplicaGroups(instr, abstract_replica_groups_supremum);
    seen.insert(instr);
  }

  // Try finding more and more conflicting collectives until we reach a
  // fixpoint. This is needed because we may get a coarser supremum with each
  // new conflicting collective.
  std::vector<HloInstruction*> conflicing_collectives;
  bool fixpoint_reached;
  do {
    fixpoint_reached = true;

    // Look at each collective in the computation.
    for (HloInstruction* instr : computation->MakeInstructionPostOrder()) {
      // Skip if not a collective or already considered for the supremum.
      if (!IsNonFusionCollective(instr) || seen.contains(instr)) continue;

      // Check if this collective is already conflicting with the coarsest
      // abstract replica groups. If it does, add to the conflicting collectives
      // and update the supremum.
      AbstractReplicaGroups groups;
      GetAbstractReplicaGroups(instr, groups);
      if (IsConflictingAbstractReplicaGroups(
              groups, abstract_replica_groups_supremum)) {
        conflicing_collectives.push_back(instr);
        GetAbstractReplicaGroups(instr, abstract_replica_groups_supremum);
        seen.insert(instr);
        fixpoint_reached = false;
      }
    }
  } while (!fixpoint_reached);

  return conflicing_collectives;
}

}  // namespace xla
