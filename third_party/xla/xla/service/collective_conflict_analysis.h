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

#ifndef XLA_SERVICE_COLLECTIVE_CONFLICT_ANALYSIS_H_
#define XLA_SERVICE_COLLECTIVE_CONFLICT_ANALYSIS_H_

#include <cstdint>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {

struct AbstractReplicaGroups {
  // Holds groups of abstract replica ids.
  std::vector<absl::flat_hash_set<int64_t>> groups;

  // Maps abstract replica id to index in groups.
  std::vector<int64_t> index_map;

  int64_t get_index(int64_t replica_id) {
    while (index_map.size() <= replica_id) index_map.push_back(-1);
    return index_map[replica_id];
  }

  void set_index(int64_t replica_id, int64_t index) {
    while (index_map.size() <= replica_id) index_map.push_back(-1);
    index_map[replica_id] = index;
  }

  void merge_groups(int64_t replica_id, int64_t other_replica_id);
};

bool IsConflictingAbstractReplicaGroups(AbstractReplicaGroups& lhs,
                                        AbstractReplicaGroups& rhs);

void GetAbstractReplicaGroups(HloInstruction* instr,
                              AbstractReplicaGroups& groups);

std::vector<HloInstruction*> FindAllConflictingCollectives(
    const HloComputation* computation,
    const std::vector<HloInstruction*>& seed_collectives);

inline std::vector<HloInstruction*> FindAllConflictingCollectives(
    HloInstruction* seed_collective) {
  return FindAllConflictingCollectives(seed_collective->parent(),
                                       {seed_collective});
}

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_CONFLICT_ANALYSIS_H_
