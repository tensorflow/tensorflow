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

#ifndef XLA_SERVICE_COLLECTIVE_OPT_UTILS_H_
#define XLA_SERVICE_COLLECTIVE_OPT_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"

namespace xla {

struct ReduceScatterSpec {
  int64_t split_dim = -1;
  int64_t sharded_partitions = 1;
  int64_t sharded_replicas = 1;
  int64_t group_size;
  std::vector<int64_t> original_split_dims;
  HloInstruction* dynamic_slice;
};

struct SplitDimSpec {
  int64_t split_dim = -1;
  std::vector<int64_t> split_dims = {};
};

// Matches the given all-reduce operation to a reduce-scatter pattern.
std::optional<ReduceScatterSpec> MatchReduceScatter(
    const HloAllReduceInstructionBase* ar, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims = false,
    bool allow_intervening_reshape = false, int64_t min_rank = 1,
    HloPredicate match_partition_id = HloPredicateIsOp<HloOpcode::kPartitionId>,
    HloPredicate match_replica_id = HloPredicateIsOp<HloOpcode::kReplicaId>,
    bool allow_intervening_bitcast = false);

// Check whether AG(ICI) and its user DS(ICI) can be canceled out.
std::optional<ReduceScatterSpec> AllGatherDynamicSliceCancellation(
    const HloAllGatherInstruction* ag, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims = false,
    bool allow_intervening_reshape = false, int64_t min_rank = 1,
    HloPredicate match_partition_id = HloPredicateIsOp<HloOpcode::kPartitionId>,
    HloPredicate match_replica_id = HloPredicateIsOp<HloOpcode::kReplicaId>,
    bool allow_intervening_bitcast = false, bool allow_multiple_users = false);

// Checks whether the replica groups in the given channel instruction are
// of the same size.
bool CheckUniformReplicaGroups(const HloChannelInstruction* instruction);

// Check if a given instruction (AllReduce or AllGather) matches a DynamicSlice;
// the DynamicSlice has to be the user of the given instruction.
std::optional<ReduceScatterSpec> MatchWithDynamicSlice(
    const HloChannelInstruction* instruction, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims = false,
    bool allow_intervening_reshape = false, int64_t min_rank = 1,
    HloPredicate match_partition_id = HloPredicateIsOp<HloOpcode::kPartitionId>,
    HloPredicate match_replica_id = HloPredicateIsOp<HloOpcode::kReplicaId>,
    bool is_constrain_layout = false, bool use_global_device_ids = false,
    bool is_cross_module = false, bool allow_intervening_bitcast = false,
    bool allow_multiple_users = false);

// Extracts the split dimension spec from a `DynamicSlice` instruction. This
// spec identifies the dimension(s) being operated on by a collective operation
// that is fused with the slice.
//
// The function first attempts a fast path by finding a single dimension where
// the input and output shapes of the `DynamicSlice` differ.
//
// If more than one dimension differs, it re-computes the split dimension by
// examining the slice's offsets. It identifies non-trivial dimensions being
// sliced. A dimension is considered trivial and skipped if its size is 1, or if
// the slice offset along it is a constant zero. This prevents misidentifying a
// dimension that isn't actually being scattered as the split dimension.
std::optional<SplitDimSpec> ExtractSplitDimSpec(
    const HloInstruction& dynamic_slice, bool allow_multiple_split_dims);

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_OPT_UTILS_H_
