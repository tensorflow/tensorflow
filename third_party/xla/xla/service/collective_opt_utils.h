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
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/util.h"

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
  std::vector<int64_t> split_dims = {};
  int64_t split_dim = -1;
  int64_t split_dim_size = -1;
};

// A map from a partitioned offset to its corresponding partition ID.
using OffsetToIdMap = absl::btree_map<int64_t, int64_t>;

// Represents the mapping of partition offsets to partition IDs for each replica
// group. This can be derived from either a dynamic-slice or an all-gather
// operation.
struct PartitionOffsetSpec {
  // A list of OffsetToIdMap, one for each replica group.
  std::vector<OffsetToIdMap> per_replica_group_offsets;
};

// A list of pairs mapping a source partition ID to a destination partition ID
// for a collective permute operation.
using PermutationPairs = std::vector<std::pair<int64_t, int64_t>>;

struct AllGatherDynamicSliceMatchSpec {
  PermutationPairs permutation_pairs;
};

// Function to map a replica/partition/global ID to an offset in the offset
// table, based on the given scalar offset HLO. For example, if the HLO is
// kPartitionId but the all-reduce uses global IDs, then the function maps
// global IDs to partition IDs. It returns -1 if the HLO cannot be understood.
using MapIdToTableOffset =
    std::function<int64_t(const HloInstruction*, int64_t)>;

// Matches the given all-reduce operation to a reduce-scatter pattern.
std::optional<ReduceScatterSpec> MatchReduceScatter(
    const HloAllReduceInstructionBase* ar, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims = false,
    bool allow_intervening_reshape = false, int64_t min_rank = 1,
    HloPredicate match_partition_id = HloPredicateIsOp<HloOpcode::kPartitionId>,
    HloPredicate match_replica_id = HloPredicateIsOp<HloOpcode::kReplicaId>,
    bool allow_intervening_bitcast = false);

// Checks whether AG(ICI) and its user DS(ICI) can be canceled out.
std::optional<ReduceScatterSpec> AllGatherDynamicSliceCancellation(
    const HloAllGatherInstruction* ag, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims = false,
    bool allow_intervening_reshape = false, int64_t min_rank = 1,
    HloPredicate match_partition_id = HloPredicateIsOp<HloOpcode::kPartitionId>,
    HloPredicate match_replica_id = HloPredicateIsOp<HloOpcode::kReplicaId>,
    bool allow_intervening_bitcast = false, bool allow_multiple_users = false);

// Matches an all-gather with a dynamic slice whose offset matches a permuted
// partition offset.
//
// This pattern is commonly used to implement permutations or data permuted
// across partitions. An all-gather collects data from all partitions, and then
// a dynamic-slice on each partition selects a slice from a remote partition,
// effectively permuting the data. This function identifies such patterns and
// extracts the permutation pairs (source partition, destination partition).
//
// The function matches a specific pattern:
//   - All-gather with flattened-id mode.
//   - Partitioning with `num_partitions > 1` and `num_replicas = 1`.
//   - AG Sharding and dynamic-sloce slicing on same dimension.
//
// For example, the following HLO performs a reverse permutation across 8
// partitions (partition `i` gets data from partition `7-i`):
//
// HloModule module
// ENTRY entry {
//   p = f32[32,8,128] parameter(0)
//   ag = f32[256,8,128] all-gather(p), replica_groups={{0,1,2,3,4,5,6,7}},
//     dimensions={0}, channel_id=1, use_global_device_ids=true
//   pid = u32[] partition-id()
//   permuted_index_list = s32[8]{0} constant({224,192,160,128,96,64,32,0})
//   offset = s32[1] dynamic-slice(permuted_index_list, pid),
//   dynamic_slice_sizes={1} offset_reshape = s32[] reshape(offset) zero = s32[]
//   constant(0) ROOT ds = f32[32,8,128] dynamic-slice(ag, offset_reshape, zero,
//   zero),
//     dynamic_slice_sizes={32,8,128}
// }
//
// This function would match this pattern and return permutation pairs like
// {{0,7}, {1,6}, ..., {7,0}}.
//

std::optional<AllGatherDynamicSliceMatchSpec>
MatchPermutedSliceAndPartitionOffset(const HloAllGatherInstruction* ag,
                                     int64_t num_partitions,
                                     int64_t num_replicas,
                                     HloPredicate match_partition_id,
                                     bool allow_multiple_users);

// Checks whether the replica groups in the given channel instruction are
// of the same size.
bool CheckUniformReplicaGroups(const HloChannelInstruction* instruction);

struct CollectiveUsers {
  HloInstruction* dynamic_slice = nullptr;
  HloInstruction* bitcast = nullptr;
  HloInstruction* reshape = nullptr;
};

// Extracts the dynamic-slice user from a collective instruction, potentially
// looking through reshapes and bitcasts.
std::optional<CollectiveUsers> FindUniqueDynamicSliceUserFromCollective(
    const HloChannelInstruction* absl_nonnull instruction,
    bool allow_multiple_users, bool allow_intervening_reshape,
    bool allow_intervening_bitcast);

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
// spec identifies the dimension(s) being operated on by a collective
// operation that is fused with the slice.
//
// The function first attempts a fast path by finding a single dimension where
// the input and output shapes of the `DynamicSlice` differ.
//
// If more than one dimension differs, it re-computes the split dimension by
// examining the slice's offsets. It identifies non-trivial dimensions being
// sliced. A dimension is considered trivial and skipped if its size is 1, or
// if the slice offset along it is a constant zero. This prevents
// misidentifying a dimension that isn't actually being scattered as the split
// dimension.
std::optional<SplitDimSpec> ExtractSplitDimSpec(
    const HloInstruction& dynamic_slice, bool allow_multiple_split_dims);

// Extracts the mapping from slice offsets to partition IDs from a dynamic-slice
// instruction that is fed by an all-gather.
std::optional<PartitionOffsetSpec> GetIndicesSpecForDynamicSlice(
    const HloAllGatherInstruction* absl_nonnull ag_instr,
    const HloInstruction* absl_nonnull offset_hlo,
    const std::function<int64_t(const HloInstruction*, int64_t)>& map_id);

// Extracts the mapping from slice offsets to partition IDs from a dynamic-slice
// instruction that is fed by an all-gather where the dynamic-slice's offset
// allows one multiply instruction.
std::optional<PartitionOffsetSpec> GetIndicesSpecForDynamicSliceWithMultiply(
    const HloAllGatherInstruction* absl_nonnull ag_instr,
    const HloInstruction* absl_nonnull offset_hlo,
    const std::function<int64_t(const HloInstruction*, int64_t)>& map_id,
    int64_t split_dim_size);

// Extracts the PartitionOffsetSpec from an all-gather instruction.
std::optional<PartitionOffsetSpec> ExtractPartitionOffsetSpec(
    const HloAllGatherInstruction* ag, int64_t num_partitions);

// Extracts pattern dynamic-slice(pad(all-gather)).
// Returns true if the pattern is found, and set pad_hlo and ag_hlo.
// Otherwise, returns false.
bool MatchDsPadAllGather(HloInstruction* ds_hlo, HloInstruction** pad_hlo,
                         HloInstruction** ag_hlo);

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_OPT_UTILS_H_
