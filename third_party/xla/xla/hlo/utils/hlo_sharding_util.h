/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_HLO_UTILS_HLO_SHARDING_UTIL_H_
#define XLA_HLO_UTILS_HLO_SHARDING_UTIL_H_

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/call_graph.h"
#include "xla/shape.h"
#include "xla/statusor.h"
#include "xla/util.h"

namespace xla {
namespace hlo_sharding_util {

struct GatherScatterParallelDims {
  absl::InlinedVector<int64_t, 1> indices_parallel_dims;
  absl::InlinedVector<int64_t, 1> operand_parallel_dims;
  std::vector<int64_t> index_parallel_in_dim;
};

// Determines if the first operand 'potential_subsharding' is a subsharding of
// the second operand 'sharding'. Subsharding means that the tiles in
// 'potential_subsharding' define tiles that have a subset or the same data that
// the tiles in 'sharding' define.
bool IsSubTilingOrEqualSharding(const Shape& shape,
                                const HloSharding& potential_subsharding,
                                const HloSharding& sharding);

// Returns true if the lhs sharding is preferable over the rhs sharding.
// The most specific sharding is tile maximal followed by single device tile
// maximal and finally replicated. This order aims to primarily reduce memory
// usage and secondly reduce total compute.
// Note: This does NOT provide a total ordering as we can have 2 different
// sharding with same preference level.
bool IsShardingMoreSpecific(const HloSharding& lhs, const HloSharding& rhs);

// Tries to refine `to_merge` by combining with `old`. Returns if the final
// `to_merge` is more specific than `old`.
bool MergeSharding(const HloSharding& to_merge, HloSharding* dst,
                   bool may_combine_partial_sharding);

// Merges `to_merge` into `dst` only if they are compatible, and the merged
// sharding has >= minimum_tiles tiles. Returns if merging happened.
bool MergeShardingIfCompatible(const HloSharding& to_merge,
                               int64_t minimum_tiles, HloSharding* dst);

// Find a reasonable common sharding for a list of shardings. The reasonable
// sharding should incur little(the least) amount of total resharding cost when
// resharding all the shardings to this common sharding.
HloSharding FindCommonSharding(
    absl::Span<const HloSharding> shardings,
    std::optional<HloSharding> default_sharding = std::nullopt);

// Given a map<device, occurrence_count>, selects the device with higher
// occurrence count (if any). If top_count in not nullptr, it will receive the
// count of the dominant device returned.
std::optional<int64_t> SelectDominantDevice(
    const std::map<int64_t, int64_t>& device_map, int64_t* top_count);

// Assigns all the instructions of a computation, to a given device.
// This API does not recurse into called computations, and does not assign
// instructions which already have sharding.
void AssignComputationDevice(HloComputation* computation, int64_t device);

// Given an instruction container, returns the device which is most commonly
// occurring among the instructions.
std::optional<int64_t> GetMostOccurringDevice(
    absl::Span<HloInstruction* const> instructions);

// Given a set of computations, tries to extract the dominant device. A device
// is dominant if the combined occurrence among all the instructions of the
// input computations, is greater/equal than/to dominant_factor (real number
// from 0 to 1).
// This API does not recurse into called computations.
// If no device exists that satisfies the condition, the returned optional will
// hold no value.
std::optional<int64_t> GetDominantDevice(
    absl::Span<HloComputation* const> computations, double dominant_factor);

// Returns the HloSharding with the tile dimensions and tile assignment
// transposed based on the specified dimension numbers. In case of a tile
// maximal sharding returns the original sharding.
HloSharding TransposeSharding(const HloSharding& sharding,
                              absl::Span<const int64_t> dimensions);

// Returns the HloSharding with the tile shape reshaped based on the source and
// target shapes and the tile assignment adjusted to correspond to the new tile
// shape or std::nullopt if the resulting reshape would create an invalid
// sharding (non continuous or non uniformly sized tiles). In case of a tile
// maximal sharding returns the original sharding.
std::optional<HloSharding> ReshapeSharding(const Shape& source_shape,
                                           const Shape& target_shape,
                                           const HloSharding& sharding);

// Propagates sharding through reshape. It tries to find partial matches on
// subsets of dimensions that could satisfy ReshapeSharding() constraints, then
// combine them. It doesn't require all dimensions to satisfy the constraints
// of ReshapeSharding().
HloSharding PropagateShardingThroughReshape(const Shape& source_shape,
                                            const Shape& target_shape,
                                            const HloSharding& sharding);

// Returns the HloSharding with the tile dimensions and tile assignment
// reversed based on the specified dimension numbers. In case of a tile
// maximal sharding returns the original sharding.
HloSharding ReverseSharding(const HloSharding& sharding,
                            absl::Span<const int64_t> dimensions);

// Returns a sharding tiled on unique dimension dim by reshaping the tile
// assignment of the sharding argument. Only dimensions in the dims span
// argument are considered for reshaping, the others are ignored.
// Assumptions: sharding is tile sharded, and dim must be included in dims.
HloSharding ReshapeToTileDimension(const HloSharding& sharding, int64_t dim,
                                   absl::Span<const int64_t> dims);

// Returns true if the provided module includes one or more instructions with
// a tile sharding.
bool ContainsTileSharding(const HloModule& module);

// Returns the preferred output sharding for a gather op based on the sharding
// of the indces.
HloSharding GatherOutputShardingFromIndexIndexPassthroughDimensions(
    const HloSharding& index_sharding, const HloInstruction* hlo);

// Returns the preferred index sharding for a gather op based on the sharding
// of the output.
HloSharding GatherIndexShardingFromOutputIndexPassthroughDimensions(
    const HloSharding& output_sharding, const HloInstruction* hlo);

// Returns a new HloSharding for a gather op so that only non offset dimensions
// are sharded. Assume "result" is returned by this function. It is ensured that
// "GetIndexSharding(result, hlo)" will have the same number of elements as
// "result".
HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo);

// Returns the preferred index sharding for a scatter op based on the sharding
// of the data.
HloSharding ScatterIndexShardingFromUpdateIndexPassthroughDimensions(
    const HloSharding& update_sharding, const HloScatterInstruction* scatter);

// Returns the preferred data sharding for a scatter op based on the sharding
// of the index.
HloSharding ScatterUpdateShardingFromIndexIndexPassthroughDimensions(
    const HloSharding& index_sharding, const HloScatterInstruction* scatter);

// Returns a new index sharding for a scatter op so that we only shard on first
// "number of scatter_window_dims" dimensions. Assume "result" is returned by
// this function. It is ensured that
// "ScatterUpdateShardingFromIndexIndexPassthroughDimensions(result, hlo)" will
// have the same number of elements as "result".
HloSharding ScatterEffectiveIndexSharding(const HloSharding& index_sharding,
                                          const HloScatterInstruction& scatter);

// Returns a new data sharding for a scatter op so that we only shard on
// scatter_window_dims. Assume "result" is returned by this function. It is
// ensured that
// "ScatterIndexShardingFromUpdateIndexPassthroughDimensions(result, hlo)" will
// have the same number of elements as "result".
HloSharding ScatterEffectiveDataSharding(const HloSharding& data_sharding,
                                         const HloScatterInstruction& scatter);

// Returns an output sharding of gather by passing through the data operand's
// sharding.
std::optional<HloSharding>
GatherOutputShardingFromOperandOperandPassthroughDimensions(
    const HloSharding& operand_sharding, const HloInstruction& hlo);

// Returns an output sharding of gather by passing through the data operand's
// sharding.
std::optional<HloSharding>
GatherOutputShardingFromOperandOperandPassthroughDimensions(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes);

// Returns an output sharding of gather by passing through the data operand's
// sharding on index parallel dimensions
std::optional<HloSharding> GatherOperandShardingFromOutputParallelDimensions(
    const HloSharding& output_sharding, const HloScatterInstruction& scatter,
    const CallGraph& call_graph);

// Returns a data operand sharding of gather by passing through the output's
// sharding.
std::optional<HloSharding> GatherOperandShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo,
    const CallGraph& call_graph);

// Returns the slice size for a scatter with given operand and update shapes.
std::vector<int64_t> GetScatterSliceSize(const Shape& operand_shape,
                                         const Shape& update_shape,
                                         const ScatterDimensionNumbers& dnums);

// Returns an output sharding of scatter by passing through the update operand's
// sharding.
std::optional<HloSharding> ScatterOutputShardingFromUpdate(
    const HloSharding& update_sharding, const HloScatterInstruction& scatter);

// Returns an update operand sharding of scatter by passing through the output's
// sharding.
std::optional<HloSharding> ScatterUpdateShardingFromOutput(
    const HloSharding& per_output_sharding,
    const HloScatterInstruction& scatter, const CallGraph& call_graph);

// Returns an update operand sharding of scatter by passing through the output's
// sharding on operand pass-through dimensions.
std::optional<HloSharding>
ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
    const HloSharding& output_sharding, const HloInstruction& hlo);

// Returns an update operand sharding of scatter by passing through the output's
// sharding on operand pass-through dimensions.
std::optional<HloSharding>
ScatterUpdateShardingFromOutputOperandPassthroughDimensions(
    const Shape& output_shape, const HloSharding& output_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes);

// Returns an update operand sharding of scatter by passing through the output's
// sharding on index parallel dimensions.
std::optional<HloSharding> ScatterUpdateShardingFromOutputParallelDimensions(
    const HloSharding& output_sharding, const HloScatterInstruction& scatter,
    const CallGraph& call_graph);

// Returns an output sharding of gather or update operand sharding of scatter by
// passing through the indices' sharding on index parallel dimensions.
HloSharding GatherOutputOrScatterUpdateShardingFromIndicesParallelDimensions(
    const HloSharding& indices_sharding,
    const int64_t output_or_update_shape_rank,
    absl::Span<const int64_t> indices_parallel_dims,
    absl::Span<const int64_t> output_or_update_parallel_dims);

// Returns an identity value and an HloOpcode for reduce computation of scatter
// instruction.
// - If computation is add/or, return 0/false with corresponding op code;
// - If computation is multiply/and, return 1/true with corresponding op code.
// - If computation is min/max, return max value/min value with corresponding op
//   code.
// - Otherwise, return error status.
absl::StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter);

// Given a sharding and a list of devices in the topology, return a
// list of the devices that `sharding` applies to.
std::vector<int64_t> DevicesForSharding(
    const HloSharding& sharding, absl::Span<const int64_t> available_devices);

// Returns a sharding that replicates data across devices along the given
// dimensions in the original sharding.
HloSharding PartiallyReplicateTiledShardingOnDims(
    const HloSharding& sharding, absl::Span<const int64_t> dims_to_replicate);

// Returns a sharding that replicates data across devices along all dimensions
// but the given ones to keep in the original sharding.
HloSharding PartiallyReplicateTiledShardingOnAllDimsExcept(
    const HloSharding& sharding, absl::Span<const int64_t> dims_to_keep);

// Returns a sharding that replicates all data dimensions, but keep manual
// subgroups. If data_rank is provided >= 0, the result sharding's data rank
// will be set to it.
HloSharding ReplicateAllDataDims(const HloSharding& sharding,
                                 int64_t data_rank = -1);

// Returns a sharding the removes given tile dimensions.
//
// Precondition: if not tile maximal, the size of each tile dimension must be 1.
HloSharding RemoveShapeDimensions(const HloSharding& sharding,
                                  absl::Span<const int64_t> dims_to_remove);

// Similar to TransposeSharding(), but allows removing/adding non-partitioned
// dimensions. In src_to_tgt and tgt_to_src, -1 represents a non-existing
// dimension.
std::optional<HloSharding> TransposeShardingWithCollapsedDims(
    const HloSharding& source, absl::Span<int64_t const> src_to_tgt,
    absl::Span<int64_t const> tgt_to_src);

// Returns the iota dimension if maybe_iota is an kIota instruction or
// equivalent to kIota.
std::optional<int64_t> GetDimensionForIota(const HloInstruction* maybe_iota,
                                           const CallGraph& call_graph);

// Returns identified parallel dimensions of operands and indices for Gather.
std::optional<GatherScatterParallelDims> GetGatherParallelBatchDims(
    const HloInstruction& hlo, const CallGraph& call_graph);

// Returns identified parallel dimensions of operands and indices for Scatter.
std::optional<GatherScatterParallelDims> GetScatterParallelBatchDims(
    const HloInstruction& hlo, const CallGraph& call_graph);

// Returns the parallel dimensions of the output of a gather based on the
// parallel dimensions of the operands and indices.
absl::InlinedVector<int64_t, 1> GetGatherParallelOutputDims(
    const HloInstruction& hlo, const GatherScatterParallelDims& parallel_dim);

// Returns the parallel dimensions of the update of a scatter based on the
// parallel dimensions of the operands and indices.
absl::InlinedVector<int64_t, 1> GetScatterParallelUpdateDims(
    const HloInstruction& hlo, const GatherScatterParallelDims& parallel_dim);

// Returns the operand pass-through dimensions for gather operand.
absl::InlinedVector<int64_t, 1> GetGatherOperandPassthroughOperandDims(
    const Shape& operand_shape, const HloInstruction& hlo,
    absl::Span<const int64_t> slice_sizes);

// Returns the operand pass-through dimensions for scatter operand(s).
absl::InlinedVector<int64_t, 1> GetScatterOperandPassthroughOperandDims(
    const Shape& operand_shape, const HloSharding& operand_sharding,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes);

absl::InlinedVector<int64_t, 1> GetGatherOperandPassthroughOutputDims(
    const Shape& output_shape, const Shape& operand_shape,
    const HloInstruction& hlo, absl::Span<const int64_t> slice_sizes);

absl::InlinedVector<int64_t, 1> GetScatterOperandPassthroughUpdateDims(
    const Shape& update_shape, const Shape& operand_shape,
    const HloSharding& operand_sharding, const HloInstruction& hlo,
    absl::Span<const int64_t> slice_sizes);

// Returns the index pass-through dimensions for gather/scatter indices.
absl::InlinedVector<int64_t, 1> GetGatherScatterIndexPassthroughIndexDims(
    const int64_t indices_rank, const int64_t index_vector_dim);

// Returns the index pass-through dimensions for gather output/scatter update.
absl::InlinedVector<int64_t, 1>
GetGatherScatterIndexPassthroughOutputOrUpdateDims(
    const int64_t output_or_update_rank,
    absl::Span<const int64_t> offset_or_window_dims);

// Returns the parallel dimensions of the data operand of a gather/scatter with
// the order of the parallel dimensions matching that of the parallel dimensions
// of the indices.
absl::InlinedVector<int64_t, 1> IndexAlignedOperandParallelDims(
    const GatherScatterParallelDims& parallel_dims);

// Represents grouping devices in a tiled sharding along certain dimensions.
// Elements in group dimensions define different device groups, and the sharding
// represents the in-group sharding.
struct GroupedSharding {
  GroupedSharding(std::vector<std::vector<int64_t>> device_groups,
                  DimensionVector group_dims, DimensionVector group_dim_sizes,
                  int64_t data_rank, HloSharding grouped_sharding,
                  bool subgroup_manual = false)
      : device_groups(std::move(device_groups)),
        group_dims(std::move(group_dims)),
        group_dim_sizes(std::move(group_dim_sizes)),
        data_rank(data_rank),
        sharding(std::move(grouped_sharding)),
        subgroup_manual(subgroup_manual) {}
  std::string ToString() const;
  std::vector<std::vector<int64_t>> device_groups;
  DimensionVector group_dims;
  DimensionVector group_dim_sizes;
  int64_t data_rank;
  HloSharding sharding;
  bool subgroup_manual;
};

// Creates a GroupedSharding for a tiled sharding with group dim shard sizes.
GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64_t> group_dims,
                                    absl::Span<const int64_t> group_dim_shards,
                                    bool subgroup_manual = false);

// Creates a GroupedSharding for a tiled sharding.
GroupedSharding GroupShardingOnDims(const HloSharding& sharding,
                                    absl::Span<const int64_t> group_dims,
                                    bool subgroup_manual = false);

// Same as above, but exclude group dims instead.
GroupedSharding GroupShardingOnAllDimsExcept(
    const HloSharding& sharding, absl::Span<const int64_t> non_group_dims,
    bool subgroup_manual = false);

// Creates a GroupedSharding by trying to do the following in sequence:
//
// 1. Group on partially replicated dimensions, which preserves the existing
// tiled sharding in the group.
// 2. If option 1 doesn't have enough dimensions, try borrowing dimensions from
// replicable_dims in order, until it has enough dimensions. This partly
// preserves the existing tiled sharding in the group. (e.g. if we need 4
// groups, while our sharding is {[4,8,2]<=[64] last_tile_dim_replicate}, and if
// we borrow 2 dimensions from the first dimension(i.e. the 4-way partition),
// combined with the partially replicated 2, we will be able to group the
// sharding into 4 groups, and we have grouped sub-sharding [2,8]<=[16] instead.
// 3. Otherwise replicate the whole thing.
//
// This does not guarantee the consistency of the ordering of the tile
// assignment, and should be used with AlignGroup where its tile assignment
// doesn't matter and will always align to some other tile assignment.
GroupedSharding GroupShardingOnReplicatedDim(
    const HloSharding& sharding, int64_t num_groups, int64_t num_tiles,
    int64_t data_rank, absl::Span<const int64_t> replicable_dims = {});

// Get group sharding for replicated sharding.
GroupedSharding GetGroupedReplicatedSharding(const int64_t num_groups,
                                             const int64_t num_tiles,
                                             const int64_t data_rank);

// Get group sharding for each manual subgroup.
GroupedSharding GetManualSubgroupSharding(const HloSharding& sharding);

// Create a group sharding over the partially replicated dimension re-using an
// existing device group subdivision to avoid unexpected devices reordering.
std::optional<GroupedSharding>
PartialReplicatedGroupShardingWithAssignedDeviceGroups(
    const HloSharding& sharding, int64_t num_shards,
    const std::vector<std::vector<int64_t>>& device_groups);

// Reconstructs the ungrouped sharding from a GroupedSharding.
HloSharding UngroupSharding(const GroupedSharding& grouped_sharding);

// Check if the device groups are match for the LHS or RHS group shardings.
bool DeviceGroupsAreMatch(GroupedSharding& lhs, GroupedSharding& rhs,
                          bool ignore_group_order = true);

// Spawns a new dimension by splitting an existing dimension and generating a
// new dimension to its right of the passed down size. The original dimension
// will be of size "original_dim_size / new_dim_size". The original dimension
// size needs to be divisible by new_dim_size.
HloSharding SplitShardingDimension(const HloSharding& sharding,
                                   int64_t dimension, int64_t new_dim_size);

// Merges a dimension
// to its left. The new dimension will be of size
// dimensions[dimension] * dimensions[dimension+1}.
HloSharding MergeShardingDimension(const HloSharding& sharding,
                                   int64_t dimension);

// Creates a tuple sharding by combining sharding on the elements of the tuple.
// If none of the elements have a sharding, return nullptr.
std::shared_ptr<const HloSharding> CreateTupleSharding(
    const Shape& shape, absl::Span<const HloInstruction* const> elements);

// Tests whether the sort operand is sharded along the sort dimension and there
// exists a free (i.e., unsharded) dimension to move the sharding into.
bool IsSortOperandShardingMovable(const HloInstruction* sort_operand,
                                  int64_t sort_dim);

// Returns a set of parallel dimensions for Gather/Scatter instructions given
// the parameters for the op.
std::optional<GatherScatterParallelDims> GetGatherScatterBatchParallelDims(
    const HloInstruction* indices, absl::Span<const int64_t> slice_sizes,
    int64_t index_vector_dim, absl::Span<const int64_t> index_map,
    const CallGraph& call_graph);

// Returns the sharding of an output of an instruction. Some instructions have
// special handling like Outfeed and this function takes care of those.
std::optional<HloSharding> GetOutputSharding(const HloInstruction* instruction);

// Returns the un-tiled shape.
Shape UntileShape(const HloSharding& sharding, const Shape& shape);

// Returns the un-tiled shape.
// REQUIRES: !sharding.IsTuple()
Shape UntileLeafShape(const HloSharding& sharding, const Shape& shape);

// Returns the tiled shape.
Shape TileShape(const HloSharding& sharding, const Shape& shape);

// Returns the tiled shape.
// REQUIRES: !sharding.IsTuple()
Shape TileLeafShape(const HloSharding& sharding, const Shape& shape);

}  // namespace hlo_sharding_util
}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_SHARDING_UTIL_H_
