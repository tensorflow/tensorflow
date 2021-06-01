/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_UTIL_H_

#include <map>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_sharding.h"

namespace xla {
namespace hlo_sharding_util {

struct GatherParallelDims {
  absl::InlinedVector<int64, 1> indices_parallel_dims;
  absl::InlinedVector<int64, 1> operand_parallel_dims;
  std::vector<int64> index_parallel_in_dim;
};

// Returns true if the lhs sharding is preferable over the rhs sharding.
// The most specific sharding is tile maximal followed by single device tile
// maximal and finally replicated. This order aims to primarily reduce memory
// usage and secondly reduce total compute.
// Note: This does NOT provide a total ordering as we can have 2 different
// sharding with same preference level.
bool IsShardingMoreSpecific(const HloSharding& lhs, const HloSharding& rhs);

// Tries to refine `to_merge` by combining with `old`. Returns if the final
// `to_merge` is more specific than `old`.
bool MergeSharding(const HloSharding& old, HloSharding* to_merge,
                   bool may_combine_partial_sharding);

// Merges `to_merge` into `dst` only if they are compatible, and the merged
// sharding has >= minimum_tiles tiles. Returns if merging happened.
bool MergeShardingIfCompatible(const HloSharding& to_merge, int64 minimum_tiles,
                               HloSharding* dst);

// Given a map<device, occurrence_count>, selects the device with higher
// occurrence count (if any). If top_count in not nullptr, it will receive the
// count of the dominant device returned.
absl::optional<int64> SelectDominantDevice(
    const std::map<int64, int64>& device_map, int64* top_count);

// Assigns all the instructions of a computation, to a given device.
// This API does not recurse into called computations, and does not assign
// instructions which already have sharding.
Status AssignComputationDevice(HloComputation* computation, int64 device);

// Given an instruction container, returns the device which is most commonly
// occurring among the instructions.
absl::optional<int64> GetMostOccurringDevice(
    absl::Span<HloInstruction* const> instructions);

// Given a set of computations, tries to extract the dominant device. A device
// is dominant if the combined occurrence among all the instructions of the
// input computations, is greater/equal than/to dominant_factor (real number
// from 0 to 1).
// This API does not recurse into called computations.
// If no device exists that satisfies the condition, the returned optional will
// hold no value.
StatusOr<absl::optional<int64>> GetDominantDevice(
    absl::Span<HloComputation* const> computations, double dominant_factor);

// Returns the HloSharding with the tile dimensions and tile assignment
// transposed based on the specified dimension numbers. In case of a tile
// maximal sharding returns the original sharding.
HloSharding TransposeSharding(const HloSharding& sharding,
                              const std::vector<int64>& dimensions);

// Returns the HloSharding with the tile shape reshaped based on the source and
// target shapes and the tile assignment adjusted to correspond to the new tile
// shape or absl::nullopt if the resulting reshape would create an invalid
// sharding (non continuous or non uniformly sized tiles). In case of a tile
// maximal sharding returns the original sharding.
absl::optional<HloSharding> ReshapeSharding(const Shape& source_shape,
                                            const Shape& target_shape,
                                            const HloSharding& sharding);

// Returns the HloSharding with the tile dimensions and tile assignment
// reversed based on the specified dimension numbers. In case of a tile
// maximal sharding returns the original sharding.
HloSharding ReverseSharding(const HloSharding& sharding,
                            absl::Span<const int64> dimensions);

// Returns a sharding tiled on unique dimension dim by reshaping the tile
// assignment of the sharding argument. Only dimensions in the dims span
// argument are considered for reshaping, the others are ignored.
// Assumptions: sharding is tile sharded, and dim must be included in dims.
HloSharding ReshapeToTileDimension(const HloSharding& sharding, int64 dim,
                                   absl::Span<const int64> dims);

// Returns true if the provided module includes one or more instructions with
// a tile sharding.
bool ContainsTileSharding(const HloModule& module);

// Returns the preferred output sharding for a gather op based on the sharding
// of the indces.
HloSharding GatherOutputSharding(const HloSharding& index_sharding,
                                 const HloInstruction* hlo);

// Returns the preferred index sharding for a gather op based on the sharding
// of the output.
HloSharding GatherIndexSharding(const HloSharding& output_sharding,
                                const HloInstruction* hlo);

// Returns a new HloSharding for a gather op so that only non offset dimensions
// are sharded. Assume "result" is returned by this function. It is ensured that
// "GetIndexSharding(result, hlo)" will have the same number of elements as
// "result".
HloSharding GatherEffectiveOutputSharding(const HloInstruction& hlo);

// Returns the preferred index sharding for a scatter op based on the sharding
// of the data.
HloSharding ScatterIndexSharding(const HloSharding& data_sharding,
                                 const HloInstruction* hlo);

// Returns the preferred data sharding for a scatter op based on the sharding
// of the index.
HloSharding ScatterDataSharding(const HloSharding& index_sharding,
                                const HloInstruction* hlo);

// Returns a new index sharding for a scatter op so that we only shard on first
// "number of scatter_window_dims" dimensions. Assume "result" is returned by
// this function. It is ensured that "ScatterDataSharding(result, hlo)" will
// have the same number of elements as "result".
HloSharding ScatterEffectiveIndexSharding(const HloSharding& index_sharding,
                                          const HloInstruction& hlo);

// Returns a new data sharding for a scatter op so that we only shard on
// scatter_window_dims. Assume "result" is returned by this function. It is
// ensured that "ScatterIndexSharding(result, hlo)" will have the same number of
// elements as "result".
HloSharding ScatterEffectiveDataSharding(const HloSharding& data_sharding,
                                         const HloInstruction& hlo);

// Returns an output sharding of gather by passing through the data operand's
// sharding.
absl::optional<HloSharding> GatherOutputShardingFromDataOperand(
    const HloSharding& data_operand_sharding, const HloInstruction& hlo,
    const Shape& output_shape, const Shape& operand_shape);

// Returns a data operand sharding of gather by passing through the output's
// sharding.
absl::optional<HloSharding> GatherDataOperandShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo);

// Returns an output sharding of scatter by passing through the update operand's
// sharding.
absl::optional<HloSharding> ScatterOutputShardingFromUpdate(
    const HloSharding& update_sharding, const HloInstruction& hlo);

// Returns an update operand sharding of scatter by passing through the output's
// sharding.
absl::optional<HloSharding> ScatterUpdateShardingFromOutput(
    const HloSharding& output_sharding, const HloInstruction& hlo);

// Returns an identity value and an HloOpcode for reduce computation of scatter
// instruction.
// - If computation is add/or, return 0/false with corresponding op code;
// - If computation is multiply/and, return 1/true with corresponding op code.
// - If computation is min/max, return max value/min value with corresponding op
//   code.
// - Otherwise, return error status.
StatusOr<std::pair<std::unique_ptr<HloInstruction>, HloOpcode>>
IdentityValueAndHloOpcodeForScatterReduceComputation(
    const HloScatterInstruction& scatter);

// Given a sharding and a list of devices in the topology, return a
// list of the devices that `sharding` applies to.
std::vector<int64> DevicesForSharding(
    const HloSharding& sharding, const std::vector<int64>& available_devices);

// Returns a sharding that replicates data across devices along the given
// dimensions in the original sharding.
HloSharding PartiallyReplicateTiledShardingOnDims(
    const HloSharding& sharding, absl::Span<const int64> dims_to_replicate);

// Returns a sharding that replicates data across devices along all dimensions
// but the given ones to keep in the original sharding.
HloSharding PartiallyReplicateTiledShardingOnAllDimsExcept(
    const HloSharding& sharding, absl::Span<const int64> dims_to_keep);

// Returns a sharding the removes given tile dimensions.
//
// Precondition: if not tile maximal, the size of each tile dimension must be 1.
HloSharding RemoveShapeDimensions(const HloSharding& sharding,
                                  const std::vector<int64>& dims_to_remove);

// Similar to TransposeSharding(), but allows removing/adding non-partitioned
// dimensions. In src_to_tgt and tgt_to_src, -1 represents a non-existing
// dimension.
absl::optional<HloSharding> TransposeShardingWithCollapsedDims(
    const HloSharding& source, absl::Span<int64 const> src_to_tgt,
    absl::Span<int64 const> tgt_to_src);

// Returns the iota dimension if maybe_iota is an kIota instruction or
// equivalent to kIota.
absl::optional<int64> GetDimensionForIota(const HloInstruction* maybe_iota);

// Returns identified parallel dimensions for Gather.
absl::optional<GatherParallelDims> GetGatherBatchParallelDims(
    const HloInstruction& hlo);

// Returns the parallel dimensions of the output of a gather based on the
// parallel dimensions of the input.
absl::InlinedVector<int64, 1> GatherParallelOutputDims(
    const HloInstruction& gather, const GatherParallelDims& parallel_dim);

// Returns the parallel dimensions of the data operand of a gather with the
// order of the parallel dimensions matching that of the parallel dimensions
// of the output.
absl::InlinedVector<int64, 1> GatherOutputAlignedOperandParallelDims(
    const HloInstruction& gather, const GatherParallelDims& parallel_dims);

}  // namespace hlo_sharding_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_UTIL_H_
