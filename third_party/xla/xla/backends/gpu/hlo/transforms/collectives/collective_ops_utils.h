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

#ifndef XLA_BACKENDS_GPU_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_
#define XLA_BACKENDS_GPU_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/device_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

enum class GPUCommunicationType {
  // The communication type could not be determined.
  UNDEFINED = 0,
  // Communication involves devices from multiple hosts, and every host
  // involved in the communication pattern has all of its devices participating.
  MULTI_HOST_WORLD_LEVEL = 1,
  // Communication involves devices from multiple hosts, but at least one of
  // the involved hosts has only a subset of its devices participating.
  MULTI_HOST_NON_WORLD_LEVEL = 2,
  // All devices participating in the collective operation reside on the same
  // fast-interconnect domain.
  SINGLE_PARTITION = 3
};

// Returns the type of communication pattern for a channel instruction.
absl::StatusOr<GPUCommunicationType> CommunicationType(
    int partition_size, const HloChannelInstruction& instr,
    const se::GpuComputeCapability& gpu_version);

// Enum to categorize collective-permute cost models based on communication
// patterns. The cost model is determined by the highest-latency pattern
// present in any device: TwoWayHasNonMutual > TwoWayAllMutual > OneWay.
enum class CollectivePermuteCostModelType {
  // This is currently only used for CollectivePermute instructions with empty
  // source-target pairs.
  // TODO(b/460155942): Remove this field once the HLO verifier stop supporting
  // empty source-target pairs.
  kUnknown,
  // Intra-partition: All devices only send or only receive data.
  kIntraPartitionOneWay,
  // Intra-partition: Devices send/receive, but only with the same peer
  // (e.g., {{0,1},{1,0}}).
  kIntraPartitionTwoWayAllMutual,
  // Intra-partition: At least one device sends to one peer and receives from
  // another (e.g., {{0,1},{1,2}}).
  kIntraPartitionTwoWayHasNonMutual,
  // Inter-partition: All devices only send or only receive data.
  kInterPartitionOneWay,
  // Inter-partition: Devices send/receive, but only with the same peer.
  kInterPartitionTwoWayAllMutual,
  // Inter-partition: At least one device sends to one peer and receives from
  // another.
  kInterPartitionTwoWayHasNonMutual,
};

// Returns cost model type based on collective-permute properties.
CollectivePermuteCostModelType GetCollectivePermuteCostModelType(
    const HloCollectivePermuteInstruction& instr,
    int64_t num_devices_per_partition);

// Returns true if instruction is a synchronous collective op.
bool IsGPUSyncCollective(const HloInstruction& instr);

// Returns true if all devices are within the same NVLink domain (slice).
bool IsIntraNVLinkDomain(const HloModuleConfig& config, int64_t slice_size);

// Returns true if the target GPU supports load-store accessible (LSA)
// symmetric memory.
bool IsLsaPossible(const GpuTopology& gpu_topology);

// Returns the size of the block of consecutive global device ids that a
// collective kernel may span: one NVLink partition (slice) if LSA is supported,
// xla_gpu_unsupported_use_cross_host_one_shot_kernel is enabled for `op_type`,
// and a partition spans multiple processes, otherwise one process. All devices
// of a replica group must be in the same block, i.e. have the same `id / size`.
int64_t GetCollectiveKernelDomainSize(
    const GpuTopology& gpu_topology, const DebugOptions& debug_options,
    std::optional<DebugOptions::CollectiveOpType> op_type);

// Returns true if all replicas in every replica group of the collective
// are located on the same host (node).
bool IsAllReplicasLocal(int64_t gpus_per_host,
                        absl::Span<const ReplicaGroup> replica_groups,
                        CollectiveOpGroupMode group_mode,
                        const DeviceAssignment* device_assignment = nullptr);

// Returns true if all replicas in every replica group of `instruction` are
// within a single process/host (`gpu_topology.num_devices_per_process()`).
absl::StatusOr<bool> IsCollectiveSingleHost(
    const GpuTopology& gpu_topology, const HloInstruction& instruction,
    const DeviceAssignment* device_assignment = nullptr);

// Returns true if cross-host one-shot collective kernel is enabled for
// `instruction` and all replicas in every replica group are within a single
// NVLink slice (`gpu_topology.slice_size()`).
absl::StatusOr<bool> IsCrossHostCollectiveKernelPossible(
    const GpuTopology& gpu_topology, const HloInstruction& instruction,
    const DeviceAssignment* device_assignment = nullptr);

// Returns true if all replicas in every replica group of the collective
// are located on the same host, or within the same NVLink slice when
// cross-host collective kernels and LSA are supported.
bool AreAllReplicasOnSameSlice(
    const GpuTopology& gpu_topology, const DebugOptions& debug_options,
    std::optional<DebugOptions::CollectiveOpType> op_type,
    absl::Span<const ReplicaGroup> replica_groups,
    CollectiveOpGroupMode group_mode,
    const DeviceAssignment* device_assignment = nullptr);

// Returns true if all replicas in every replica group of `instruction` are on
// the same host, or within the same NVLink slice when cross-host collective
// kernels and LSA are supported.
absl::StatusOr<bool> AreAllReplicasOnSameSlice(
    const GpuTopology& gpu_topology, const HloInstruction& instruction,
    const DeviceAssignment* device_assignment = nullptr);

bool IsAllReplicasLocal(const GpuTopology& gpu_topology,
                        absl::Span<const ReplicaGroup> replica_groups,
                        CollectiveOpGroupMode group_mode,
                        const DeviceAssignment* device_assignment = nullptr);

// Returns true if the instruction was generated by the SPMD partitioner.
// Checks both the kSpmdGeneratedAttr frontend attribute and the
// is_spmd_generated field in CollectiveBackendConfig (set when collectives
// that carried the attribute are combined).
bool IsSpmdGenerated(const HloInstruction& instr);

// Returns true if the instruction is a Triton collective kernel.
bool IsTritonCollectiveKernel(
    CollectiveBackendConfig::CollectiveKernelStrategy kernel_strategy);

// Returns true if `instr` is a two-shot Triton all-gather instruction or a
// custom fusion containing one.
bool IsTritonTwoShotAllGather(const HloInstruction& instr);

// Returns the set of HloOpcodes that should be fused.
absl::StatusOr<absl::flat_hash_set<HloOpcode>> OpcodesForTritonCollectives(
    const DebugOptions& debug_options);

// Heuristic maximum warps per block across collective kernels.
// Triton expects a power of 2 for num_warps.
inline constexpr int64_t kMaxCollectiveWarpsPerBlock = 16;

// Returns a tile such that each dimension is a power of two and that the total
// number of blocks does not exceed num_blocks. Returns the tile sizes in
// same order as the output shape. It picks tiles greedily from the most major
// dimension to the most minor dimension. Eg:
// - [1024, 1024] with num_blocks = 32 returns [32, 1024]
// - [8192, 16] with num_blocks = 32 returns [256, 16].
// - [5, 1024] with num_blocks = 32 returns [2, 128].
// Note that it can be that the number of blocks used is less than num_blocks.
// @pre num_blocks > 0.
llvm::SmallVector<int64_t> GreedyPowerOfTwoTiles(const Shape& output_shape,
                                                 int32_t num_blocks);

// Returns the output shapes of `instr` as a span (a 1-element span for
// array-shaped instructions, or `tuple_shapes()` for tuple-shaped ones).
// The returned span aliases `instr.shape()` and must not outlive `instr`.
absl::Span<const Shape> GetOutputShapes(
    const HloInstruction& instr ABSL_ATTRIBUTE_LIFETIME_BOUND);

// Tiles one rank's part of a collective output across `blocks_per_rank`
// programs. `collective_dim` is first divided by `num_devices` (R).
//
// All-gather: output f32[16] on 2 ranks -> one rank's part is f32[8].
//
// Reduce-scatter: output f32[8] on 2 ranks. Each block copies one input tile
// and writes one output tile. The input tile is R times the output tile.
// With two blocks on rank 0:
//   block 0: copies input 0..8 (for rank 0). Reduces and writes output 0..4.
//   block 1: copies input 8..16 (for rank 1). Reduces and writes output 4..8.
absl::StatusOr<llvm::SmallVector<int64_t>> GetCollectiveOutputTile(
    const Shape& output_shape, int64_t collective_dim, int64_t num_devices,
    int32_t blocks_per_rank);

// Computes one output tile per output of an all-gather or reduce-scatter
// `instr` (see GetCollectiveOutputTile). Each output is tiled independently
// using `num_blocks / num_devices` programs per rank; for variadic collectives,
// smaller outputs get padded with masked-out blocks by
// `GetIndexWiseVariadicSchedule`.
absl::StatusOr<std::vector<llvm::SmallVector<int64_t>>>
GetCollectiveOutputTiles(const HloInstruction& instr, int64_t collective_dim,
                         int64_t num_devices, int32_t num_blocks);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_HLO_TRANSFORMS_COLLECTIVES_COLLECTIVE_OPS_UTILS_H_
