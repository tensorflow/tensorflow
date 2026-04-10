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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/p2p_thunk_common.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/buffer_use.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/computation_placer.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// Thunk that performs a collective permute.
class CollectivePermuteThunk : public CollectiveThunk {
 public:
  // LocalPermuteState keeps a state of collective permute operation which
  // is executed on a local GPU clique using p2p memory access.
  //
  // When collective permute is done with a local GPU clique, we can perform
  // it as a sequence of simple D2D copies between ranks. To be able to do that,
  // first we need to exchange source and target buffers from all participating
  // ranks. We do it in thunk initialization time, when buffers allocations are
  // already available.
  struct LocalPermuteState {
    absl::flat_hash_map<RankId, std::vector<DeviceBufferPair>> buffer_pairs;
  };

  CollectivePermuteThunk(ThunkInfo thunk_info,
                         const HloCollectivePermuteInstruction* instr,
                         int64_t replica_count, int64_t partition_count,
                         const std::vector<Buffer>& buffers,
                         bool p2p_memcpy_enabled);
  CollectivePermuteThunk(ThunkInfo thunk_info, const P2PConfig& config,
                         const std::vector<Buffer>& buffers,
                         bool p2p_memcpy_enabled);

  static P2PConfig GetP2PConfig(const HloCollectivePermuteInstruction* instr,
                                int64_t replica_count, int64_t partition_count);

  static bool IsDegenerate(const HloCollectivePermuteInstruction* instr,
                           int64_t replica_count, int64_t partition_count);

  static CollectiveOpGroupMode GetGroupMode(
      const HloCollectivePermuteInstruction* instr);

  static absl::string_view GetHloOpName() { return "collective-permute-start"; }

  const CollectiveConfig& config() const override { return config_.config; }

  absl::Span<const Buffer> buffers() const { return buffers_; }

  const P2PConfig& p2p_config() const { return config_; }

  static absl::StatusOr<std::unique_ptr<CollectivePermuteThunk>> FromProto(
      ThunkInfo thunk_info, const CollectivePermuteStartThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations);

  absl::StatusOr<ThunkProto> ToProto() const override;

  BufferUses buffer_uses() const override;

 protected:
  // No rendezvous needed when using P2P memcpy in local mode instead of NCCL.
  bool RequiresRendezvous() const override { return !p2p_memcpy_enabled_; }

  absl::Status InitializeCollective(const InitializeParams& params,
                                    const GpuCliqueKey& clique_key) override;

  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& clique_key, se::Stream& stream,
                             Communicator& comm) override;

 private:
  // Computes connected components from the source-target pairs in config.
  // Returns a map from each logical ID to its component members.
  static absl::flat_hash_map<int64_t, std::vector<int64_t>>
  InitConnectedComponents(const P2PConfig& config, bool p2p_memcpy_enabled);

  // Builds a GpuCliqueKey covering only the devices in the connected component
  // that `current_id` belongs to. Returns the key and whether it is fully
  // local.
  absl::StatusOr<GpuCliqueKey> BuildCommunicatingCliqueKey(
      int64_t current_id, const CollectiveParams& params) const;

  const P2PConfig config_;
  std::vector<Buffer> buffers_;
  bool p2p_memcpy_enabled_ = false;

  // Cached connected components: maps each logical ID to the sorted list of
  // logical IDs in its connected component. Computed once at construction time.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> id_to_component_members_;
};

absl::Status RunCollectivePermute(P2PConfig::SourceTargetRanks source_target,
                                  const std::vector<DeviceBufferPair>& buffers,
                                  se::Stream& stream, Communicator& comm,
                                  absl::string_view device_string,
                                  int64_t current_id,
                                  bool use_symmetric_buffer = false);

//===----------------------------------------------------------------------===//
// Collective-permute communicating cliques helpers.
//===----------------------------------------------------------------------===//

// Computes connected components of the source-target pairs graph using
// Union-Find. Returns a map from component root to sorted member IDs. All IDs
// in [0, num_participants) are included; IDs not in any pair become singleton
// components.
absl::flat_hash_map<int64_t, std::vector<int64_t>>
SourceTargetConnectedComponents(
    int64_t num_participants,
    absl::Span<const std::pair<int64_t, int64_t>> source_target_pairs);

// Remaps source/target IDs from partition/replica space to communicator-local
// ranks. When communicators are scoped to connected components (subsets of
// devices), the partition/replica IDs used in HLO source_target_pairs don't
// correspond to NCCL ranks. This function translates them via:
//   logical_id -> GlobalDeviceId -> clique_key.rank().
absl::StatusOr<P2PConfig::SourceTargetRanks> RemapSourceTargetToCliqueRanks(
    const P2PConfig::SourceTargetMapEntry& source_target,
    const GpuCliqueKey& clique_key, const DeviceAssignment& device_assn,
    CollectiveOpGroupMode group_mode, GlobalDeviceId global_device_id);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_PERMUTE_THUNK_H_
