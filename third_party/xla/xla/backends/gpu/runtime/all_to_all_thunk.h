/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/per_device_state.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct AllToAllConfig {
  CollectiveConfig config;
  bool has_split_dimension;
};

// Thunk that performs an All-to-All among CUDA GPU-based replicas.
class AllToAllThunk : public CollectiveThunk {
 public:
  AllToAllThunk(ThunkInfo thunk_info, const HloAllToAllInstruction* instr,
                std::vector<Buffer> buffers, bool p2p_memcpy_enabled,
                int devices_per_host);

  AllToAllThunk(ThunkInfo thunk_info, const AllToAllConfig& config,
                std::vector<CollectiveThunk::Buffer> buffers,
                bool p2p_memcpy_enabled, int devices_per_host);

  // Returns whether the given instruction can be lowered to an all-to-all
  // call.
  static absl::Status CheckImplementable(const HloAllToAllInstruction* instr,
                                         int64_t replica_count,
                                         int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static absl::string_view GetHloOpName() { return "all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloAllToAllInstruction* instr);

  static absl::StatusOr<std::unique_ptr<AllToAllThunk>> FromProto(
      ThunkInfo thunk_info, const AllToAllThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      int devices_per_host);

  absl::StatusOr<ThunkProto> ToProto() const override;

  const CollectiveConfig& config() const override { return config_.config; }
  bool has_split_dimension() const { return config_.has_split_dimension; }

 protected:
  // No rendezvous needed when using P2P memcpy in local mode instead of NCCL.
  bool RequiresRendezvous() const override { return !p2p_memcpy_enabled_; }

  absl::Status RunCollective(const ExecuteParams& params,
                             const GpuCliqueKey& clique_key, se::Stream& stream,
                             Communicator& comm) override;

  bool CanUseSymmetricBuffer() const override { return true; }

 private:
  struct DeviceState {
    // A uint64_t array of size num_devices. The array is used in each call to
    // RunCollective(), but is preallocated as CUDA host memory and written to
    // in the first call to Initialize(), since addresses won't change across
    // calls to RunCollective().
    std::unique_ptr<se::MemoryAllocation> receive_pointer_map;
    // Event to synchronize streams on different devices at the start/end of the
    // kernel.
    std::unique_ptr<se::Event> event;
    // Events for all ranks in the clique, populated once during Initialize().
    // Not internally synchronized; relies on host-side thunk initialization and
    // execution being serialized per device.
    std::vector<se::Event*> events;
  };

  const AllToAllConfig config_;
  bool p2p_memcpy_enabled_ = false;
  PerDeviceState<DeviceState> per_device_states_;
};

absl::Status RunAllToAll(bool has_split_dimension,
                         std::vector<DeviceBufferPair>& buffers,
                         se::Stream& stream, Communicator& comm,
                         bool use_symmetric_buffer = false);

absl::Status RunMemCpyAllToAll(bool has_split_dimension,
                               std::vector<DeviceBufferPair>& buffers,
                               se::Stream& stream, Communicator& comm,
                               uint64_t receive_pointer_map[],
                               const GpuCliqueKey& clique_key, RankId rank,
                               se::Event* event,
                               absl::Span<se::Event* const> events);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_ALL_TO_ALL_THUNK_H_
