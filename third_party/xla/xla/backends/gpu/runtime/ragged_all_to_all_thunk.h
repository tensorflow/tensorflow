/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {

struct RaggedAllToAllConfig {
  CollectiveConfig config;
  int64_t num_total_updates = 1;
  int64_t num_input_rows = 1;
  int64_t num_row_elements = 1;
};

// Thunk that performs a NCCL-based Ragged-All-to-All among CUDA GPU-based
// replicas.
class RaggedAllToAllStartThunk : public CollectiveThunk {
 public:
  RaggedAllToAllStartThunk(ThunkInfo thunk_info,
                           const HloRaggedAllToAllInstruction* instr,
                           std::vector<Buffer> buffers,
                           bool p2p_memcpy_enabled);
  RaggedAllToAllStartThunk(ThunkInfo thunk_info,
                           const RaggedAllToAllConfig& config,
                           std::shared_ptr<AsyncEvents> async_events,
                           std::vector<CollectiveThunk::Buffer> buffers,
                           bool one_shot_kernel_enabled);

  // Returns whether the given instruction can be lowered to a nccl
  // ragged-all-to-all call.
  static absl::Status CheckImplementable(
      const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static absl::string_view GetHloOpName() { return "ragged-all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloRaggedAllToAllInstruction* instr);

  const CollectiveConfig& config() const override { return config_.config; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

  static absl::StatusOr<std::unique_ptr<RaggedAllToAllStartThunk>> FromProto(
      ThunkInfo thunk_info, const RaggedAllToAllStartThunkProto& thunk_proto,
      absl::Span<const BufferAllocation> buffer_allocations,
      CollectiveThunk::AsyncEventsMap& async_events_map);

  absl::StatusOr<ThunkProto> ToProto() const override;

  BufferUses buffer_uses() const override {
    BufferUses uses;
    uses.reserve(buffers_.size() * 2);
    for (const Buffer& buffer : buffers_) {
      uses.push_back(BufferUse::Read(buffer.source_buffer.slice,
                                     buffer.source_buffer.shape));
      uses.push_back(BufferUse::Write(buffer.destination_buffer.slice,
                                      buffer.destination_buffer.shape));
    }
    return uses;
  }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     const GpuCliqueKey& clique_key,
                                     se::Stream& stream,
                                     Communicator& comm) override;

 private:
  // Contains the values that are passed between host threads with rendezvous.
  struct RendezvousValue {
    RankId rank;
    se::DeviceAddressBase output_buffer;
    se::Event* start_event = nullptr;
    se::Event* end_event = nullptr;

    bool operator<(const RendezvousValue& other) const {
      return rank < other.rank;
    }
  };

  struct StreamState {
    int device_ordinal;
    RankId rank;

    // Host memory allocations for ragged metadata.
    absl::InlinedVector<std::unique_ptr<se::MemoryAllocation>, 8>
        host_buffer_allocs;

    // Device memory buffer for output offsets.
    se::DeviceAddressHandle output_offsets_device_buffer;

    // Event to synchronize streams on different devices at the start of the
    // kernel.
    std::unique_ptr<se::Event> start_event;

    // Event to synchronize streams on different devices at the end of the
    // kernel.
    std::unique_ptr<se::Event> end_event;

    StreamState(int device_ordinal, RankId rank)
        : device_ordinal(device_ordinal), rank(rank) {}
  };

  // Executes the rendezvous before the kernel start.
  // Inserts CUDA events into the stream to ensure that all devices have reached
  // the start event before the kernel starts.
  absl::StatusOr<std::shared_ptr<std::vector<RendezvousValue>>>
  RendezvousBeforeKernelStart(const GpuCliqueKey& clique_key,
                              se::Stream& stream, const StreamState& state,
                              const se::DeviceAddressBase& output_buffer);

  // Executes the rendezvous after the kernel finish. Waits for all devices to
  // reach the end event.
  absl::Status RendezvousAfterKernelFinish(
      const GpuCliqueKey& clique_key, se::Stream& stream,
      const StreamState& state,
      const std::vector<RendezvousValue>& rendezvous_values);

  absl::Status RunOneShotRaggedAllToAll(
      const GpuCliqueKey& clique_key, se::Stream& stream,
      const StreamState& state, absl::Span<DeviceBufferPair const> buffers);

  bool is_local() const;

  const RaggedAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
  int64_t device_count_ = -1;
  const bool one_shot_kernel_enabled_;

  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamState>>
      per_stream_states_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
