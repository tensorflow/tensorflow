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
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/hlo/ir/collective_op_group_mode.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream.h"

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

  // Returns whether the given instruction can be lowered to a nccl
  // ragged-all-to-all call.
  static absl::Status CheckImplementable(
      const HloRaggedAllToAllInstruction* instr, int64_t replica_count,
      int64_t partition_count);

  absl::Status Initialize(const InitializeParams& params) override;

  static const char* GetHloOpName() { return "ragged-all-to-all-start"; }

  static CollectiveOpGroupMode GetGroupMode(
      const HloRaggedAllToAllInstruction* instr);

  const CollectiveConfig& config() const override { return config_.config; }
  absl::Span<const Buffer> buffers() const { return buffers_; }

 protected:
  absl::StatusOr<bool> RunCollective(const ExecuteParams& params,
                                     se::Stream& stream,
                                     CommunicatorHandle comm) override;

 private:
  struct StreamState {
    int device_ordinal;
    RankId rank;

    // Host memory allocations for ragged metadata.
    absl::InlinedVector<std::unique_ptr<se::MemoryAllocation>, 8>
        host_buffer_allocs;

    // Device memory buffer for output offsets.
    se::DeviceMemoryHandle output_offsets_device_buffer;

    // Event to synchronize streams on different devices at the start of the
    // kernel.
    std::unique_ptr<se::Event> start_event;

    // Event to synchronize streams on different devices at the end of the
    // kernel.
    std::unique_ptr<se::Event> end_event;

    StreamState(int device_ordinal, RankId rank)
        : device_ordinal(device_ordinal), rank(rank) {}
  };

  absl::Status RunMemCpyRaggedAllToAll(
      const GpuCliqueKey& clique_key, se::Stream& stream,
      const StreamState& state, absl::Span<DeviceBufferPair const> buffers,
      absl::Span<int64_t* const> ragged_metadata_allocs);

  absl::Status RunOneShotRaggedAllToAll(
      const GpuCliqueKey& clique_key, se::Stream& stream,
      const StreamState& state, absl::Span<DeviceBufferPair const> buffers);

  bool is_local() const;
  bool should_use_memcpy() const { return p2p_memcpy_enabled_ && is_local(); }

  const RaggedAllToAllConfig config_;
  const std::vector<Buffer> buffers_;
  int64_t device_count_ = -1;
  const bool p2p_memcpy_enabled_;
  const bool one_shot_kernel_enabled_;

  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamState>>
      per_stream_states_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_RAGGED_ALL_TO_ALL_THUNK_H_
