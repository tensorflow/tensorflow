/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.*/

#ifndef XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_THUNK_H_
#define XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_THUNK_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/stream.h"

namespace xla::gpu {

// A thunk that runs single-host collective operations in a single shot.
// Assumes multiple devices are present, but all on the same host.
// Basic mode of operation is as follows:
// Prepare:
// - Add the clique key to the required resources.
// Initialize:
// - Allocated local buffers and signal flags.
// ExecuteOnStream:
// - Rendezvous with all devices on the host and sort the allocated buffers.
// - Launch the kernel.
class CollectiveKernelThunk : public Thunk {
 public:
  CollectiveKernelThunk(ThunkInfo info, CollectiveConfig collective_config,
                        bool is_async, CollectiveThunk::Buffer buffer)
      : Thunk{Thunk::kCollectiveKernel, info},
        is_async_(is_async),
        collective_config_(std::move(collective_config)),
        buffer_(buffer) {}

  // The single host collective thunk actually requires a clique key.
  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) final;

  // Allocate buffers and events as needed for cross device communication.
  absl::Status Initialize(const InitializeParams& params) final;

  // Execute the kernel on all devices.
  absl::Status ExecuteOnStream(const ExecuteParams& params) final;

 private:
  // Internal method to sync thread after Initialize.
  absl::Status RendezvousAfterInit(const GpuCliqueKey& clique_key, RankId rank,
                                   int64_t num_ranks);

  // Whether the collective is run on an async stream.
  const bool is_async_;
  // Collective config being used. Copied over to avoid lifetime issues.
  const CollectiveConfig collective_config_;
  // Reference to the buffer related information required for the collective.
  CollectiveThunk::Buffer buffer_;

  // Per-executor state that needs to be synchronized for access.
  struct StreamState {
    RankId rank;
    // Buffers and signal flags allocated for the collective.
    se::DeviceMemoryHandle local_buffer;
    se::DeviceMemoryHandle signal_buffer;
  };
  // Guard access to the stream state across different threads (which control
  // different streams).
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, StreamState> per_stream_state_
      ABSL_GUARDED_BY(mutex_);

  // These vectors are merely pointers into the StreamState(s) above ordered by
  // RankId. They are initialized once at the end of Initialize() and never
  // changed.
  std::vector<se::DeviceMemoryBase> local_buffer_ptrs_;
  std::vector<se::DeviceMemoryBase> signal_buffer_ptrs_;
};
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_THUNK_H_
