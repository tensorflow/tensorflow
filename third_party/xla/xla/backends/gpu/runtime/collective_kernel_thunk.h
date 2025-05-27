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

#include <memory>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
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
  static constexpr auto kMaxNumExecutors =
      ::stream_executor::gpu::kMaxNumAllReduceInputPtrs;

  CollectiveKernelThunk(ThunkInfo info, CollectiveConfig collective_config,
                        ReductionKind reduction_kind, bool is_async,
                        CollectiveThunk::Buffer buffer)
      : Thunk{Thunk::kCollectiveKernel, info},
        is_async_(is_async),
        collective_config_(std::move(collective_config)),
        reduction_kind_(reduction_kind),
        buffer_(buffer) {
    per_stream_state_.reserve(kMaxNumExecutors);
  }

  // The single host collective thunk actually requires a clique key.
  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) final;

  // Allocate buffers and events as needed for cross device communication.
  absl::Status Initialize(const InitializeParams& params) final;

  // Execute the kernel on all devices.
  absl::Status ExecuteOnStream(const ExecuteParams& params) final;

 private:
  // Per-executor state that needs to be synchronized for access.
  struct StreamState {
    int device_ordinal;
    RankId rank;
    // Buffers and signal flags allocated for the collective.
    se::DeviceMemoryHandle local_buffer;
    se::DeviceMemoryHandle signal_buffer;
    // These vectors are merely pointers into the buffer(s) above ordered
    // by RankId. They are initialized once at the end of Initialize() and never
    // changed.
    absl::InlinedVector<se::DeviceMemoryBase, kMaxNumExecutors>
        remote_buffer_ptrs{};
    absl::InlinedVector<se::DeviceMemoryBase, kMaxNumExecutors>
        signal_buffer_ptrs{};

    // Constructor to make OSS builds happy.
    StreamState() = default;
    StreamState(int device_ordinal_arg, RankId rank_arg,
                se::DeviceMemoryHandle local_buffer_arg,
                se::DeviceMemoryHandle signal_buffer_arg)
        : device_ordinal(device_ordinal_arg),
          rank(rank_arg),
          local_buffer(std::move(local_buffer_arg)),
          signal_buffer(std::move(signal_buffer_arg)) {}
  };

  // Internal method to sync thread after Initialize.
  // Modifies the state to include pointers to all buffers in the clique.
  absl::Status RendezvousAfterInit(const GpuCliqueKey& clique_key,
                                   StreamState& state);

  // Whether the collective is run on an async stream.
  const bool is_async_;
  // Collective config being used. Copied over to avoid lifetime issues.
  const CollectiveConfig collective_config_;
  // Reduction kind being to use for AllReduce collective.
  ReductionKind reduction_kind_;
  // Reference to the buffer related information required for the collective.
  CollectiveThunk::Buffer buffer_;

  // Guard access to the stream state across different threads (which control
  // different streams).
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamState>>
      per_stream_state_ ABSL_GUARDED_BY(mutex_);
};
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_THUNK_H_
