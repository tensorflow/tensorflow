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

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#include "xla/stream_executor/kernel.h"
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
                        absl::Span<const CollectiveThunk::Buffer> buffers,
                        bool is_collective_kernel_enabled,
                        absl::string_view kernel_name = "",
                        bool is_multimem_enabled = false)
      : Thunk{Thunk::kCollectiveKernel, info},
        collective_kernel_enabled_(is_collective_kernel_enabled),
        is_async_(is_async),
        collective_config_(std::move(collective_config)),
        reduction_kind_(reduction_kind),
        kernel_name_(kernel_name),
        buffers_(buffers),
        is_multimem_enabled_(is_multimem_enabled) {
    per_stream_state_.reserve(kMaxNumExecutors);
  }

  // Returns true if the collective kernel is supported for the given clique.
  absl::StatusOr<bool> IsSupported(
      const GpuCliqueKey& clique_key,
      const CollectiveCliques* collective_cliques) const;

  // The single host collective thunk actually requires a clique key.
  absl::Status Prepare(const PrepareParams& params,
                       ResourceRequestsInterface& resource_requests) final;

  // Allocate buffers and events as needed for cross device communication.
  // If InitializeParams contains a PTX kernel, it will be used instead of the
  // custom cuda kernel.
  absl::Status Initialize(const InitializeParams& params) final;

  // Execute the kernel on all devices.
  absl::Status ExecuteOnStream(const ExecuteParams& params) final;

 private:
  // We use a double buffering strategy for the buffers.
  // See docs on struct StreamState for more details.
  static constexpr int64_t kNumBuffers = 2;
  // Per-executor state that needs to be synchronized for access.
  struct StreamState {
    int device_ordinal;
    RankId rank;
    // Buffers and signal flags allocated for the collective.
    // Buffers are double buffered to allow for consecutive invocation
    // of the kernel on different GPUs.
    // - GPUs sync on Buffer 0 on first invocation.
    // - GPUs sync on Buffer 1 on second invocation.
    //   This implies that all GPUs must have finished the first invocation
    //   before they can sync on the second invocation.
    // - Alternate back to Buffer 0 on third invocation. And so on.
    se::DeviceMemoryHandle local_buffer;

    // Pointer to the collective kernel metadata on device.
    se::DeviceMemoryBase metadata;

    // These vectors are merely pointers into the buffer(s) above ordered
    // by RankId. They are initialized once at the end of Initialize() and never
    // changed.
    std::array<se::DeviceMemoryBase, kNumBuffers> remote_buffer_ptrs;
    std::array<se::DeviceMemoryBase, kNumBuffers> signal_buffer_ptrs;
    // Kernel entry for the stream executor.
    std::unique_ptr<se::Kernel> kernel;
    uint32_t invocation_count = 0;

    void* multicast_device_ptr = nullptr;

    // Constructor to make OSS builds happy.
    StreamState() = default;
    StreamState(int device_ordinal_arg, RankId rank_arg,
                se::DeviceMemoryHandle local_buffer_arg,
                std::unique_ptr<se::Kernel> kernel_arg)
        : device_ordinal(device_ordinal_arg),
          rank(rank_arg),
          local_buffer(std::move(local_buffer_arg)),
          kernel(std::move(kernel_arg)) {}
  };

  // Returns the input size in bytes for the collective.
  int64_t GetInputSizeBytes() const;

  // Internal method to sync thread after Initialize.
  // Returns the collective kernel metadata for the given clique key.
  absl::Status ExchangeStateMetadata(const GpuCliqueKey& clique_key,
                                     const InitializeParams& params,
                                     StreamState& state);

  // Initializes and multimem memory. Each thunk participant should call this
  // method once. Multimem should be setup before usage when multimem strategy
  // is selected.
  absl::Status SetupMultimem(const GpuCliqueKey& clique_key,
                             const se::StreamExecutor* stream_executor,
                             StreamState& state);

  // Whether the one-shot kernel is enabled.
  const bool collective_kernel_enabled_;
  // Whether the collective is run on an async stream.
  const bool is_async_;
  // Collective config being used. Copied over to avoid lifetime issues.
  const CollectiveConfig collective_config_;
  // Reduction kind being to use for AllReduce collective.
  const ReductionKind reduction_kind_;
  // Kernel name to execute. Required when Codegen/PTX kernel is used.
  // Must match the kernel name in the generated PTX kernel.
  const std::string kernel_name_;
  // Reference to the buffer related information required for the collective.
  absl::Span<const CollectiveThunk::Buffer> buffers_;

  std::unique_ptr<stream_executor::gpu::GpuExecutor::MulticastMemory>
      multicast_memory_;
  // Guard access to the stream state across different threads (which control
  // different streams).
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamState>>
      per_stream_state_ ABSL_GUARDED_BY(mutex_);
  const bool is_multimem_enabled_;
};
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_THUNK_H_
