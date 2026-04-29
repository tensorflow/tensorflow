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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/collective_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/core/collectives/reduction_kind.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_handle.h"
#include "xla/stream_executor/gpu/all_reduce_kernel.h"
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
// When codegen kernel is used, kernel_name, launch_dimensions, shmem_bytes
// must be set.
class CollectiveKernelThunk : public Thunk {
 public:
  static constexpr auto kMaxNumExecutors =
      ::stream_executor::gpu::kMaxNumAllReduceInputPtrs;

  CollectiveKernelThunk(
      ThunkInfo info, CollectiveConfig collective_config,                //
      ReductionKind reduction_kind,                                      //
      bool is_async,                                                     //
      std::vector<CollectiveThunk::Buffer> buffers,                      //
      bool is_collective_kernel_enabled,                                 //
      absl::string_view kernel_name = "",                                //
      std::optional<LaunchDimensions> launch_dimensions = std::nullopt,  //
      int32_t shmem_bytes = 0,                                           //
      bool is_multimem_enabled = false)
      : Thunk{Thunk::kCollectiveKernel, info},
        collective_kernel_enabled_(is_collective_kernel_enabled),
        is_async_(is_async),
        collective_config_(std::move(collective_config)),
        reduction_kind_(reduction_kind),
        launch_dimensions_(launch_dimensions),
        kernel_name_(kernel_name),
        shmem_bytes_(shmem_bytes),
        buffers_(std::move(buffers)),
        is_multimem_enabled_(is_multimem_enabled) {
    per_stream_state_.reserve(kMaxNumExecutors);
  }

  bool is_multimem_enabled() const { return is_multimem_enabled_; }

  int32_t shmem_bytes() const { return shmem_bytes_; }

  absl::string_view kernel_name() const { return kernel_name_; }

  bool collective_kernel_enabled() const { return collective_kernel_enabled_; }
  bool is_async() const { return is_async_; }
  std::optional<LaunchDimensions> launch_dimensions() const {
    return launch_dimensions_;
  }

  // Returns true if the collective kernel is supported for the given clique.
  absl::StatusOr<bool> IsSupported(
      const GpuCliqueKey& clique_key, se::StreamExecutor& executor,
      const CollectiveParams& collective_params) const;

  // The single host collective thunk actually requires a clique key.
  absl::Status Prepare(const PrepareParams& params) final;

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

  // Per-executor scratch memory.
  struct StreamMemory {
    // Buffers allocated for the collective.
    // Buffers are double buffered to allow for consecutive invocation
    // of the kernel on different GPUs.
    // - GPUs sync on Buffer 0 on first invocation.
    // - GPUs sync on Buffer 1 on second invocation.
    //   This implies that all GPUs must have finished the first invocation
    //   before they can sync on the second invocation.
    // - Alternate back to Buffer 0 on third invocation. And so on.
    se::DeviceAddressHandle local_buffers_handle;

    // Signal buffers allocated for the collective.
    // Also double buffered for the same reason as local buffers.
    se::DeviceAddressHandle signal_buffers_handle;

    se::gpu::AllReduceStrategy strategy;

    const int64_t local_buffer_size_bytes = 0;
    const int64_t signal_buffer_size_bytes = 0;
  };

  // Per-executor state that needs to be synchronized for access.
  struct StreamState {
    int device_ordinal = 0;
    RankId rank = RankId(0);

    // Pointer to the collective kernel metadata on device.
    se::DeviceAddressBase metadata;

    // These vectors are merely pointers into the buffer(s) above ordered
    // by RankId. They are initialized once at the end of Initialize() and never
    // changed.
    std::array<se::DeviceAddressBase, kNumBuffers> remote_buffer_ptrs;
    std::array<se::DeviceAddressBase, kNumBuffers> signal_buffer_ptrs;
    // Kernel entry for the stream executor.
    std::unique_ptr<se::Kernel> kernel;
    uint32_t invocation_count = 0;

    // Constructor to make OSS builds happy.
    StreamState() = default;
    StreamState(int device_ordinal_arg, RankId rank_arg,
                std::unique_ptr<se::Kernel> kernel_arg)
        : device_ordinal(device_ordinal_arg),
          rank(rank_arg),
          kernel(std::move(kernel_arg)) {}
  };

  // Returns the input size in bytes for the collective.
  int64_t GetInputSizeBytes() const;

  // Calculate the device memory base for the given parameter index.
  // The size of the returned memory is num_devices pointers.
  static absl::StatusOr<se::DeviceAddressBase> GetParameterDeviceMemoryBase(
      se::DeviceAddressBase metadata, int64_t num_parameters,
      int64_t num_devices, int64_t parameter_index);

  // Whether the one-shot kernel is enabled.
  const bool collective_kernel_enabled_;
  // Whether the collective is run on an async stream.
  const bool is_async_;
  // Collective config being used. Copied over to avoid lifetime issues.
  const CollectiveConfig collective_config_;
  // Reduction kind being to use for AllReduce collective.
  const ReductionKind reduction_kind_;
  // Launch dimensions for the kernel. Only relevant when the codegen kernel
  // is used.
  std::optional<LaunchDimensions> launch_dimensions_;
  // Kernel name to execute. Required when Codegen/PTX kernel is used.
  // Must match the kernel name in the generated PTX kernel.
  const std::string kernel_name_;
  // Number of bytes of shared memory used by the kernel.
  // Only useful when the codegen kernel is used.
  const int32_t shmem_bytes_;
  // Reference to the buffer related information required for the collective.
  std::vector<CollectiveThunk::Buffer> buffers_;

  // Guard access to the stream state across different threads (which control
  // different streams).
  absl::Mutex mutex_;
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamState>>
      per_stream_state_ ABSL_GUARDED_BY(mutex_);
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamMemory>>
      per_stream_memory_ ABSL_GUARDED_BY(mutex_);
  const bool is_multimem_enabled_;
};
}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COLLECTIVE_KERNEL_THUNK_H_
