/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_PJRT_LINEARIZE_THROTTLER_H_
#define XLA_PJRT_LINEARIZE_THROTTLER_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "absl/functional/any_invocable.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/numa.h"

namespace xla {

// Abstraction for throttling and allocating staging buffers for linearizing and
// delinearizing buffers with backpressure.
class LinearizeThrottler {
 public:
  struct Options {
    size_t max_inflight_linearization_bytes = 1024ULL << 30;
    size_t max_inflight_delinearization_bytes = 1024ULL << 30;
  };

  LinearizeThrottler(HostMemoryAllocator* host_allocator,
                     AsyncWorkRunner* async_runner, Options options);

  ~LinearizeThrottler();

  // Creates a linearized buffer directly from host data.
  tsl::AsyncValueRef<PjRtStagingBuffer> CreateFromData(
      const void* data, size_t size,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer);

  // Allocates a staging buffer with the given size, backpressure starts upon
  // invocation until the end of life of the returned staging buffer. This
  // method can not overdraft the delinearization capacity and it never blocks.
  // If the allocation cannot be fulified immediately, the returned
  // staging buffer will be unconstructed.
  tsl::AsyncValueRef<PjRtStagingBuffer> AllocateForDelinearizationAsync(
      size_t size, int numa_node, std::optional<LocalDeviceId> local_device_id);

  // Runs `delinearize_fn` inline while holding `size` bytes of delinearization
  // capacity. This method can overdraft the delinearization capacity of other
  // delinearization tasks and it never blocks.
  absl::Status ThrottleSyncDelinearize(
      size_t size, absl::FunctionRef<absl::Status()> delinearize_fn);

  // Allocates a destination buffer for linearizing into which may be allocated
  // from pinned memory or reuse an existing buffer. If sync is true, the buffer
  // will be allocated immediately.
  tsl::AsyncValueRef<PjRtStagingBuffer> AllocateStagingDest(
      bool sync, size_t size, PjRtRawBufferRef reused_buffer,
      int numa_node = tsl::port::kNUMANoAffinity,
      std::optional<LocalDeviceId> local_device_id = std::nullopt);

 private:
  class Throttler;
  class LinearizedBuffer;
  class LazyLinearizedBuffer;
  HostMemoryAllocator* const host_allocator_;
  std::unique_ptr<Throttler> linearize_throttler_;
  std::unique_ptr<Throttler> delinearize_throttler_;
};

}  // namespace xla

#endif  // XLA_PJRT_LINEARIZE_THROTTLER_H_
