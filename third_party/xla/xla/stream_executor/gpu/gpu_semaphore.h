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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_SEMAPHORE_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_SEMAPHORE_H_

#include <memory>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {
enum struct GpuSemaphoreState { kHold, kRelease, kTimedOut };

// A basic semaphore that allows synchronization between host and GPU.
// It uses pinned host memory as the communication channel.
class GpuSemaphore {
 public:
  // Creates an invalid semaphore instance
  GpuSemaphore() = default;

  // Creates a valid semaphore. Allocates some pinned host memory using
  // `executor`.
  static absl::StatusOr<GpuSemaphore> Create(StreamExecutor* executor);

  // Returns true if this semaphore is valid, otherwise false.
  explicit operator bool() const { return bool{ptr_}; }

  GpuSemaphoreState& operator*() {
    return *static_cast<GpuSemaphoreState*>(ptr_->opaque());
  }
  DeviceMemory<GpuSemaphoreState> device();

 private:
  explicit GpuSemaphore(std::unique_ptr<MemoryAllocation> alloc)
      : ptr_{std::move(alloc)} {}
  std::unique_ptr<MemoryAllocation> ptr_;
};
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_SEMAPHORE_H_
