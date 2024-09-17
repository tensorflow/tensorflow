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

#include "xla/stream_executor/gpu/gpu_semaphore.h"

#include <utility>

#include "absl/status/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/statusor.h"

namespace stream_executor {
absl::StatusOr<GpuSemaphore> GpuSemaphore::Create(StreamExecutor* executor) {
  // Allocate the value in pinned host memory that can be read from both
  // host and device.
  TF_ASSIGN_OR_RETURN(auto alloc,
                      executor->HostMemoryAllocate(sizeof(GpuSemaphoreState)));
  return GpuSemaphore{std::move(alloc)};
}

DeviceMemory<GpuSemaphoreState> GpuSemaphore::device() {
  // This assumes unified addressing, as we do not explicitly translate the
  // host pointer into a device pointer.
  return DeviceMemory<GpuSemaphoreState>::MakeFromByteSize(
      ptr_->opaque(), sizeof(GpuSemaphoreState));
}
}  // namespace stream_executor
