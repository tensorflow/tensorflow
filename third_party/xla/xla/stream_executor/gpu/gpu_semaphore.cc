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

#include <memory>
#include <utility>

#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {
GpuSemaphore GpuSemaphore::Create(
    std::unique_ptr<MemoryAllocation> allocation) {
  return GpuSemaphore{std::move(allocation)};
}

DeviceAddress<GpuSemaphoreState> GpuSemaphore::device() {
  // This assumes unified addressing, as we do not explicitly translate the
  // host pointer into a device pointer.
  return DeviceAddress<GpuSemaphoreState>::MakeFromByteSize(
      ptr_->address().opaque(), sizeof(GpuSemaphoreState));
}
}  // namespace stream_executor
