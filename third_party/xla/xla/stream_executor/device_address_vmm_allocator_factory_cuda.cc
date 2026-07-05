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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor {

absl::StatusOr<std::unique_ptr<DeviceAddressVmmAllocator>>
DeviceAddressVmmAllocator::Create(
    const Platform* platform, double memory_fraction,
    std::optional<int64_t> gpu_system_memory_size,
    absl::Span<const std::pair<StreamExecutor*, Stream*>> devices) {
  ASSIGN_OR_RETURN(
      std::unique_ptr<gpu::CudaDeviceAddressVmmAllocator> allocator,
      gpu::CudaDeviceAddressVmmAllocator::Create(
          platform, memory_fraction, gpu_system_memory_size, devices));
  return std::unique_ptr<DeviceAddressVmmAllocator>(std::move(allocator));
}

}  // namespace stream_executor
