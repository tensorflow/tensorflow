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

#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"

#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace stream_executor::gpu {

absl::StatusOr<std::unique_ptr<RocmRawMemoryAllocation>>
RocmRawMemoryAllocation::Create(StreamExecutor* executor, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  hipDevice_t device;
  RETURN_IF_ERROR(ToStatus(hipDeviceGet(&device, executor->device_ordinal())));

  hipMemAllocationProp props = {};
  props.type = hipMemAllocationTypePinned;
  props.location.type = hipMemLocationTypeDevice;
  props.location.id = device;
  props.requestedHandleTypes = hipMemHandleTypeNone;

  size_t granularity = 0;
  RETURN_IF_ERROR(ToStatus(hipMemGetAllocationGranularity(
      &granularity, &props, hipMemAllocationGranularityRecommended)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);

  hipMemGenericAllocationHandle_t handle;
  RETURN_IF_ERROR(ToStatus(hipMemCreate(&handle, padded_size, &props, 0ULL)));

  return std::unique_ptr<RocmRawMemoryAllocation>(
      new RocmRawMemoryAllocation(executor, handle, padded_size));
}

RocmRawMemoryAllocation::RocmRawMemoryAllocation(
    StreamExecutor* executor, hipMemGenericAllocationHandle_t handle,
    uint64_t size)
    : executor_(executor), handle_(handle), size_(size) {}

DeviceAddressBase RocmRawMemoryAllocation::address() const {
  // handle_ is an opaque allocation handle, not a device pointer. We expose it
  // as a DeviceAddressBase so the base class can use opaque() for identity
  // tracking; callers never dereference this address.
  return DeviceAddressBase(static_cast<void*>(handle_), size_);
}

RocmRawMemoryAllocation::~RocmRawMemoryAllocation() {
  if (handle_ == nullptr) {
    return;
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  auto status = ToStatus(hipMemRelease(handle_), "Error releasing ROCm memory");
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
}

}  // namespace stream_executor::gpu
