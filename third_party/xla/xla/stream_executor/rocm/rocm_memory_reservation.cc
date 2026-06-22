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

#include "xla/stream_executor/rocm/rocm_memory_reservation.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/rocm/rocm_raw_memory_allocation.h"
#include "xla/stream_executor/rocm/rocm_status.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace stream_executor::gpu {

absl::StatusOr<std::unique_ptr<RocmMemoryReservation>>
RocmMemoryReservation::Create(StreamExecutor* executor, uint64_t size) {
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

  void* ptr = nullptr;
  RETURN_IF_ERROR(ToStatus(
      hipMemAddressReserve(&ptr, padded_size, granularity, nullptr, 0ULL)));

  return std::unique_ptr<RocmMemoryReservation>(new RocmMemoryReservation(
      executor, static_cast<char*>(ptr), padded_size));
}

RocmMemoryReservation::RocmMemoryReservation(StreamExecutor* executor,
                                             char* ptr, uint64_t size)
    : executor_(executor), ptr_(ptr), size_(size) {}

DeviceAddressBase RocmMemoryReservation::address() const {
  return DeviceAddressBase(ptr_, size_);
}

absl::Status RocmMemoryReservation::Map(size_t reservation_offset,
                                        size_t allocation_offset, size_t size,
                                        MemoryAllocation& allocation) {
  auto* rocm_alloc = dynamic_cast<RocmRawMemoryAllocation*>(&allocation);
  if (rocm_alloc == nullptr) {
    return absl::InvalidArgumentError(
        "RocmMemoryReservation::Map requires a RocmRawMemoryAllocation");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  absl::Status status =
      ToStatus(hipMemMap(ptr_ + reservation_offset, size, allocation_offset,
                         rocm_alloc->GetHandle(), 0ULL));
  if (status.ok()) {
    mapped_bytes_ += size;
  }
  return status;
}

absl::Status RocmMemoryReservation::SetAccess(uint64_t reservation_offset,
                                              size_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();

  int device_count = 0;
  RETURN_IF_ERROR(
      ToStatus(hipGetDeviceCount(&device_count), "hipGetDeviceCount"));

  for (int peer = 0; peer < device_count; ++peer) {
    if (peer != executor_->device_ordinal()) {
      auto peer_executor_or = const_cast<Platform*>(executor_->GetPlatform())
                                  ->ExecutorForDevice(peer);
      if (!peer_executor_or.ok() ||
          !executor_->CanEnablePeerAccessTo(peer_executor_or.value())) {
        continue;
      }
    }
    hipMemAccessDesc desc = {};
    desc.location.type = hipMemLocationTypeDevice;
    desc.location.id = peer;
    desc.flags = hipMemAccessFlagsProtReadWrite;
    RETURN_IF_ERROR(
        ToStatus(hipMemSetAccess(ptr_ + reservation_offset, size, &desc, 1),
                 "hipMemSetAccess for peer device"));
  }
  return absl::OkStatus();
}

absl::Status RocmMemoryReservation::UnMap(size_t offset, size_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  absl::Status status = ToStatus(hipMemUnmap(ptr_ + offset, size));
  if (status.ok()) {
    mapped_bytes_ = mapped_bytes_ >= size ? mapped_bytes_ - size : 0;
  }
  return status;
}

RocmMemoryReservation::~RocmMemoryReservation() {
  if (ptr_ == nullptr) {
    return;
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  // Only unmap if a mapping is still active. The ScopedMapping returned by
  // MapTo/Remap normally unmaps the range before this reservation is destroyed;
  // calling hipMemUnmap again on an already-unmapped range fails with
  // HIP_ERROR_InvalidValue. Tracking mapped_bytes_ avoids that redundant (and
  // very noisy) double-unmap at teardown.
  if (mapped_bytes_ > 0) {
    auto unmap_status =
        ToStatus(hipMemUnmap(ptr_, size_), "Error unmapping ROCm memory");
    if (!unmap_status.ok()) {
      LOG(ERROR) << unmap_status.message();
    }
  }
  auto free_status = ToStatus(hipMemAddressFree(ptr_, size_),
                              "Error freeing ROCm address range");
  if (!free_status.ok()) {
    LOG(ERROR) << free_status.message();
  }
}

}  // namespace stream_executor::gpu
