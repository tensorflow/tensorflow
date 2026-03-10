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

#include "xla/stream_executor/cuda/cuda_memory_reservation.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace stream_executor::gpu {
namespace {

CUmemAllocationProp BuildAllocationProperties(CUdevice device) {
  CUmemAllocationProp props = {};
  props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  props.location.id = device;
  props.requestedHandleTypes =
      static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_NONE);
  return props;
}

}  // namespace

absl::StatusOr<std::unique_ptr<CudaMemoryReservation>>
CudaMemoryReservation::Create(StreamExecutor* executor, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdevice device;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));

  CUmemAllocationProp props = BuildAllocationProperties(device);

  size_t granularity = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);

  CUdeviceptr ptr;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuMemAddressReserve(&ptr, padded_size, granularity, 0, 0)));

  return std::unique_ptr<CudaMemoryReservation>(
      new CudaMemoryReservation(executor, ptr, padded_size));
}

CudaMemoryReservation::CudaMemoryReservation(StreamExecutor* executor,
                                             CUdeviceptr ptr, uint64_t size)
    : executor_(executor), ptr_(ptr), size_(size) {}

DeviceAddressBase CudaMemoryReservation::address() const {
  return DeviceAddressBase(reinterpret_cast<void*>(ptr_), size_);
}

absl::Status CudaMemoryReservation::Map(size_t reservation_offset,
                                        size_t allocation_offset, size_t size,
                                        MemoryAllocation& allocation) {
  auto* cuda_alloc = dynamic_cast<CudaRawMemoryAllocation*>(&allocation);
  if (cuda_alloc == nullptr) {
    return absl::InvalidArgumentError(
        "CudaMemoryReservation::Map requires a CudaRawMemoryAllocation");
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return cuda::ToStatus(cuMemMap(ptr_ + reservation_offset, size,
                                 allocation_offset, cuda_alloc->GetHandle(),
                                 0));
}

absl::Status CudaMemoryReservation::SetAccess(uint64_t reservation_offset,
                                              size_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  CUmemAccessDesc desc = {};
  desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  desc.location.id = static_cast<int>(executor_->device_ordinal());
  desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return cuda::ToStatus(
      cuMemSetAccess(ptr_ + reservation_offset, size, &desc, 1));
}

absl::Status CudaMemoryReservation::UnMap(size_t offset, size_t size) {
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  return cuda::ToStatus(cuMemUnmap(ptr_ + offset, size));
}

CudaMemoryReservation::~CudaMemoryReservation() {
  if (ptr_ == 0) {
    return;
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  // Attempt to unmap the full range before freeing the virtual address space.
  // Sub-ranges already unmapped by ScopedMapping destructors will cause this
  // call to fail; the error is logged and the address range is freed anyway.
  auto unmap_status =
      cuda::ToStatus(cuMemUnmap(ptr_, size_), "Error unmapping CUDA memory");
  if (!unmap_status.ok()) {
    LOG(ERROR) << unmap_status.message();
  }
  auto free_status = cuda::ToStatus(cuMemAddressFree(ptr_, size_),
                                    "Error freeing CUDA address range");
  if (!free_status.ok()) {
    LOG(ERROR) << free_status.message();
  }
}

}  // namespace stream_executor::gpu
