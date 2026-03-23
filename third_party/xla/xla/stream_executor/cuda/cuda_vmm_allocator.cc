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

#include "xla/stream_executor/cuda/cuda_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/numbers.h"
#include "xla/tsl/platform/status_macros.h"

namespace stream_executor::gpu {

static CUmemAllocationProp GetVmmAllocationProperties(CUdevice device,
                                                      bool is_rdma_supported) {
  CUmemAllocationProp properties = {};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.requestedHandleTypes =
      static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_NONE);
  properties.location.id = device;
  properties.allocFlags.gpuDirectRDMACapable = is_rdma_supported ? 1 : 0;
  return properties;
}

static CUmemAccessDesc GetVmmAccessDescriptor(int device) {
  CUmemAccessDesc descriptor = {};
  descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  descriptor.location.id = device;
  descriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return descriptor;
}

// Allocates device memory using CUDA VMM APIs. Returns the mapped pointer,
// the padded size, and the allocation handle.
static absl::StatusOr<std::tuple<void*, uint64_t, CUmemGenericAllocationHandle>>
VmmAllocate(StreamExecutor* executor, bool is_rdma_supported, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdevice device;
  RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));

  CUmemAllocationProp properties =
      GetVmmAllocationProperties(device, is_rdma_supported);
  size_t granularity = 0;
  RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);
  CUmemGenericAllocationHandle handle;

  // Create physical memory allocation.
  RETURN_IF_ERROR(
      cuda::ToStatus(cuMemCreate(&handle, padded_size, &properties, 0)));

  // Reserve and map virtual address space.
  CUdeviceptr ptr;
  absl::Status status =
      cuda::ToStatus(cuMemAddressReserve(&ptr, padded_size, granularity, 0, 0));
  if (!status.ok()) {
    // Clean up the physical allocation on failure.
    cuda::ToStatus(cuMemRelease(handle)).IgnoreError();
    return status;
  }

  status = cuda::ToStatus(cuMemMap(ptr, padded_size, 0, handle, 0));
  if (!status.ok()) {
    cuda::ToStatus(cuMemAddressFree(ptr, padded_size)).IgnoreError();
    cuda::ToStatus(cuMemRelease(handle)).IgnoreError();
    return status;
  }

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "VMM allocated " << ptr << " requested size: " << size
      << " padded size: " << padded_size << " granularity: " << granularity;

  // Set access for this device and all peers.
  int device_count = 0;
  RETURN_IF_ERROR(cuda::ToStatus(cudaGetDeviceCount(&device_count)));
  for (int peer = 0; peer < device_count; peer++) {
    if (peer == executor->device_ordinal() ||
        executor->CanEnablePeerAccessTo(peer)) {
      CUmemAccessDesc access_desc = GetVmmAccessDescriptor(peer);
      RETURN_IF_ERROR(
          cuda::ToStatus(cuMemSetAccess(ptr, padded_size, &access_desc, 1)));
    }
  }

  return std::make_tuple(reinterpret_cast<void*>(ptr), padded_size, handle);
}

// Frees VMM memory previously allocated by VmmAllocate.
static void VmmDeallocate(StreamExecutor* executor, void* ptr,
                          uint64_t padded_size,
                          CUmemGenericAllocationHandle handle) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdeviceptr device_ptr = reinterpret_cast<CUdeviceptr>(ptr);

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "VMM deallocating " << ptr << " padded size: " << padded_size;

  absl::Status status = cuda::ToStatus(cuMemUnmap(device_ptr, padded_size));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
        << "Failed to unmap VMM memory at " << ptr << ": " << status;
  }
  status = cuda::ToStatus(cuMemRelease(handle));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
        << "Failed to release VMM handle for " << ptr << ": " << status;
  }
  status = cuda::ToStatus(cuMemAddressFree(device_ptr, padded_size));
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
        << "Failed to free VMM address at " << ptr << ": " << status;
  }
}

namespace {

// A VMM memory allocation backed by CUDA VMM APIs.
class CudaVmmMemoryAllocation : public MemoryAllocation {
 public:
  CudaVmmMemoryAllocation(StreamExecutor* executor, void* ptr,
                          uint64_t requested_size, uint64_t padded_size,
                          CUmemGenericAllocationHandle handle)
      : executor_(executor),
        ptr_(ptr),
        requested_size_(requested_size),
        padded_size_(padded_size),
        handle_(handle) {}

  ~CudaVmmMemoryAllocation() final {
    if (ptr_ != nullptr) {
      VmmDeallocate(executor_, ptr_, padded_size_, handle_);
    }
  }

  DeviceAddressBase address() const final {
    return DeviceAddressBase(ptr_, requested_size_);
  }

  std::string ToString() const final {
    return absl::StrFormat(
        "CudaVmmMemoryAllocation[device=%d, ptr=%p, size=%d, padded_size=%d]",
        executor_->device_ordinal(), ptr_, requested_size_, padded_size_);
  }

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t requested_size_;
  uint64_t padded_size_;
  CUmemGenericAllocationHandle handle_;
};

}  // namespace

CudaVmmAllocator::CudaVmmAllocator(StreamExecutor* executor,
                                   bool is_rdma_supported)
    : executor_(executor), is_rdma_supported_(is_rdma_supported) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> CudaVmmAllocator::Allocate(
    uint64_t size) {
  if (size == 0) {
    return std::make_unique<CudaVmmMemoryAllocation>(executor_, nullptr, 0, 0,
                                                     0);
  }

  ASSIGN_OR_RETURN(auto result,
                   VmmAllocate(executor_, is_rdma_supported_, size));
  auto [ptr, padded_size, handle] = result;

  return std::make_unique<CudaVmmMemoryAllocation>(executor_, ptr, size,
                                                   padded_size, handle);
}

}  // namespace stream_executor::gpu
