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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>

#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/tsl/platform/status_macros.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace stream_executor::gpu {

static CUmemAllocationProp GetAllocationProp(
    CUdevice device, const CudaVmmAllocator::Options& options) {
  CUmemAllocationProp properties = {};
  properties.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  properties.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  properties.location.id = device;
  properties.allocFlags.gpuDirectRDMACapable = options.enable_rdma ? 1 : 0;

  int handle_types = CU_MEM_HANDLE_TYPE_NONE;
  if (options.enable_posix_fd_handle) {
    handle_types |= CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }
  if (options.enable_fabric_handle) {
    handle_types |= CU_MEM_HANDLE_TYPE_FABRIC;
  }
  properties.requestedHandleTypes =
      static_cast<CUmemAllocationHandleType>(handle_types);
  return properties;
}

// Creates a physical VMM allocation. Tries cuMemCreate with the given
// properties and falls back through progressively simpler handle types:
//   FABRIC+POSIX_FD -> POSIX_FD -> NONE
static absl::StatusOr<CUmemGenericAllocationHandle> CreatePhysicalAllocation(
    CUmemAllocationProp properties, uint64_t padded_size) {
  CUmemGenericAllocationHandle handle;

  auto try_create = [&](const char* description)
      -> std::optional<absl::StatusOr<CUmemGenericAllocationHandle>> {
    CUresult result = cuMemCreate(&handle, padded_size, &properties, 0);
    if (result == CUDA_SUCCESS) {
      return handle;
    }
    if (result == CUDA_ERROR_NOT_PERMITTED ||
        result == CUDA_ERROR_NOT_SUPPORTED ||
        result == CUDA_ERROR_INVALID_VALUE) {
      XLA_LOG_DEVICE(WARNING, properties.location.id)
          << "VMM cuMemCreate with " << description
          << " handle types failed: " << cuda::ToStatus(result)
          << "; will retry with simpler handle types.";
      return std::nullopt;
    }
    return cuda::ToStatus(result);
  };

  bool has_fabric = properties.requestedHandleTypes & CU_MEM_HANDLE_TYPE_FABRIC;
  bool has_posix_fd = properties.requestedHandleTypes &
                      CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  if (has_fabric) {
    if (auto r = try_create("FABRIC+POSIX_FD")) {
      return *r;
    }
    properties.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(
        properties.requestedHandleTypes & ~CU_MEM_HANDLE_TYPE_FABRIC);
  }

  if (has_posix_fd) {
    if (auto r = try_create("POSIX_FD")) {
      return *r;
    }
    properties.requestedHandleTypes =
        static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_NONE);
  }

  RETURN_IF_ERROR(
      cuda::ToStatus(cuMemCreate(&handle, padded_size, &properties, 0)));
  return handle;
}

static CUmemAccessDesc GetAccessDesc(int device) {
  CUmemAccessDesc descriptor = {};
  descriptor.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  descriptor.location.id = device;
  descriptor.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  return descriptor;
}

// Allocates device memory using CUDA Virtual Memory Management (VMM) API.
// Returns (virtual_address, padded_size, allocation_handle).
static absl::StatusOr<std::tuple<void*, uint64_t, CUmemGenericAllocationHandle>>
VmmAllocate(StreamExecutor* executor, const CudaVmmAllocator::Options& options,
            uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdevice device;
  RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));

  // Query VMM allocation granularity and pad size to alignment boundary.
  CUmemAllocationProp properties = GetAllocationProp(device, options);
  size_t granularity = 0;
  RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &properties, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  size_t effective_alignment = std::max(options.alignment, granularity);
  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, effective_alignment);

  ASSIGN_OR_RETURN(CUmemGenericAllocationHandle handle,
                   CreatePhysicalAllocation(properties, padded_size));

  absl::Cleanup release_handle = [&] {
    absl::Status status = cuda::ToStatus(cuMemRelease(handle));
    if (!status.ok()) {
      XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
          << "Failed to release VMM handle during cleanup: " << status;
    }
  };

  // Reserve virtual address range and map the physical allocation into it.
  CUdeviceptr ptr;
  RETURN_IF_ERROR(cuda::ToStatus(
      cuMemAddressReserve(&ptr, padded_size, effective_alignment, 0, 0)));
  absl::Cleanup free_address = [&] {
    absl::Status status = cuda::ToStatus(cuMemUnmap(ptr, padded_size));
    if (!status.ok()) {
      XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
          << "Failed to unmap VMM memory during cleanup: " << status;
    }
    status = cuda::ToStatus(cuMemAddressFree(ptr, padded_size));
    if (!status.ok()) {
      XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
          << "Failed to free VMM address during cleanup: " << status;
    }
  };
  RETURN_IF_ERROR(cuda::ToStatus(cuMemMap(ptr, padded_size, 0, handle, 0)));

  // Grant read/write access — to all peers if peer access is enabled,
  // otherwise only to the owning device.
  if (options.enable_peer_access) {
    int device_count = 0;
    RETURN_IF_ERROR(cuda::ToStatus(cudaGetDeviceCount(&device_count)));
    for (int peer = 0; peer < device_count; peer++) {
      if (peer == executor->device_ordinal() ||
          executor->CanEnablePeerAccessTo(peer)) {
        XLA_VLOG_DEVICE(5, executor->device_ordinal())
            << "Setting VMM access for peer device " << peer;
        CUmemAccessDesc access_desc = GetAccessDesc(peer);
        RETURN_IF_ERROR(
            cuda::ToStatus(cuMemSetAccess(ptr, padded_size, &access_desc, 1)));
      }
    }
  } else {
    CUmemAccessDesc access_desc = GetAccessDesc(executor->device_ordinal());
    RETURN_IF_ERROR(
        cuda::ToStatus(cuMemSetAccess(ptr, padded_size, &access_desc, 1)));
  }

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "Allocated ptr=" << absl::bit_cast<void*>(ptr)
      << " requested size: " << size << " padded size: " << padded_size
      << " granularity: " << granularity
      << " effective alignment: " << effective_alignment;

  // Success — cancel cleanups and return ownership to caller.
  std::move(release_handle).Cancel();
  std::move(free_address).Cancel();
  return std::make_tuple(absl::bit_cast<void*>(ptr), padded_size, handle);
}

static void VmmDeallocate(StreamExecutor* executor, void* ptr,
                          uint64_t padded_size,
                          CUmemGenericAllocationHandle handle) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  XLA_VLOG_DEVICE(3, executor->device_ordinal())
      << "Deallocating " << ptr << " padded size: " << padded_size;

  // Unlike cuMemFree which defers until in-flight kernels complete, cuMemUnmap
  // immediately invalidates the virtual address mapping. Synchronize to ensure
  // no kernels are still accessing this memory. This is fine, because we never
  // use this allocator on a hot path and always wrap it into BFCAllocator that
  // does arena-based allocation.
  absl::Status status = cuda::ToStatus(cuCtxSynchronize());
  if (!status.ok()) {
    XLA_LOG_DEVICE(ERROR, executor->device_ordinal())
        << "Failed to synchronize before VMM deallocation at " << ptr << ": "
        << status;
  }

  CUdeviceptr device_ptr = absl::bit_cast<CUdeviceptr>(ptr);
  status = cuda::ToStatus(cuMemUnmap(device_ptr, padded_size));
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
    return DeviceAddressBase(ptr_, padded_size_);
  }

  std::string ToString() const final {
    return absl::StrFormat(
        "CudaVmmMemoryAllocation[device=%d, ptr=%p, size=%d, "
        "padded_size=%d, handle=%llu]",
        executor_->device_ordinal(), ptr_, requested_size_, padded_size_,
        handle_);
  }

 private:
  StreamExecutor* executor_;
  void* ptr_;
  uint64_t requested_size_;
  uint64_t padded_size_;
  CUmemGenericAllocationHandle handle_;
};

}  // namespace

CudaVmmAllocator::CudaVmmAllocator(StreamExecutor* executor, Options options)
    : executor_(executor), options_(options) {}

absl::StatusOr<std::unique_ptr<MemoryAllocation>> CudaVmmAllocator::Allocate(
    uint64_t size) {
  if (size == 0) {
    return std::make_unique<CudaVmmMemoryAllocation>(executor_, nullptr, 0, 0,
                                                     0);
  }

  ASSIGN_OR_RETURN(auto result, VmmAllocate(executor_, options_, size));
  auto [ptr, padded_size, handle] = result;

  return std::make_unique<CudaVmmMemoryAllocation>(executor_, ptr, size,
                                                   padded_size, handle);
}

}  // namespace stream_executor::gpu
