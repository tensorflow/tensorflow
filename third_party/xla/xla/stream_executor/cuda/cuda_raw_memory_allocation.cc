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

#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_device_allocator.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"

namespace stream_executor::gpu {

absl::StatusOr<std::unique_ptr<CudaRawMemoryAllocation>>
CudaRawMemoryAllocation::Create(StreamExecutor* executor, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdevice device;
  RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));

  ASSIGN_OR_RETURN(CudaDeviceAllocator::Options options,
                   QueryDeviceAllocatorOptions(device));
  CUmemAllocationProp props = BuildVmmAllocationProp(device, options);

  size_t granularity = 0;
  RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);

  CUmemGenericAllocationHandle handle;
  CUresult err = cuMemCreate(&handle, padded_size, &props, 0);
#if CUDA_VERSION >= 12030
  // Strip FABRIC and retry on NOT_PERMITTED/NOT_SUPPORTED.
  if ((err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) &&
      (static_cast<int>(props.requestedHandleTypes) &
       CU_MEM_HANDLE_TYPE_FABRIC)) {
    props.requestedHandleTypes = static_cast<CUmemAllocationHandleType>(
        static_cast<int>(props.requestedHandleTypes) &
        ~CU_MEM_HANDLE_TYPE_FABRIC);
    err = cuMemCreate(&handle, padded_size, &props, 0);
  }
#endif
  RETURN_IF_ERROR(cuda::ToStatus(err, "cuMemCreate"));

  return std::unique_ptr<CudaRawMemoryAllocation>(
      new CudaRawMemoryAllocation(executor, handle, padded_size));
}

CudaRawMemoryAllocation::CudaRawMemoryAllocation(
    StreamExecutor* executor, CUmemGenericAllocationHandle handle,
    uint64_t size)
    : executor_(executor), handle_(handle), size_(size) {}

DeviceAddressBase CudaRawMemoryAllocation::address() const {
  return DeviceAddressBase(
      reinterpret_cast<void*>(static_cast<uintptr_t>(handle_)), size_);
}

CudaRawMemoryAllocation::~CudaRawMemoryAllocation() {
  if (handle_ == 0) {
    return;
  }
  std::unique_ptr<ActivateContext> activation = executor_->Activate();
  auto status =
      cuda::ToStatus(cuMemRelease(handle_), "Error releasing CUDA memory");
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
}

}  // namespace stream_executor::gpu
