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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
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
  ABSL_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));

  ABSL_ASSIGN_OR_RETURN(CudaDeviceAllocator::Options options,
                   QueryDeviceAllocatorOptions(device));
  return CreateWithDevice(executor, device, size, options);
}

absl::StatusOr<std::unique_ptr<CudaRawMemoryAllocation>>
CudaRawMemoryAllocation::Create(StreamExecutor* executor, uint64_t size,
                                const CudaDeviceAllocator::Options& options) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdevice device;
  ABSL_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));
  return CreateWithDevice(executor, device, size, options);
}

absl::StatusOr<std::unique_ptr<CudaRawMemoryAllocation>>
CudaRawMemoryAllocation::CreateWithDevice(
    StreamExecutor* executor, CUdevice device, uint64_t size,
    const CudaDeviceAllocator::Options& options) {
  if (!options.use_vmm) {
    return absl::InvalidArgumentError(
        "CudaRawMemoryAllocation requires CUDA VMM, but options.use_vmm is "
        "false");
  }

  // The granularity query itself can be rejected for unsupported handle
  // types; probe with the same fallback CudaDeviceAllocator uses and create
  // the allocation with the handle types the driver accepted.
  ABSL_ASSIGN_OR_RETURN(VmmGranularityProbe probe,
                   ProbeVmmGranularity(device, options));
  CUmemAllocationProp props = BuildVmmAllocationProp(device, probe.options);

  // Same effective alignment as CudaDeviceAllocator.
  size_t alignment = std::max(options.alignment, probe.granularity);
  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, alignment);

  // Shared fallback: FABRIC+POSIX_FD -> POSIX_FD -> NONE, including on
  // CUDA_ERROR_INVALID_VALUE from older drivers.
  ABSL_ASSIGN_OR_RETURN(CUmemGenericAllocationHandle handle,
                   CreateVmmPhysicalAllocation(props, padded_size));

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
