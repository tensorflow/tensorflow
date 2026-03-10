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

#include <cstdint>
#include <memory>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
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

absl::StatusOr<std::unique_ptr<CudaRawMemoryAllocation>>
CudaRawMemoryAllocation::Create(StreamExecutor* executor, uint64_t size) {
  std::unique_ptr<ActivateContext> activation = executor->Activate();

  CUdevice device;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&device, executor->device_ordinal())));

  CUmemAllocationProp props = BuildAllocationProperties(device);

  size_t granularity = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(cuMemGetAllocationGranularity(
      &granularity, &props, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)));

  uint64_t padded_size = xla::RoundUpTo<uint64_t>(size, granularity);

  CUmemGenericAllocationHandle handle;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuMemCreate(&handle, padded_size, &props, 0)));

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
