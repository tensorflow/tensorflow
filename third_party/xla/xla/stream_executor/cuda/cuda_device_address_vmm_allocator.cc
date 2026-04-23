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

#include "xla/stream_executor/cuda/cuda_device_address_vmm_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/cuda/cuda_memory_reservation.h"
#include "xla/stream_executor/cuda/cuda_raw_memory_allocation.h"
#include "xla/stream_executor/cuda/cuda_status.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"
#include "xla/tsl/platform/statusor.h"

namespace stream_executor::gpu {

CudaDeviceAddressVmmAllocator::CudaDeviceAddressVmmAllocator(
    const Platform* platform)
    : DeviceAddressVmmAllocator(platform) {}

absl::StatusOr<std::unique_ptr<CudaDeviceAddressVmmAllocator>>
CudaDeviceAddressVmmAllocator::Create(const Platform* platform,
                                      absl::Span<const DeviceConfig> devices) {
  auto allocator =
      absl::WrapUnique(new CudaDeviceAddressVmmAllocator(platform));
  TF_RETURN_IF_ERROR(PopulateDevices(allocator.get(), devices));
  return allocator;
}

absl::StatusOr<std::unique_ptr<CudaDeviceAddressVmmAllocator>>
CudaDeviceAddressVmmAllocator::Create(StreamExecutor* executor, Stream* stream,
                                      uint64_t pa_budget) {
  return Create(executor->GetPlatform(),
                {{DeviceConfig{executor, stream, pa_budget}}});
}

absl::StatusOr<std::unique_ptr<CudaDeviceAddressVmmAllocator>>
CudaDeviceAddressVmmAllocator::Create(
    const Platform* platform, double memory_fraction,
    std::optional<int64_t> gpu_system_memory_size,
    absl::Span<const std::pair<StreamExecutor*, Stream*>> devices) {
  LOG(INFO) << "Using VMM (Virtual Memory Management) allocator.";
  std::vector<DeviceConfig> device_configs;
  device_configs.reserve(devices.size());
  for (const auto& [executor, stream] : devices) {
    int64_t free_memory;
    int64_t total_memory;
    if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
      return absl::UnavailableError(
          absl::StrFormat("Failed to query available memory from device %i",
                          executor->device_ordinal()));
    }
    // Calculate pa_budget the same way as BFCAllocator.
    uint64_t pa_budget = total_memory * memory_fraction;
    // If gpu_system_memory_size is set, use it instead.
    if (gpu_system_memory_size.has_value()) {
      pa_budget = gpu_system_memory_size.value();
    }
    LOG(INFO) << "VMM allocator pa_budget for device "
              << executor->device_ordinal() << ": " << pa_budget << " bytes.";
    device_configs.push_back({executor, stream, pa_budget});
  }
  return Create(platform, device_configs);
}

absl::Status CudaDeviceAddressVmmAllocator::InitializeDeviceState(
    PerDeviceState& state) {
  int ordinal = state.executor->device_ordinal();

  // Verify that the device supports 64-bit stream memory operations
  // (cuStreamWriteValue64), which requires compute capability >= 7.0.
  CUdevice cu_device;
  TF_RETURN_IF_ERROR(
      cuda::ToStatus(cuDeviceGet(&cu_device, ordinal), "cuDeviceGet"));
  int supported = 0;
  TF_RETURN_IF_ERROR(cuda::ToStatus(
      cuDeviceGetAttribute(&supported,
                           CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS,
                           cu_device),
      "cuDeviceGetAttribute"));
  if (!supported) {
    return absl::UnimplementedError(absl::StrFormat(
        "Device %d does not support 64-bit stream memory operations "
        "(cuStreamWriteValue64 requires compute capability >= 7.0). "
        "Query CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS returned "
        "false.",
        ordinal));
  }

  // Allocate one uint64_t of pinned host memory as the per-device timeline
  // counter, then obtain the device-side pointer used by
  // cuStreamWriteValue64. CU_MEMHOSTALLOC_PORTABLE makes it accessible from
  // all CUDA contexts (important for multi-device scenarios).
  void* host_ptr = nullptr;
  CUdeviceptr dev_ptr = 0;
  {
    std::unique_ptr<ActivateContext> activation = state.executor->Activate();
    TF_RETURN_IF_ERROR(cuda::ToStatus(
        cuMemHostAlloc(&host_ptr, sizeof(uint64_t), CU_MEMHOSTALLOC_PORTABLE),
        "cuMemHostAlloc for timeline counter"));
    *static_cast<volatile uint64_t*>(host_ptr) = 0;
    if (auto status =
            cuda::ToStatus(cuMemHostGetDevicePointer(&dev_ptr, host_ptr,
                                                     /*flags=*/0),
                           "cuMemHostGetDevicePointer");
        !status.ok()) {
      cuMemFreeHost(host_ptr);
      return status;
    }
  }

  CUmemAllocationProp alloc_props = {};
  alloc_props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  alloc_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  alloc_props.location.id = cu_device;
  alloc_props.requestedHandleTypes =
      static_cast<CUmemAllocationHandleType>(CU_MEM_HANDLE_TYPE_NONE);
  size_t granularity = 0;
  if (auto s = cuda::ToStatus(
          cuMemGetAllocationGranularity(&granularity, &alloc_props,
                                        CU_MEM_ALLOC_GRANULARITY_RECOMMENDED),
          "cuMemGetAllocationGranularity");
      !s.ok()) {
    LOG(ERROR) << "Failed to get allocation granularity for device " << ordinal
               << ": " << s;
  }

  state.allocation_granularity = static_cast<uint64_t>(granularity);
  state.pinned_timeline = static_cast<volatile uint64_t*>(host_ptr);
  state.timeline_dev_ptr = static_cast<uint64_t>(dev_ptr);

  // cuMemFreeHost is safe to call without context activation for
  // CU_MEMHOSTALLOC_PORTABLE memory.
  state.destroy_fn = [host_ptr, ordinal]() {
    auto status = cuda::ToStatus(cuMemFreeHost(host_ptr),
                                 "cuMemFreeHost for timeline counter");
    if (!status.ok()) {
      LOG(WARNING) << "Failed to free pinned timeline memory for device "
                   << ordinal << ": " << status;
    }
  };

  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<MemoryAllocation>>
CudaDeviceAddressVmmAllocator::CreateAllocation(StreamExecutor* executor,
                                                uint64_t size) {
  return CudaRawMemoryAllocation::Create(executor, size);
}

absl::StatusOr<std::unique_ptr<MemoryReservation>>
CudaDeviceAddressVmmAllocator::CreateReservation(StreamExecutor* executor,
                                                 uint64_t size) {
  return CudaMemoryReservation::Create(executor, size);
}

absl::Status CudaDeviceAddressVmmAllocator::EnqueueDeferredDeallocation(
    PerDeviceState& state, uint64_t seqno) {
  CUstream cu_stream =
      static_cast<CUstream>(state.stream->platform_specific_handle().stream);
  return cuda::ToStatus(
      cuStreamWriteValue64(cu_stream,
                           static_cast<CUdeviceptr>(state.timeline_dev_ptr),
                           seqno, /*flags=*/0),
      "cuStreamWriteValue64");
}

}  // namespace stream_executor::gpu
