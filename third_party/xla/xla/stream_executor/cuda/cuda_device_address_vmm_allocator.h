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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ADDRESS_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ADDRESS_VMM_ALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/vmm_device_address_allocator.h"

namespace stream_executor::gpu {

// CUDA implementation of DeviceAddressVmmAllocator.
//
// Uses cuMemCreate/cuMemAddressReserve for physical and virtual memory
// management, and cuStreamWriteValue64 for GPU timeline-based deferred
// deallocation. Requires compute capability >= 7.0 (Volta and later) for
// cuStreamWriteValue64 support.
//
// Use the Create() factories to obtain an instance; they return an error if
// the device does not meet the compute capability requirement.
class CudaDeviceAddressVmmAllocator : public DeviceAddressVmmAllocator {
 public:
  // Creates an allocator supporting multiple devices.
  //
  // Returns an error if any device does not support cuStreamWriteValue64
  // (compute capability < 7.0).
  //
  // Precondition: all entries in `devices` have distinct device ordinals.
  static absl::StatusOr<std::unique_ptr<CudaDeviceAddressVmmAllocator>> Create(
      const Platform* platform, absl::Span<const DeviceConfig> devices);

  // Creates an allocator supporting multiple devices, computing the pa_budget
  // for each device by querying DeviceMemoryUsage and applying memory_fraction.
  // If gpu_system_memory_size is set, it overrides the memory_fraction budget.
  //
  // Precondition: all entries in `devices` have distinct device ordinals.
  static absl::StatusOr<std::unique_ptr<CudaDeviceAddressVmmAllocator>> Create(
      const Platform* platform, double memory_fraction,
      std::optional<int64_t> gpu_system_memory_size,
      absl::Span<const std::pair<StreamExecutor*, Stream*>> devices);

  // Creates an allocator for a single device.
  //
  // Returns an error if the device does not support cuStreamWriteValue64
  // (compute capability < 7.0).
  //
  // Parameters:
  //   executor:  StreamExecutor for this device. Must outlive the allocator.
  //   stream:    Stream used for deferred deallocation. Must outlive the
  //              allocator. This should typically be the main compute stream
  //              from ServiceExecutableRunOptions.
  //   pa_budget: Maximum bytes of physical memory that may be simultaneously
  //              allocated on this device. Defaults to unlimited.
  static absl::StatusOr<std::unique_ptr<CudaDeviceAddressVmmAllocator>> Create(
      StreamExecutor* executor, Stream* stream,
      uint64_t pa_budget = UINT64_MAX);

 protected:
  // Verifies compute capability >= 7.0, allocates the pinned timeline counter
  // via cuMemHostAlloc, obtains the device pointer via
  // cuMemHostGetDevicePointer, and queries the allocation granularity via
  // cuMemGetAllocationGranularity.
  absl::Status InitializeDeviceState(PerDeviceState& state) override;

  // Creates a physical memory allocation via CudaRawMemoryAllocation::Create.
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> CreateAllocation(
      StreamExecutor* executor, uint64_t size) override;

  // Creates a virtual address reservation via CudaMemoryReservation::Create.
  absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* executor, uint64_t size) override;

  // Enqueues a cuStreamWriteValue64 on the device's stream to write `seqno`
  // to the pinned timeline when the GPU reaches this point in the stream.
  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override;

 private:
  explicit CudaDeviceAddressVmmAllocator(const Platform* platform);
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ADDRESS_VMM_ALLOCATOR_H_
