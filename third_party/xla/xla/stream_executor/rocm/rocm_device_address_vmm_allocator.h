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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_DEVICE_ADDRESS_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_DEVICE_ADDRESS_VMM_ALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_address_vmm_allocator.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// ROCm/HIP implementation of DeviceAddressVmmAllocator.
//
// Uses hipMemCreate/hipMemAddressReserve for physical and virtual memory
// management, and hipStreamWriteValue64 for GPU timeline-based deferred
// deallocation. Requires ROCm >= 6.0 for HIP VMM API support.
//
// The timeline counter is allocated as signal memory via
// hipExtMallocWithFlags(hipMallocSignalMemory), which is required by
// hipStreamWriteValue64 on AMD hardware.
//
// Use the Create() factories to obtain an instance.
class RocmDeviceAddressVmmAllocator : public DeviceAddressVmmAllocator {
 public:
  // Creates an allocator supporting multiple devices.
  //
  // Precondition: all entries in `devices` have distinct device ordinals.
  static absl::StatusOr<std::unique_ptr<RocmDeviceAddressVmmAllocator>> Create(
      const Platform* platform, absl::Span<const DeviceConfig> devices,
      std::optional<int64_t> reclaim_exempt_memory_space = std::nullopt);

  // Creates an allocator supporting multiple devices, computing the pa_budget
  // for each device by querying DeviceMemoryUsage and applying memory_fraction.
  // If gpu_system_memory_size is set, it overrides the memory_fraction budget.
  //
  // Precondition: all entries in `devices` have distinct device ordinals.
  static absl::StatusOr<std::unique_ptr<RocmDeviceAddressVmmAllocator>> Create(
      const Platform* platform, double memory_fraction,
      std::optional<int64_t> gpu_system_memory_size,
      absl::Span<const std::pair<StreamExecutor*, Stream*>> devices,
      std::optional<int64_t> reclaim_exempt_memory_space = std::nullopt);

  // Creates an allocator for a single device.
  //
  // Parameters:
  //   executor:  StreamExecutor for this device. Must outlive the allocator.
  //   stream:    Stream used for deferred deallocation. Must outlive the
  //              allocator.
  //   pa_budget: Maximum bytes of physical memory that may be simultaneously
  //              allocated on this device. Defaults to unlimited.
  static absl::StatusOr<std::unique_ptr<RocmDeviceAddressVmmAllocator>> Create(
      StreamExecutor* executor, Stream* stream, uint64_t pa_budget = UINT64_MAX,
      std::optional<int64_t> reclaim_exempt_memory_space = std::nullopt);

 protected:
  // Allocates signal memory via hipExtMallocWithFlags(hipMallocSignalMemory)
  // for the per-device timeline counter, and queries the allocation
  // granularity via hipMemGetAllocationGranularity.
  absl::Status InitializeDeviceState(PerDeviceState& state) override;

  // Creates a physical memory allocation via RocmRawMemoryAllocation::Create.
  absl::StatusOr<std::unique_ptr<MemoryAllocation>> CreateAllocation(
      StreamExecutor* executor, uint64_t size) override;

  // Creates a virtual address reservation via RocmMemoryReservation::Create.
  absl::StatusOr<std::unique_ptr<MemoryReservation>> CreateReservation(
      StreamExecutor* executor, uint64_t size) override;

  // Enqueues a hipStreamWriteValue64 on the device's stream to write `seqno`
  // to the signal memory timeline when the GPU reaches this point.
  absl::Status EnqueueDeferredDeallocation(PerDeviceState& state,
                                           uint64_t seqno) override;

 private:
  explicit RocmDeviceAddressVmmAllocator(
      const Platform* platform,
      std::optional<int64_t> reclaim_exempt_memory_space = std::nullopt);
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_DEVICE_ADDRESS_VMM_ALLOCATOR_H_
