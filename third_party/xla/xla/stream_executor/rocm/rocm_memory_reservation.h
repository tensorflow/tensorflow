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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_RESERVATION_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_RESERVATION_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// RAII wrapper for a ROCm virtual address range reserved via
// hipMemAddressReserve. Physical memory can be mapped into sub-ranges of the
// reservation via MapTo, which also enables device access before returning.
class RocmMemoryReservation : public MemoryReservation {
 public:
  // Reserves a virtual address range of at least `size` bytes using
  // hipMemAddressReserve. StreamExecutor is used only for context activation.
  static absl::StatusOr<std::unique_ptr<RocmMemoryReservation>> Create(
      StreamExecutor* executor, uint64_t size);

  // Returns the base address and padded size of the reserved virtual range.
  DeviceAddressBase address() const override;

  ~RocmMemoryReservation() override;

  // Non-movable: the base class MemoryReservation already deletes its move
  // operations, and instances are only ever created through Create(), which
  // returns a std::unique_ptr. Ownership is therefore transferred via the
  // pointer, never by moving the object, so there is no moved-from state to
  // reason about. Matches CudaMemoryReservation.
  RocmMemoryReservation(RocmMemoryReservation&&) = delete;
  RocmMemoryReservation& operator=(RocmMemoryReservation&&) = delete;

 private:
  explicit RocmMemoryReservation(StreamExecutor* executor, char* ptr,
                                 uint64_t size);

  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, MemoryAllocation& allocation) override;

  absl::Status SetAccess(uint64_t reservation_offset, size_t size) override;

  absl::Status UnMap(size_t reservation_offset, size_t size) override;

  StreamExecutor* executor_;
  // Base of the reserved virtual address range. Non-null after construction
  // (the class is non-movable and exposes no release path); the destructor's
  // null check is purely defensive.
  char* ptr_;
  uint64_t size_;
  // Bytes currently mapped into the reservation. Kept in sync by Map()/UnMap()
  // so the destructor can skip a redundant hipMemUnmap when the ScopedMapping
  // that owns the mapping has already unmapped the range (which would otherwise
  // fail with HIP_ERROR_InvalidValue).
  uint64_t mapped_bytes_ = 0;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_MEMORY_RESERVATION_H_
