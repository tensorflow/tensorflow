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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_MEMORY_RESERVATION_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_MEMORY_RESERVATION_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_reservation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// RAII wrapper for a CUDA virtual address range reserved via
// cuMemAddressReserve. Physical memory can be mapped into sub-ranges of the
// reservation via MapTo, which also enables device access before returning.
class CudaMemoryReservation : public MemoryReservation {
 public:
  // Reserves a virtual address range of at least `size` bytes using
  // cuMemAddressReserve. StreamExecutor is used only for context activation.
  static absl::StatusOr<std::unique_ptr<CudaMemoryReservation>> Create(
      StreamExecutor* executor, uint64_t size);

  // Returns the base address and padded size of the reserved virtual range.
  DeviceAddressBase address() const override;

  ~CudaMemoryReservation() override;
  CudaMemoryReservation(CudaMemoryReservation&&) = delete;
  CudaMemoryReservation& operator=(CudaMemoryReservation&&) = delete;

 private:
  explicit CudaMemoryReservation(StreamExecutor* executor, CUdeviceptr ptr,
                                 uint64_t size);

  // Maps [reservation_offset, reservation_offset+size) in the reservation to
  // [allocation_offset, allocation_offset+size) in allocation via cuMemMap.
  // allocation must be a CudaRawMemoryAllocation.
  absl::Status Map(size_t reservation_offset, size_t allocation_offset,
                   size_t size, MemoryAllocation& allocation) override;

  // Enables read/write access to the full reservation for the owning device
  // via cuMemSetAccess.
  absl::Status SetAccess(uint64_t reservation_offset, size_t size) override;

  // Unmaps [offset, offset+size) within this reservation via cuMemUnmap.
  absl::Status UnMap(size_t reservation_offset, size_t size) override;

  StreamExecutor* executor_;
  CUdeviceptr ptr_;  // 0 means moved-from / released
  uint64_t size_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_MEMORY_RESERVATION_H_
