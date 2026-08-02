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

#ifndef XLA_STREAM_EXECUTOR_ROCM_ROCM_RAW_MEMORY_ALLOCATION_H_
#define XLA_STREAM_EXECUTOR_ROCM_ROCM_RAW_MEMORY_ALLOCATION_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "rocm/include/hip/hip_runtime.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// RAII wrapper for a physical memory allocation created via hipMemCreate.
// The handle is not mapped to any virtual address; this class only manages
// the lifetime of the hipMemGenericAllocationHandle_t.
class RocmRawMemoryAllocation : public MemoryAllocation {
 public:
  // Creates a physical memory allocation of at least `size` bytes using
  // hipMemCreate. StreamExecutor is used only for context activation.
  static absl::StatusOr<std::unique_ptr<RocmRawMemoryAllocation>> Create(
      StreamExecutor* executor, uint64_t size);

  // Returns a DeviceAddressBase whose opaque() holds the raw
  // hipMemGenericAllocationHandle_t cast to void*, and size() is the
  // padded allocation size.
  DeviceAddressBase address() const override;

  hipMemGenericAllocationHandle_t GetHandle() const { return handle_; }

  ~RocmRawMemoryAllocation() override;
  RocmRawMemoryAllocation(const RocmRawMemoryAllocation&) = delete;
  RocmRawMemoryAllocation& operator=(const RocmRawMemoryAllocation&) = delete;
  RocmRawMemoryAllocation(RocmRawMemoryAllocation&&) = delete;
  RocmRawMemoryAllocation& operator=(RocmRawMemoryAllocation&&) = delete;

 private:
  explicit RocmRawMemoryAllocation(StreamExecutor* executor,
                                   hipMemGenericAllocationHandle_t handle,
                                   uint64_t size);

  StreamExecutor* executor_;
  hipMemGenericAllocationHandle_t handle_;
  uint64_t size_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_ROCM_ROCM_RAW_MEMORY_ALLOCATION_H_
