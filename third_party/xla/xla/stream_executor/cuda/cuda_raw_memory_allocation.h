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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_RAW_MEMORY_ALLOCATION_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_RAW_MEMORY_ALLOCATION_H_

#include <cstdint>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// RAII wrapper for a physical memory allocation created via cuMemCreate.
// The handle is not mapped to any virtual address; this class only manages
// the lifetime of the CUmemGenericAllocationHandle.
class CudaRawMemoryAllocation : public MemoryAllocation {
 public:
  // Creates a physical memory allocation of at least `size` bytes using
  // cuMemCreate. StreamExecutor is used only for context activation.
  static absl::StatusOr<std::unique_ptr<CudaRawMemoryAllocation>> Create(
      StreamExecutor* executor, uint64_t size);

  // Returns a DeviceAddressBase whose opaque() holds the raw
  // CUmemGenericAllocationHandle cast to void*, and size() is the
  // padded allocation size.
  DeviceAddressBase address() const override;

  CUmemGenericAllocationHandle GetHandle() const { return handle_; }

  ~CudaRawMemoryAllocation() override;
  CudaRawMemoryAllocation(const CudaRawMemoryAllocation&) = delete;
  CudaRawMemoryAllocation& operator=(const CudaRawMemoryAllocation&) = delete;

 private:
  explicit CudaRawMemoryAllocation(StreamExecutor* executor,
                                   CUmemGenericAllocationHandle handle,
                                   uint64_t size);

  StreamExecutor* executor_;
  CUmemGenericAllocationHandle handle_;  // 0 means moved-from / released
  uint64_t size_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_RAW_MEMORY_ALLOCATION_H_
