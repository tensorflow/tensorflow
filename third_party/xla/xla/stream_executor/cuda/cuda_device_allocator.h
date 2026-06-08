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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// CUDA device memory allocator. The current implementation uses CUDA Virtual
// Memory Management (VMM) APIs (cuMemCreate/cuMemAddressReserve/cuMemMap).
class CudaDeviceAllocator : public MemoryAllocator {
 public:
  struct Options {
    // Minimum alignment for allocations. The actual alignment is the maximum
    // of this value and the device-reported allocation granularity.
    size_t alignment = 4096;

    // Whether to enable peer access from all accessible devices.
    bool enable_peer_access = false;

    // Whether to request POSIX_FILE_DESCRIPTOR handle type.
    bool enable_posix_fd_handle = true;

    // Whether to request FABRIC handle type.
    bool enable_fabric_handle = false;

    // Whether to mark allocations as GPUDirect RDMA capable.
    bool enable_rdma = false;
  };

  explicit CudaDeviceAllocator(StreamExecutor* executor);
  CudaDeviceAllocator(StreamExecutor* executor, Options options);

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) final;

  const Options& options() const { return options_; }

 private:
  StreamExecutor* executor_;
  Options options_;
};

CUmemAllocationProp BuildVmmAllocationProp(
    CUdevice device, const CudaDeviceAllocator::Options& options);

// Probes device allocator options supported by CUDA VMM; falls back through
// simpler handle-type combinations if the device rejects the strongest one.
absl::StatusOr<CudaDeviceAllocator::Options> QueryDeviceAllocatorOptions(
    CUdevice device);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_DEVICE_ALLOCATOR_H_
