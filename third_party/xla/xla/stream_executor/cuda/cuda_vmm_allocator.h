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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_VMM_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_VMM_ALLOCATOR_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// Device memory allocator using CUDA Virtual Memory Management (VMM) APIs.
// Returned memory allocations are always mapped to a valid device address range
// and accessible from the device and its peers.
class CudaVmmAllocator : public MemoryAllocator {
 public:
  CudaVmmAllocator(StreamExecutor* executor, bool is_rdma_supported);

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) final;

 private:
  StreamExecutor* executor_;
  bool is_rdma_supported_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_VMM_ALLOCATOR_H_
