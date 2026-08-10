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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_HOST_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_HOST_ALLOCATOR_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/numa.h"

namespace stream_executor::gpu {

// MemoryAllocator that allocates pinned host memory using the CUDA driver API.
// Supports NUMA-aware allocation: when numa_node is set, memory is allocated
// via NUMAMalloc and registered with CUDA; otherwise uses cuMemHostAlloc.
class CudaHostAllocator : public MemoryAllocator {
 public:
  explicit CudaHostAllocator(StreamExecutor* executor,
                             int32_t numa_node = tsl::port::kNUMANoAffinity);

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) final;

 private:
  StreamExecutor* executor_;
  int32_t numa_node_;
};

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_HOST_ALLOCATOR_H_
