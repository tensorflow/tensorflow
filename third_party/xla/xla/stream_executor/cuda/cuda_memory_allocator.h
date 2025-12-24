/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_MEMORY_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_MEMORY_ALLOCATOR_H_

#include <memory>
#include <string>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"

namespace stream_executor::gpu {

// A type of memory allocator for kCollective memory space.
enum class CollectiveAllocatorType { kNccl, kNvshmem };

template <typename Sink>
void AbslStringify(Sink& sink, CollectiveAllocatorType allocator_type) {
  switch (allocator_type) {
    case CollectiveAllocatorType::kNccl:
      sink.Append("NCCL");
      break;
    case CollectiveAllocatorType::kNvshmem:
      sink.Append("NVSHMEM");
      break;
  }
}

using CollectiveAllocatorFactory =  // NOLINT
    absl::AnyInvocable<std::unique_ptr<MemoryAllocator>(StreamExecutor*)>;

// Static registration of a collective memory allocator factory. NCCL and
// NVSHMEM allocators are not supported in all build configurations, and
// we rely on the static registration pattern as a way to ensure that
// we can dynamically select between available allocators.
void RegisterCollectiveAllocatorFactory(
    CollectiveAllocatorType allocator_type,
    CollectiveAllocatorFactory allocator_factory);

// Creates a collective memory allocator for the given allocator type.
absl::StatusOr<std::unique_ptr<MemoryAllocator>>
CreateCollectiveMemoryAllocator(StreamExecutor* executor,
                                CollectiveAllocatorType allocator_type);

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_MEMORY_ALLOCATOR_H_
