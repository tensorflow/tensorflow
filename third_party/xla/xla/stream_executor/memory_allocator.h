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

#ifndef XLA_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_

#include <cstdint>
#include <memory>

#include "absl/status/statusor.h"
#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {

// A base class for stream executor memory allocators.
//
// Memory allocators are responsible allocating physical memory for a given
// stream executor, this physical memory might reside in different memory spaces
// such as device memory, unified memory, host memory, etc. See MemoryAllocation
// documentation for more details.
class MemoryAllocator {
 public:
  virtual ~MemoryAllocator() = default;

  // Allocates a region of memory, or returns an error if the allocation fails.
  virtual absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) = 0;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_
