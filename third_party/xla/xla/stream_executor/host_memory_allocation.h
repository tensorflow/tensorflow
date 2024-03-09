/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_HOST_MEMORY_ALLOCATION_H_
#define XLA_STREAM_EXECUTOR_HOST_MEMORY_ALLOCATION_H_

#include <cstdint>

#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {

namespace internal {
class StreamExecutorInterface;
}

// RAII container for pinned host memory allocation allocated on an underlying
// device owned by `*this`.
class HostMemoryAllocation final : public MemoryAllocation {
 public:
  HostMemoryAllocation(void* ptr, uint64_t size,
                       internal::StreamExecutorInterface* executor);
  ~HostMemoryAllocation() final;

  void* opaque() const final { return ptr_; }
  uint64_t size() const final { return size_; }

 private:
  void* ptr_ = nullptr;
  uint64_t size_ = 0;
  internal::StreamExecutorInterface* executor_ = nullptr;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_HOST_MEMORY_ALLOCATION_H_
