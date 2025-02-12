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

#ifndef XLA_STREAM_EXECUTOR_GENERIC_MEMORY_ALLOCATION_H_
#define XLA_STREAM_EXECUTOR_GENERIC_MEMORY_ALLOCATION_H_

#include <cstdint>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {

// RAII container for memory allocated using an absl::AnyInvocable function to
// delete the memory.
class GenericMemoryAllocation final : public MemoryAllocation {
 public:
  GenericMemoryAllocation(void* ptr, uint64_t size,
                          absl::AnyInvocable<void(void*, uint64_t)> deleter)
      : ptr_(ptr), size_(size), deleter_(std::move(deleter)) {}
  ~GenericMemoryAllocation() override {
    if (ptr_ != nullptr) {
      deleter_(ptr_, size_);
      ptr_ = nullptr;
      size_ = 0;
    }
  }

  void* opaque() const final { return ptr_; }
  uint64_t size() const final { return size_; }

 private:
  void* ptr_ = nullptr;
  uint64_t size_ = 0;
  absl::AnyInvocable<void(void*, uint64_t)> deleter_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GENERIC_MEMORY_ALLOCATION_H_
