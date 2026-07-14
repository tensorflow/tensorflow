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

#ifndef XLA_STREAM_EXECUTOR_GENERIC_MEMORY_ALLOCATOR_H_
#define XLA_STREAM_EXECUTOR_GENERIC_MEMORY_ALLOCATOR_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/memory_allocator.h"

namespace stream_executor {

// This class implements the MemoryAllocator interface using an AnyInvocable
// function to allocate a region of memory.
class GenericMemoryAllocator : public MemoryAllocator {
 public:
  explicit GenericMemoryAllocator(
      absl::AnyInvocable<
          absl::StatusOr<std::unique_ptr<MemoryAllocation>>(uint64_t)>
          allocate)
      : allocate_(std::move(allocate)) {}

  absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) override {
    return allocate_(size);
  }

 private:
  absl::AnyInvocable<absl::StatusOr<std::unique_ptr<MemoryAllocation>>(
      uint64_t)>
      allocate_;
};

}  // namespace stream_executor
#endif  // XLA_STREAM_EXECUTOR_GENERIC_MEMORY_ALLOCATOR_H_
