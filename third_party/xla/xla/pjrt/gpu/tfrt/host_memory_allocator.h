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

#ifndef XLA_PJRT_GPU_TFRT_HOST_MEMORY_ALLOCATOR_H_
#define XLA_PJRT_GPU_TFRT_HOST_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <functional>
#include <memory>
#include <utility>

#include "xla/tsl/framework/allocator.h"

namespace xla {
class HostMemoryAllocator {
 public:
  explicit HostMemoryAllocator(std::unique_ptr<tsl::Allocator> allocator)
      : allocator_(std::move(allocator)) {}

  // Uses tsl::Allocator destructor as the deleter for owned pointer.
  using OwnedPtr = std::unique_ptr<void, std::function<void(void*)>>;
  OwnedPtr Allocate(size_t size) {
    if (size == 0) return OwnedPtr(nullptr, [](void* ptr) {});
    return OwnedPtr(
        allocator_->AllocateRaw(tsl::Allocator::kAllocatorAlignment, size),
        [this](void* ptr) { allocator_->DeallocateRaw(ptr); });
  }

 private:
  std::unique_ptr<tsl::Allocator> allocator_;
};
}  // namespace xla

#endif  // XLA_PJRT_GPU_TFRT_HOST_MEMORY_ALLOCATOR_H_
