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

#include "xla/pjrt/host_memory_allocator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "xla/tsl/framework/allocator.h"

namespace xla {

BasicHostMemoryAllocator::BasicHostMemoryAllocator(
    std::unique_ptr<tsl::Allocator> allocator, size_t alignment)
    : allocator_(std::move(allocator)), alignment_(alignment) {}

HostMemoryAllocator::OwnedPtr BasicHostMemoryAllocator::Allocate(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  return OwnedPtr(
      reinterpret_cast<uint8_t*>(allocator_->AllocateRaw(alignment_, size)),
      {
          +[](void* ptr, void* arg) {
            reinterpret_cast<tsl::Allocator*>(arg)->DeallocateRaw(ptr);
          },
          allocator_.get(),
      });
}

}  // namespace xla
