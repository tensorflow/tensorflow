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

#ifndef XLA_PJRT_HOST_MEMORY_ALLOCATOR_H_
#define XLA_PJRT_HOST_MEMORY_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/framework/allocator.h"

namespace xla {

// An interface for host memory allocation.
class HostMemoryAllocator {
 public:
  struct Options {
    // Minimum alignment of the allocated memory.
    size_t alignment = tsl::Allocator::kAllocatorAlignment;

    // Functions for mapping and unmapping the allocated memory.
    absl::AnyInvocable<absl::Status(void*, size_t)> map_fn;
    absl::AnyInvocable<absl::Status(void*)> unmap_fn;
  };

  using Factory =
      std::function<absl::StatusOr<std::unique_ptr<HostMemoryAllocator>>(
          Options options)>;

  struct Deleter {
    void operator()(void* ptr) { deleter(ptr, arg); }
    void (*deleter)(void* ptr, void* arg);
    void* arg;
  };
  using OwnedPtr = std::unique_ptr<uint8_t[], Deleter>;

  virtual ~HostMemoryAllocator() = default;

  // Allocates `size` bytes of memory. The returned pointer is guaranteed to be
  // aligned to `options_.alignment`.
  virtual OwnedPtr Allocate(size_t size) = 0;
};

// `HostMemoryAllocator` implementation that uses a `tsl::Allocator` to back
// allocations.
class BasicHostMemoryAllocator : public HostMemoryAllocator {
 public:
  explicit BasicHostMemoryAllocator(
      std::unique_ptr<tsl::Allocator> allocator,
      size_t alignment = tsl::Allocator::kAllocatorAlignment);

  OwnedPtr Allocate(size_t size) override;

 private:
  const std::unique_ptr<tsl::Allocator> allocator_;
  const size_t alignment_;
};

}  // namespace xla

#endif  // XLA_PJRT_HOST_MEMORY_ALLOCATOR_H_
