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

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {

// Abstract base class for stream executor memory allocators.
//
// Concrete implementations allocate physical memory in a specific memory space
// (device, host, unified, etc.) and return RAII MemoryAllocation objects that
// automatically free the memory on destruction.
class MemoryAllocator {
 public:
  virtual ~MemoryAllocator() = default;

  // Allocates a contiguous region of at least `size` bytes, or returns an error
  // if the allocation fails. The returned MemoryAllocation owns the memory and
  // frees it on destruction. The actual allocation may be larger than `size`
  // due to platform-specific granularity requirements (e.g. CUDA VMM).
  virtual absl::StatusOr<std::unique_ptr<MemoryAllocation>> Allocate(
      uint64_t size) = 0;

  // Bridges the RAII MemoryAllocation API to a raw-pointer allocate/free
  // interface (similar to malloc/free).
  //
  // AllocationTracker takes ownership of MemoryAllocation objects via Track()
  // and returns a DeviceAddressBase that callers can use as an opaque handle.
  // When Free() is called with that handle, the corresponding MemoryAllocation
  // is destroyed, releasing the underlying memory.
  //
  // Thread-safe: all methods are synchronized internally.
  //
  // Only complete address ranges returned by Track() may be freed; passing a
  // sub-range or an untracked address to Free() is an error.
  class AllocationTracker {
   public:
    // Takes ownership of `allocation` and returns its address as a
    // DeviceAddressBase handle. Returns an error if the allocation is null or
    // if the address is already tracked.
    absl::StatusOr<DeviceAddressBase> Track(
        std::unique_ptr<MemoryAllocation> allocation);

    // Returns true if the given address is currently tracked.
    bool IsTracked(const DeviceAddressBase& addr) const;

    // Releases the allocation previously registered via Track() for the given
    // address. Returns an error if the address is not tracked.
    absl::Status Free(DeviceAddressBase addr);

   private:
    mutable absl::Mutex mu_;
    // Keyed by the raw opaque pointer rather than DeviceAddressBase, because
    // callers of Free() may not know the original allocation size (e.g.
    // DeviceMemAllocator::Free constructs a DeviceAddressBase with size=0).
    absl::flat_hash_map<void*, std::unique_ptr<MemoryAllocation>> allocations_
        ABSL_GUARDED_BY(mu_);
  };
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_MEMORY_ALLOCATOR_H_
