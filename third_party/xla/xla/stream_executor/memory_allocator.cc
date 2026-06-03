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

#include "xla/stream_executor/memory_allocator.h"

#include <cstdint>
#include <memory>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/memory_allocation.h"

namespace stream_executor {

absl::StatusOr<DeviceAddressBase> MemoryAllocator::AllocationTracker::Track(
    std::unique_ptr<MemoryAllocation> allocation) {
  if (allocation == nullptr) {
    return absl::InvalidArgumentError("Cannot track a null MemoryAllocation");
  }

  DeviceAddressBase addr = allocation->address();
  void* ptr = addr.opaque();

  absl::MutexLock lock(mu_);
  uint64_t id = next_allocation_id_++;
  // Set the payload on the DeviceAddressBase handle being returned. This
  // allows subsequent operations using this specific handle to quickly find
  // the tracked allocation using the payload ID. The MemoryAllocation object
  // stored in 'allocations_' does not have its internal DeviceAddressBase
  // updated with this payload, as the tracker manages the IDs externally.
  addr.SetPayload(id);

  if (ptr != nullptr) {
    if (ptr_to_id_.contains(ptr)) {
      return absl::AlreadyExistsError(absl::StrFormat(
          "Allocation at address %p (size %d) is already tracked", ptr,
          addr.size()));
    }
    ptr_to_id_.emplace(ptr, id);
  }

  allocations_.emplace(id, std::move(allocation));
  return addr;
}

bool MemoryAllocator::AllocationTracker::IsTracked(
    const DeviceAddressBase& addr) const {
  absl::MutexLock lock(mu_);
  uint64_t id = addr.payload();
  if (id == 0 && addr.opaque() != nullptr) {
    auto it = ptr_to_id_.find(addr.opaque());
    if (it != ptr_to_id_.end()) {
      id = it->second;
    }
  }
  return id != 0 && allocations_.contains(id);
}

absl::Status MemoryAllocator::AllocationTracker::Free(DeviceAddressBase addr) {
  std::unique_ptr<MemoryAllocation> allocation_to_free;
  {
    absl::MutexLock lock(mu_);
    uint64_t id = addr.payload();
    // If payload is 0, the caller may have reconstructed the DeviceAddressBase
    // using only the void* pointer. We fall back to looking up the unique ID
    // using the pointer, if it is not null.
    if (id == 0 && addr.opaque() != nullptr) {
      auto it = ptr_to_id_.find(addr.opaque());
      if (it != ptr_to_id_.end()) {
        id = it->second;
      }
    }

    if (id == 0) {
      return absl::NotFoundError(
          absl::StrFormat("No tracked allocation for address %p (size %d)",
                          addr.opaque(), addr.size()));
    }

    auto alloc_it = allocations_.find(id);
    if (alloc_it == allocations_.end()) {
      return absl::NotFoundError(absl::StrFormat(
          "No tracked allocation for payload ID %d (address %p, size %d)", id,
          addr.opaque(), addr.size()));
    }

    const DeviceAddressBase& tracked_addr = alloc_it->second->address();
    if (addr.opaque() != nullptr && addr.opaque() != tracked_addr.opaque()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Address mismatch for payload ID %d: provided %p, tracked %p", id,
          addr.opaque(), tracked_addr.opaque()));
    }
    if (addr.size() > tracked_addr.size()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Size mismatch for payload ID %d: provided %d, tracked %d", id,
          addr.size(), tracked_addr.size()));
    }

    void* stored_ptr = alloc_it->second->address().opaque();
    if (stored_ptr != nullptr) {
      ptr_to_id_.erase(stored_ptr);
    }
    allocation_to_free = std::move(alloc_it->second);
    allocations_.erase(alloc_it);
  }
  return absl::OkStatus();
}

}  // namespace stream_executor
