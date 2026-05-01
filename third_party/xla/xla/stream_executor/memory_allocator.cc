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

  absl::MutexLock lock(&mu_);
  auto [it, inserted] = allocations_.emplace(ptr, std::move(allocation));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrFormat("Allocation at address %p (size %d) is already tracked",
                        ptr, addr.size()));
  }
  return addr;
}

bool MemoryAllocator::AllocationTracker::IsTracked(
    const DeviceAddressBase& addr) const {
  absl::MutexLock lock(&mu_);
  return allocations_.contains(addr.opaque());
}

absl::Status MemoryAllocator::AllocationTracker::Free(DeviceAddressBase addr) {
  absl::MutexLock lock(&mu_);
  auto it = allocations_.find(addr.opaque());
  if (it == allocations_.end()) {
    return absl::NotFoundError(
        absl::StrFormat("No tracked allocation at address %p (size %d)",
                        addr.opaque(), addr.size()));
  }
  allocations_.erase(it);
  return absl::OkStatus();
}

}  // namespace stream_executor
