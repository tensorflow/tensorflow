/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/buffer_allocations.h"

#include <cstdint>
#include <optional>
#include <set>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"

namespace xla {
namespace gpu {

absl::Status BufferAllocations::TearDown(
    const std::set<se::DeviceAddressBase>& live_addresses,
    absl::Span<const BufferAllocation* const> allocations) {
  // Deallocate temporary buffers, taking care to try to deallocate all of them
  // even if one of the deallocations fails.
  absl::Status status;
  const int64_t num_buffers = allocations.size();
  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = *allocations[i];
    se::DeviceAddressBase buffer_address = GetDeviceAddress(allocation.index());
    // Deallocate buffers marked "maybe_live_out" but aren't actually live out,
    // and temp buffers.
    if ((allocation.maybe_live_out() &&
         !live_addresses.count(buffer_address)) ||
        allocation.IsPreallocatedTempBuffer()) {
      auto dealloc_result =
          memory_allocator_->Deallocate(device_ordinal_, buffer_address);
      if (!dealloc_result.ok() && status.ok()) {
        status = dealloc_result;
      }
    }
  }
  return status;
}

se::DeviceAddressBase BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceAddressBase& BufferAllocations::GetMutableDeviceAddress(
    BufferAllocation::Index buffer_index) {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceAddressBase BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  int64_t index = buffer_slice.index();
  se::DeviceAddressBase base = GetDeviceAddress(index);

  int64_t offset = buffer_slice.offset();
  CHECK_LE(buffer_slice.offset(), base.size())
      << "slice offset " << offset << " must be smaller than buffer #" << index
      << " size " << base.size();

  int64_t extent = offset + buffer_slice.size();
  CHECK_LE(extent, base.size())
      << "slice extent " << extent << " must be smaller than buffer #" << index
      << " size " << base.size();

  return base.GetByteSlice(buffer_slice.offset(), buffer_slice.size());
}

std::optional<BufferAllocation::Index> BufferAllocations::FindAllocationIndex(
    const se::DeviceAddressBase& addr) const {
  for (BufferAllocation::Index i = 0; i < buffers_.size(); ++i) {
    auto* buf = static_cast<char*>(buffers_[i].opaque());
    auto* ptr = static_cast<char*>(addr.opaque());
    if (ptr >= buf && ptr <= buf + buffers_[i].size()) {
      return i;
    }
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
