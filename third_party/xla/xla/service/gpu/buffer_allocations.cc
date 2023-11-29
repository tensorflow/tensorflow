/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <set>

#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"

namespace xla {
namespace gpu {

Status BufferAllocations::TearDown(
    const std::set<se::DeviceMemoryBase>& live_addresses,
    absl::Span<const BufferAllocation> allocations) {
  // Deallocate temporary buffers, taking care to try to deallocate all of them
  // even if one of the deallocations fails.
  Status status;
  const int64_t num_buffers = allocations.size();
  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = allocations[i];
    se::DeviceMemoryBase buffer_address = GetDeviceAddress(allocation.index());
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

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceMemoryBase& BufferAllocations::GetMutableDeviceAddress(
    BufferAllocation::Index buffer_index) {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return buffers_[buffer_index];
}

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  se::DeviceMemoryBase base = GetDeviceAddress(buffer_slice.index());
  CHECK_LE(buffer_slice.offset(), base.size());
  CHECK_LE(buffer_slice.offset() + buffer_slice.size(), base.size());
  return se::DeviceMemoryBase(
      static_cast<char*>(base.opaque()) + buffer_slice.offset(),
      buffer_slice.size());
}

StatusOr<se::DeviceMemoryBase> BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice,
    const ExternalAllocations& external_allocations) const {
  // Check if base memory address is an external allocation.
  se::DeviceMemoryBase base = GetDeviceAddress(buffer_slice.index());
  return reinterpret_cast<uintptr_t>(base.opaque()) == kExternalAllocationMarker
             ? external_allocations.GetDeviceAddress(buffer_slice)
             : GetDeviceAddress(buffer_slice);
}

}  // namespace gpu
}  // namespace xla
