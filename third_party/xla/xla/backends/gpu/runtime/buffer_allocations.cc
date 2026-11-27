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

#include "xla/backends/gpu/runtime/buffer_allocations.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/call_once.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/util/safe_reinterpret_cast.h"

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
  // The caller may change the address through the returned reference.
  ResetAddressIndex();
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

BufferAllocations& BufferAllocations::operator=(BufferAllocations&& other) {
  if (this != &other) {
    buffers_ = std::move(other.buffers_);
    other.buffers_.clear();
    device_ordinal_ = other.device_ordinal_;
    memory_allocator_ = other.memory_allocator_;
    ResetAddressIndex();
    other.ResetAddressIndex();
  }
  return *this;
}

void BufferAllocations::ResetAddressIndex() {
  address_index_once_.emplace();
  address_index_.clear();
}

void BufferAllocations::BuildAddressIndex() const {
  std::vector<AddressIndexEntry>& entries = address_index_;
  entries.reserve(buffers_.size());
  for (BufferAllocation::Index i = 0; i < buffers_.size(); ++i) {
    if (buffers_[i].is_null() || buffers_[i].size() == 0) {
      continue;
    }
    entries.push_back(
        {tsl::safe_reinterpret_cast<uintptr_t>(buffers_[i].opaque()),
         buffers_[i].size(), i});
  }
  absl::c_sort(entries,
               [](const AddressIndexEntry& a, const AddressIndexEntry& b) {
                 return a.begin < b.begin;
               });
  for (size_t i = 1; i < entries.size(); ++i) {
    const AddressIndexEntry& prev = entries[i - 1];
    const AddressIndexEntry& curr = entries[i];
    if (curr.begin == prev.begin && curr.size == prev.size) {
      continue;
    }
    if (curr.begin - prev.begin < prev.size) {
      LOG(ERROR) << "Overlapping buffer allocations detected: buffer "
                 << prev.index << " (" << buffers_[prev.index].opaque() << ", "
                 << prev.size << " B) and buffer " << curr.index << " ("
                 << buffers_[curr.index].opaque() << ", " << curr.size << " B)";
      entries = {};
      return;
    }
  }
}

std::optional<BufferAllocation::Index> BufferAllocations::FindAllocationIndex(
    const se::DeviceAddressBase& addr) const {
  if (addr.is_null()) {
    return std::nullopt;
  }
  absl::call_once(*address_index_once_, &BufferAllocations::BuildAddressIndex,
                  this);
  uintptr_t ptr = tsl::safe_reinterpret_cast<uintptr_t>(addr.opaque());
  auto it = absl::c_upper_bound(
      address_index_, ptr, [](uintptr_t p, const AddressIndexEntry& entry) {
        return p < entry.begin;
      });
  if (it == address_index_.begin()) {
    return std::nullopt;
  }
  --it;
  uint64_t offset = ptr - it->begin;
  if (offset < it->size && addr.size() <= it->size - offset) {
    return it->index;
  }
  return std::nullopt;
}

}  // namespace gpu
}  // namespace xla
