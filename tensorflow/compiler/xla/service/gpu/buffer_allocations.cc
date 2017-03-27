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

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"

#include <utility>

#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace se = ::perftools::gputools;

namespace xla {
namespace gpu {

void BufferAllocations::Builder::RegisterBuffer(BufferAllocation::Index index,
                                                se::DeviceMemoryBase address) {
  InsertOrDie(&registered_buffers_, index, address);
}

StatusOr<std::unique_ptr<BufferAllocations>> BufferAllocations::Builder::Build(
    const BufferAssignment& buffer_assignment, int device_ordinal,
    DeviceMemoryAllocator* memory_allocator) {
  const int64 num_buffers = buffer_assignment.Allocations().size();
  auto buffer_allocations = WrapUnique(
      new BufferAllocations(num_buffers, device_ordinal, memory_allocator));

  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    // If buffer #i's address is already registered (e.g. external arguments or
    // result buffers), use that registered buffer.
    if (registered_buffers_.count(i)) {
      buffer_allocations->SetBuffer(i, FindOrDie(registered_buffers_, i));
      continue;
    }

    // Allocate each allocation that might escape, or is the temp buffer.
    bool seen_temp_buffer = false;
    const BufferAllocation& allocation = buffer_assignment.GetAllocation(i);
    if (allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer()) {
      const int64 buffer_size = allocation.size();
      se::DeviceMemoryBase buffer_address;
      if (buffer_size > 0) {
        TF_ASSIGN_OR_RETURN(buffer_address, memory_allocator->Allocate(
                                                device_ordinal, buffer_size));
        if (buffer_address == nullptr) {
          return ResourceExhausted(
              "Out of memory when allocating %s for buffer %lld.",
              tensorflow::strings::HumanReadableNumBytes(buffer_size).c_str(),
              i);
        }
      }
      buffer_allocations->SetBuffer(i, buffer_address);
      if (allocation.IsPreallocatedTempBuffer()) {
        if (seen_temp_buffer) {
          LOG(FATAL) << "Multiple temporary buffers detected.  BufferAssigner "
                     << "must guarantee at most one temporary buffer.";
        }
        seen_temp_buffer = true;
        buffer_allocations->temp_buffer_base_ = buffer_address;
      }
    }
  }

  return std::move(buffer_allocations);
}

tensorflow::Status BufferAllocations::TearDown(
    const std::set<se::DeviceMemoryBase>& live_addresses,
    const BufferAssignment& buffer_assignment) {
  // Deallocate temporary buffers.
  for (auto i = 0; i < buffer_assignment.Allocations().size(); ++i) {
    const BufferAllocation& allocation = buffer_assignment.GetAllocation(i);
    se::DeviceMemoryBase buffer_address = GetDeviceAddress(allocation.index());
    // Deallocate buffers marked "maybe_live_out" but aren't actually live out,
    // and temp buffers.
    if ((allocation.maybe_live_out() &&
         !live_addresses.count(buffer_address)) ||
        allocation.IsPreallocatedTempBuffer()) {
      TF_RETURN_IF_ERROR(
          memory_allocator_->Deallocate(device_ordinal_, &buffer_address));
    }
  }
  return tensorflow::Status::OK();
}

se::DeviceMemoryBase BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
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
      buffer_slice.size(), /*is_sub_buffer=*/true);
}

void BufferAllocations::SetBuffer(BufferAllocation::Index buffer_index,
                                  se::DeviceMemoryBase buffer) {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  buffers_[buffer_index] = buffer;
}

}  // namespace gpu
}  // namespace xla
