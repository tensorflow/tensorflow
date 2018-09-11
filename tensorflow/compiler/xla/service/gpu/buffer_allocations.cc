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

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/map_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

void BufferAllocations::Builder::RegisterBuffer(BufferAllocation::Index index,
                                                se::DeviceMemoryBase address) {
  InsertOrDie(&registered_buffers_, index, address);
}

StatusOr<std::unique_ptr<BufferAllocations>> BufferAllocations::Builder::Build(
    const BufferAssignment* buffer_assignment, int device_ordinal,
    DeviceMemoryAllocator* memory_allocator) {
  const int64 num_buffers = buffer_assignment->Allocations().size();
  auto buffer_allocations = absl::WrapUnique(new BufferAllocations(
      num_buffers, device_ordinal, memory_allocator, buffer_assignment));

  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = buffer_assignment->GetAllocation(i);
    const int64 expected_alignment = [&] {
      if (allocation.is_entry_computation_parameter()) {
        return kEntryParameterAlignBytes;
      } else if (allocation.is_constant()) {
        return kConstantBufferAlignBytes;
      } else {
        return kXlaAllocatedBufferAlignBytes;
      }
    }();

    // If buffer #i's address is already registered (e.g. external arguments or
    // result buffers), use that registered buffer.
    if (registered_buffers_.count(i)) {
      se::DeviceMemoryBase address = FindOrDie(registered_buffers_, i);
      if (reinterpret_cast<uintptr_t>(address.opaque()) % expected_alignment !=
          0) {
        return InternalError(
            "Address of registered buffer %d must be a multiple of %x, but "
            "was %p",
            i, kEntryParameterAlignBytes, address.opaque());
      }
      buffer_allocations->SetBuffer(i, FindOrDie(registered_buffers_, i));
      continue;
    }

    // Allocate each allocation that might escape, or is the temp buffer.
    bool seen_temp_buffer = false;
    if (allocation.maybe_live_out() || allocation.IsPreallocatedTempBuffer()) {
      const int64 buffer_size = allocation.size();
      se::DeviceMemoryBase buffer_address;
      if (buffer_size > 0) {
        OwningDeviceMemory buffer;
        TF_ASSIGN_OR_RETURN(
            buffer, memory_allocator->Allocate(device_ordinal, buffer_size));
        if (reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment !=
            0) {
          return InternalError(
              "Address returned by memory_allocator->Allocate must be a "
              "multiple of 0x%x, but was %p",
              kXlaAllocatedBufferAlignBytes, buffer.opaque());
        }
        // We do manual memory management within BufferAllocations.  Be sure not
        // to do a TF_RETURN_IF_ERROR between this line and the
        // buffer_allocations->SetBuffer(buffer_address) call below!
        buffer_address = buffer.Forget();
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

  if (VLOG_IS_ON(2)) {
    for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
      const auto& buf = buffer_allocations->buffers_[i];
      VLOG(2) << "Buffer " << i << " -> " << buf.opaque() << " (" << buf.size()
              << "B)";
    }
  }
  return std::move(buffer_allocations);
}

BufferAllocations::~BufferAllocations() {
  if (!torn_down_) {
    // Presumably if we're executing this branch, the caller is in an error
    // state, otherwise it would have explicitly called TearDown so it could
    // save some set of live addresses.  So ignoring any errors in TearDown is
    // sensible.
    TearDown(/*live_addresses=*/{}).IgnoreError();
  }
}

Status BufferAllocations::TearDown(
    const std::set<se::DeviceMemoryBase>& live_addresses) {
  // Deallocate temporary buffers, taking care to try to deallocate all of them
  // even if one of the deallocations fails.
  Status status;
  const int64 num_buffers = buffer_assignment_->Allocations().size();
  for (BufferAllocation::Index i = 0; i < num_buffers; ++i) {
    const BufferAllocation& allocation = buffer_assignment_->GetAllocation(i);
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
  torn_down_ = true;
  return status;
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

bool ShouldEmitLiteralInLlvmIr(const Literal& literal) {
  // LLVM can sometimes do interesting optimizations using scalar constants.
  return ShapeUtil::IsScalar(literal.shape());
}

}  // namespace gpu
}  // namespace xla
