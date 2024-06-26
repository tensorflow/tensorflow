/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_
#define XLA_SERVICE_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/optimization.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/util.h"

namespace xla::cpu {

// Buffer allocation is a container for device buffers allocated for a
// particular XLA execution. Buffers are indexed by the buffer allocation index.
class BufferAllocations {
 public:
  explicit inline BufferAllocations(
      absl::Span<const MaybeOwningDeviceMemory> buffers);

  // Returns the device address of buffer `buffer_index`. `buffer_index` must be
  // a valid index, i.e., in [0, buffer_count).
  inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::StatusOr<se::DeviceMemoryBase>
  GetDeviceAddress(BufferAllocation::Index buffer_index) const;

  // Same as above, but also adjusts the returned address for the offset and
  // size contained in the given slice.
  inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::StatusOr<se::DeviceMemoryBase>
  GetDeviceAddress(const BufferAllocation::Slice& buffer_slice) const;

 private:
  std::vector<se::DeviceMemoryBase> buffers_;
  size_t num_buffers_;
};

BufferAllocations::BufferAllocations(
    absl::Span<const MaybeOwningDeviceMemory> buffers)
    : buffers_(buffers.size()), num_buffers_(buffers_.size()) {
  for (size_t i = 0; i < buffers.size(); ++i) {
    buffers_[i] = buffers[i].AsDeviceMemoryBase();
  }
}

absl::StatusOr<se::DeviceMemoryBase> BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index index) const {
  if (ABSL_PREDICT_FALSE(index < 0 || index >= num_buffers_)) {
    return InvalidArgument(
        "Invalid buffer index %d. It must be in the range [0, %d)", index,
        num_buffers_);
  }

  return buffers_[index];
}

absl::StatusOr<se::DeviceMemoryBase> BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  // Handle empty slices explicitly and return a null pointer device memory to
  // guarantee that we do not accidentally write through the empty slice which
  // would hide a real bug in the code.
  if (ABSL_PREDICT_FALSE(buffer_slice.size() == 0)) {
    return se::DeviceMemoryBase(nullptr, 0);
  }

  int64_t index = buffer_slice.index();
  if (ABSL_PREDICT_FALSE(index < 0 || index >= num_buffers_)) {
    return InvalidArgument(
        "Invalid buffer index %d. It must be in the range [0, %d)", index,
        num_buffers_);
  }
  const se::DeviceMemoryBase& base = buffers_[index];

  int64_t offset = buffer_slice.offset();
  int64_t extent = offset + buffer_slice.size();

  if (ABSL_PREDICT_FALSE(offset < 0)) {
    return InvalidArgument("Buffer slice offset %d must be non-negative",
                           offset);
  }

  if (ABSL_PREDICT_FALSE(offset >= base.size())) {
    return InvalidArgument(
        "Buffer slice offset %d is out of range for buffer #%d of size %d",
        offset, index, base.size());
  }

  if (ABSL_PREDICT_FALSE(extent > base.size())) {
    return InvalidArgument(
        "Buffer slice extent %d is out of range for buffer #%d of size %d",
        extent, index, base.size());
  }

  return base.GetByteSlice(offset, buffer_slice.size());
}

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_
