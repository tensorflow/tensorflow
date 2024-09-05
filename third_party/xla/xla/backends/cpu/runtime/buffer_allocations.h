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

#ifndef XLA_BACKENDS_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_
#define XLA_BACKENDS_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_

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
  explicit BufferAllocations(absl::Span<const MaybeOwningDeviceMemory> buffers);

  // Returns the device address of buffer at the given index. Returns an error
  // if the index is out of range.
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Index index) const;

  // Same as above, but also adjusts the returned address for the offset and
  // size contained in the given slice.
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Slice slice) const;

  // Unchecked version of `GetDeviceAddress` that does not check the buffer
  // index and assumes it is valid.
  se::DeviceMemoryBase GetDeviceAddressUnchecked(
      BufferAllocation::Index buffer_index) const;

  // Unchecked version of `GetDeviceAddress` that does not check the slice
  // buffer index, offset and size and assumes they all are valid.
  se::DeviceMemoryBase GetDeviceAddressUnchecked(
      BufferAllocation::Slice slice) const;

 private:
  std::vector<se::DeviceMemoryBase> buffers_;
  se::DeviceMemoryBase* buffers_data_;  // buffers_.data()
  size_t num_buffers_;
};

inline BufferAllocations::BufferAllocations(
    absl::Span<const MaybeOwningDeviceMemory> buffers)
    : buffers_(buffers.size()),
      buffers_data_(buffers_.data()),
      num_buffers_(buffers_.size()) {
  for (size_t i = 0; i < buffers.size(); ++i) {
    buffers_[i] = buffers[i].AsDeviceMemoryBase();
  }
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::StatusOr<se::DeviceMemoryBase>
BufferAllocations::GetDeviceAddress(BufferAllocation::Index index) const {
  if (ABSL_PREDICT_FALSE(index < 0 || index >= num_buffers_)) {
    return InvalidArgument(
        "Invalid buffer index %d. It must be in the range [0, %d)", index,
        num_buffers_);
  }

  return buffers_[index];
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE absl::StatusOr<se::DeviceMemoryBase>
BufferAllocations::GetDeviceAddress(BufferAllocation::Slice slice) const {
  // Handle empty slices explicitly and return a null pointer device memory to
  // guarantee that we do not accidentally write through the empty slice which
  // would hide a real bug in the code.
  if (ABSL_PREDICT_FALSE(slice.size() == 0)) {
    return se::DeviceMemoryBase(nullptr, 0);
  }

  int64_t index = slice.index();
  if (ABSL_PREDICT_FALSE(index < 0 || index >= num_buffers_)) {
    return InvalidArgument(
        "Invalid buffer index %d. It must be in the range [0, %d)", index,
        num_buffers_);
  }
  const se::DeviceMemoryBase& base = buffers_data_[index];

  int64_t offset = slice.offset();
  int64_t extent = offset + slice.size();

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

  return base.GetByteSlice(offset, slice.size());
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE se::DeviceMemoryBase
BufferAllocations::GetDeviceAddressUnchecked(
    BufferAllocation::Index buffer_index) const {
  return buffers_data_[buffer_index];
}

// Unchecked version of `GetDeviceAddress` that does not check the slice
// buffer index, offset and size and assumes they are valid.
inline ABSL_ATTRIBUTE_ALWAYS_INLINE se::DeviceMemoryBase
BufferAllocations::GetDeviceAddressUnchecked(
    BufferAllocation::Slice slice) const {
  return buffers_data_[slice.index()].GetByteSlice(slice.offset(),
                                                   slice.size());
}

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_
