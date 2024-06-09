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

#include "xla/service/cpu/runtime/buffer_allocations.h"

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/statusor.h"

namespace xla::cpu {

absl::StatusOr<se::DeviceMemoryBase> BufferAllocations::GetDeviceAddress(
    BufferAllocation::Index buffer_index) const {
  if (buffer_index < 0 || buffer_index >= buffers_.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid buffer_index ", buffer_index,
        " value. It must be in the range [0, ", buffers_.size(), ")"));
  }

  return buffers_[buffer_index].AsDeviceMemoryBase();
}

absl::StatusOr<se::DeviceMemoryBase> BufferAllocations::GetDeviceAddress(
    const BufferAllocation::Slice& buffer_slice) const {
  // Handle empty slices explicitly and return a null pointer device memory to
  // guarantee that we do not accidentally write through the empty slice which
  // would hide a real bug in the code.
  if (buffer_slice.size() == 0) {
    return se::DeviceMemoryBase(nullptr, 0);
  }

  int64_t index = buffer_slice.index();
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase base, GetDeviceAddress(index));

  int64_t offset = buffer_slice.offset();
  int64_t extent = offset + buffer_slice.size();

  if (offset < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Buffer slice offset ", offset, " must be non-negative"));
  }

  if (offset >= base.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Buffer slice offset ", offset, " is out of range for buffer #", index,
        " of size ", base.size()));
  }

  if (extent > base.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Buffer slice extent ", extent, " is out of range for buffer #", index,
        " of size ", base.size()));
  }

  return base.GetByteSlice(offset, buffer_slice.size());
}

}  // namespace xla::cpu
