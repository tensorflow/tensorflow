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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/stream_executor/device_memory.h"

namespace xla::cpu {

// Buffer allocation is a container for device buffers allocated for a
// particular XLA execution. Buffers are indexed by the buffer allocation index.
//
// TODO(b/342513610): BufferAllocations should be unified with a same class in
// the XLA:GPU runtime, probably as a part of `buffer_assignment.h`.
class BufferAllocations {
 public:
  explicit BufferAllocations(absl::Span<const MaybeOwningDeviceMemory> buffers)
      : buffers_(buffers) {}

  // Returns the device address of buffer `buffer_index`. `buffer_index` must be
  // a valid index, i.e., in [0, buffer_count).
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      BufferAllocation::Index buffer_index) const;

  // Same as above, but also adjusts the returned address for the offset and
  // size contained in the given slice.
  absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
      const BufferAllocation::Slice& buffer_slice) const;

 private:
  // TODO(ezhulenev): Make BufferAllocations an owner of the buffers.
  absl::Span<const MaybeOwningDeviceMemory> buffers_;  // not owned
};

}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_RUNTIME_BUFFER_ALLOCATIONS_H_
