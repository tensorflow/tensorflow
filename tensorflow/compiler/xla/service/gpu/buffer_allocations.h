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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_

#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"

namespace xla {
namespace gpu {

// A thread-compatible class that encapsulates the base addresses of the
// allocated device buffers.
class BufferAllocations {
 public:
  BufferAllocations(absl::Span<se::DeviceMemoryBase const> buffers,
                    int device_ordinal,
                    se::DeviceMemoryAllocator* memory_allocator)
      : buffers_(buffers.begin(), buffers.end()),
        device_ordinal_(device_ordinal),
        memory_allocator_(memory_allocator) {}

  BufferAllocations(BufferAllocations&& other) = default;
  BufferAllocations& operator=(BufferAllocations&& other) = default;
  BufferAllocations(const BufferAllocations&) = delete;
  BufferAllocations& operator=(const BufferAllocations&) = delete;

  se::DeviceMemoryAllocator* memory_allocator() const {
    return memory_allocator_;
  }
  int device_ordinal() const { return device_ordinal_; }

  // Returns the device address of buffer `buffer_index`. `buffer_index` must be
  // a valid index, i.e., in [0, buffer_count). This function returns null if
  // `buffer_index` is not assigned to a buffer address.
  se::DeviceMemoryBase GetDeviceAddress(
      BufferAllocation::Index buffer_index) const;

  // Returns a mutable value for the allocation at a given `buffer_index`.
  se::DeviceMemoryBase& GetMutableDeviceAddress(
      BufferAllocation::Index buffer_index);

  // Same as above, but also adjusts the returned address for the offset and
  // size contained in the given slice.
  se::DeviceMemoryBase GetDeviceAddress(
      const BufferAllocation::Slice& buffer_slice) const;

  // Tears down all buffers allocated by this object that are not in
  // `live_addresses`.
  Status TearDown(const std::set<se::DeviceMemoryBase>& live_addresses,
                  absl::Span<const BufferAllocation> allocations);

  std::string ToString() {
    std::string out;
    for (BufferAllocation::Index i = 0; i < buffers_.size(); ++i) {
      const auto& buf = buffers_[i];
      absl::StrAppendFormat(&out, "Buffer %d -> %p (%d B)", i, buf.opaque(),
                            buf.size());
    }
    return out;
  }

 private:
  // An array of device pointers that stores the address of each buffer
  // indexed by Index. Each element can point to a temporary buffer, an
  // input buffer, or nullptr if no buffer is needed for that Index.
  std::vector<se::DeviceMemoryBase> buffers_;
  int device_ordinal_;
  se::DeviceMemoryAllocator* memory_allocator_;
};

// LLVM and PTXAS don't deal well with large constants, so we only emit very
// small constants directly in LLVM IR.  Larger constants are emitted with zero
// initializers in LLVM IR and are later overwritten when the PTX/CUBIN is
// loaded.
bool ShouldEmitLiteralInLlvmIr(const Literal& literal);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_
