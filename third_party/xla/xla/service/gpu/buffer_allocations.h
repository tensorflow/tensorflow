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

#ifndef XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_
#define XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"

namespace xla {
namespace gpu {

// A thread-compatible class that encapsulates the base addresses of the
// allocated device buffers.
class BufferAllocations {
 public:
  // This special address is used to indicate that the allocation is not
  // allocated at construction time and instead will be lazily allocated and
  // owned by the XLA executable itself (we use this special marker to handle
  // buffer allocations allocated within command buffers, which for CUDA
  // backends means that buffer allocation is done via memory allocation node).
  //
  // TODO(ezhulenev): Replace magic bit pattern with std::optional or
  // std::variant to distinguish external allocations from a regular ones.
  static constexpr uintptr_t kExternalAllocationMarker = 0xDEADBEEF;

  // A virtual base class for external allocations that provides a mapping
  // from a buffer index to an externally-managed device memory.
  class ExternalAllocations {
   public:
    virtual ~ExternalAllocations() = default;

    // Return a device address for a given buffer allocation. Returns error if
    // corresponding allocation is not yet allocated.
    virtual absl::StatusOr<se::DeviceMemoryBase> GetDeviceAddress(
        BufferAllocation::Index index) const = 0;

    // Adds an external allocation for a given buffer index. Returns error if
    // allocation already exists.
    virtual absl::Status AddAllocation(BufferAllocation::Index index,
                                       se::DeviceMemoryBase memory) = 0;

    // Erases an external allocation for a given buffer index. Returns error if
    // allocation does not exists.
    virtual absl::Status EraseAllocation(BufferAllocation::Index index) = 0;
  };

  BufferAllocations(absl::Span<se::DeviceMemoryBase const> buffers,
                    int device_ordinal,
                    se::DeviceMemoryAllocator* memory_allocator,
                    ExternalAllocations* external_allocations = nullptr)
      : buffers_(buffers.begin(), buffers.end()),
        device_ordinal_(device_ordinal),
        memory_allocator_(memory_allocator),
        external_allocations_(external_allocations) {}

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

  // Add new allocation allocated by external allocator.
  absl::Status AddExternalAllocation(BufferAllocation::Index index,
                                     se::DeviceMemoryBase memory) const;

  // Remove allocation freed by external allocator.
  absl::Status EraseExternalAllocation(BufferAllocation::Index index) const;

  // Tears down all buffers allocated by this object that are not in
  // `live_addresses`.
  absl::Status TearDown(const std::set<se::DeviceMemoryBase>& live_addresses,
                        absl::Span<const BufferAllocation> allocations);

  std::string ToString() const {
    std::string out;
    for (BufferAllocation::Index i = 0; i < buffers_.size(); ++i) {
      const auto& buf = buffers_[i];
      absl::StrAppendFormat(&out, "Buffer %d -> %p (%d B)", i, buf.opaque(),
                            buf.size());
    }
    return out;
  }

  size_t size() const { return buffers_.size(); }

 private:
  // An array of device pointers that stores the address of each buffer
  // indexed by Index. Each element can point to a temporary buffer, an
  // input buffer, or nullptr if no buffer is needed for that Index.

  // a special address (se::kExternalAllocationMarker) with non-zero size buffer
  // is assumed to be lazily allocated buffer, and will be allocated through
  // command buffer Allocate command during runtime.
  std::vector<se::DeviceMemoryBase> buffers_;
  int device_ordinal_;
  se::DeviceMemoryAllocator* memory_allocator_;

  // For buffer address that marked as ExternalAllocations, tracks its real
  // address here.
  ExternalAllocations* external_allocations_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_
