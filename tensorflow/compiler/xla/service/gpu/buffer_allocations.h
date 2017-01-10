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

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/gpu/temp_buffer_offsets.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// A thread-compatible class that encapsulates the base addresses of the
// allocated device buffers.
class BufferAllocations {
 public:
  // This inner class encapsulates methods that build a BufferAllocations from
  // the given buffer assignment.
  class Builder {
   public:
    // Registers preallocated buffers (such as parameter addresses and
    // user-specified result buffers) to the given buffer index. The builder
    // will skip allocating buffers for registered buffer indices.
    void RegisterBuffer(BufferAllocation::Index index,
                        perftools::gputools::DeviceMemoryBase address);

    // Builds a BufferAllocations object from the given buffer assignment.
    // `memory_allocator` is what this function uses to allocate device memory.
    // `device_ordinal` is the number of the device this function allocates
    // memory on.
    StatusOr<std::unique_ptr<BufferAllocations>> Build(
        const BufferAssignment& buffer_assignment,
        const TempBufferOffsets& temp_buffer_offsets, int device_ordinal,
        DeviceMemoryAllocator* memory_allocator);

   private:
    std::map<BufferAllocation::Index, perftools::gputools::DeviceMemoryBase>
        registered_buffers_;
  };

  BufferAllocations(const BufferAllocations&) = delete;
  BufferAllocations& operator=(const BufferAllocations&) = delete;

  DeviceMemoryAllocator* memory_allocator() const { return memory_allocator_; }
  int device_ordinal() const { return device_ordinal_; }

  // Returns the device address of buffer `buffer_index`. `buffer_index` must be
  // a valid index, i.e., in [0, buffer_count). This function returns null if
  // `buffer_index` is not assigned to a buffer address.
  perftools::gputools::DeviceMemoryBase GetDeviceAddress(
      BufferAllocation::Index buffer_index) const;

  perftools::gputools::DeviceMemoryBase GetTempBufferBase() const {
    return temp_buffer_base_;
  }

  // Tears down all buffers allocated by this object that are not in
  // `live_addresses`.
  tensorflow::Status TearDown(
      const std::set<perftools::gputools::DeviceMemoryBase>& live_addresses,
      const BufferAssignment& buffer_assignment);

 private:
  BufferAllocations(BufferAllocation::Index buffer_count,
                    perftools::gputools::DeviceMemoryBase temp_buffer_base,
                    int device_ordinal, DeviceMemoryAllocator* memory_allocator)
      : buffers_(buffer_count),
        temp_buffer_base_(
            perftools::gputools::DeviceMemory<void*>(temp_buffer_base)),
        device_ordinal_(device_ordinal),
        memory_allocator_(memory_allocator) {}

  // Sets the device address of buffer `buffer_index`.
  void SetBuffer(BufferAllocation::Index buffer_index,
                 perftools::gputools::DeviceMemoryBase buffer);

  // An array of device pointers that stores the address of each buffer
  // indexed by Index. Each element can point to a temporary buffer, an
  // input buffer, or nullptr if no buffer is needed for that Index.
  std::vector<perftools::gputools::DeviceMemoryBase> buffers_;

  // The base address of the memory block that contains all temporary buffers.
  perftools::gputools::DeviceMemory<void*> temp_buffer_base_;

  int device_ordinal_;

  DeviceMemoryAllocator* memory_allocator_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_BUFFER_ALLOCATIONS_H_
