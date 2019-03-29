/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDZONE_ALLOCATOR_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDZONE_ALLOCATOR_H_

#include <vector>

#include "tensorflow/compiler/xla/service/device_memory_allocator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_constants.h"
#include "tensorflow/compiler/xla/service/owning_device_memory.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// An allocator that allocates a bit of extra memory around the beginning/end of
// every allocation and can check that this memory is unmodified.
//
// This can be used to check for out-of-bounds writes, and, if the redzone is
// filled with a sufficiently "ugly" pattern, may also be able to check for
// out-of-bounds reads.  The default fill pattern of -1 is an unusual NaN
// pattern when interpreted as a floating-point number, so hopefully works for
// out-of-bounds reads and writes in those cases.
//
// This class implements se::ScratchAllocator, so can be used to allocate temp
// memory for cudnn convolutions.
class RedzoneAllocator : public se::ScratchAllocator {
 public:
  RedzoneAllocator(int device_ordinal, DeviceMemoryAllocator* memory_allocator,
                   int64 redzone_size = 1 << 23,  // 8MiB per side, 16MiB total
                   uint8 redzone_pattern = -1)
      : device_ordinal_(device_ordinal),
        redzone_size_(
            RoundUpToNearest(redzone_size, kXlaAllocatedBufferAlignBytes)),
        redzone_pattern_(redzone_pattern),
        memory_allocator_(memory_allocator) {}

  // Redzones don't count towards the memory limit.
  int64 GetMemoryLimitInBytes(se::Stream* stream) override {
    return 1LL << 32;  // 4GB.  TODO(jlebar): Tune this?
  }
  int64 TotalAllocatedBytesExcludingRedzones() const {
    return allocated_bytes_excluding_redzones_;
  }

  StatusOr<se::DeviceMemory<uint8>> AllocateBytes(se::Stream* stream,
                                                  int64 byte_size) override;

  // Determines whether redzones around all allocated buffers are unmodified.
  Status CheckRedzones(se::Stream* stream) const;

 private:
  // Checks that one buffer's redzones are unmodified.  buf should point to the
  // user-editable buffer, i.e. it should not include redzones.
  Status CheckBufferRedzones(se::DeviceMemoryBase buf,
                             se::Stream* stream) const;

  const int device_ordinal_;

  // Redzone size on *one side* of allocation.
  //
  // Must be a multiple of kXlaAllocatedBufferAlignBytes, otherwise the buffers
  // returned to users will be misaligned.
  const int64 redzone_size_;

  const uint8 redzone_pattern_;
  DeviceMemoryAllocator* memory_allocator_;

  // The second element of the pair is the size of the user allocation.  This
  // isn't necessarily just first.size() - 2 * redzone_size_ because when the
  // user allocation size is not a multiple of 4 bytes, we round up the size of
  // the RHS redzone.
  std::vector<std::pair<OwningDeviceMemory, int64>> allocated_buffers_;

  int64 allocated_bytes_excluding_redzones_ = 0;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_REDZONE_ALLOCATOR_H_
