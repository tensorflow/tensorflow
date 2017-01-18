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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TEMP_BUFFER_OFFSETS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TEMP_BUFFER_OFFSETS_H_

#include <map>

#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// GpuExecutable merges all temporary buffers into one memory block. This class
// stores the offset of each temporary buffer in that memory block.
class TempBufferOffsets {
 public:
  explicit TempBufferOffsets(const BufferAssignment& buffer_assignment);

  int64 GetOffset(BufferAllocation::Index index) const;
  int64 TotalSizeInBytes() const { return total_size_of_temp_buffers_; }

 private:
  std::map<BufferAllocation::Index, int64> buffer_index_to_offset_;

  // The total size of all temporary buffers. This includes paddings that are
  // necessary for alignment.
  int64 total_size_of_temp_buffers_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TEMP_BUFFER_OFFSETS_H_
