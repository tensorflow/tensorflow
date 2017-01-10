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

#include "tensorflow/compiler/xla/service/gpu/temp_buffer_offsets.h"

#include "tensorflow/compiler/xla/map_util.h"

namespace xla {
namespace gpu {

namespace {
int64 RoundUpToAlignment(int64 value) {
  // Any address of a variable residing in global memory or returned by one of
  // the memory allocation routines from the driver or runtime API is always
  // aligned to at least 256 bytes.
  // (http://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses)
  static constexpr int64 kCudaMallocAlignment = 256;
  return (value + kCudaMallocAlignment - 1) / kCudaMallocAlignment *
         kCudaMallocAlignment;
}
}  // namespace

TempBufferOffsets::TempBufferOffsets(
    const BufferAssignment& buffer_assignment) {
  total_size_of_temp_buffers_ = 0;
  for (auto i = 0; i < buffer_assignment.Allocations().size(); ++i) {
    const BufferAllocation& allocation = buffer_assignment.GetAllocation(i);
    if (allocation.IsPreallocatedTempBuffer()) {
      InsertOrDie(&buffer_index_to_offset_, i, total_size_of_temp_buffers_);
      total_size_of_temp_buffers_ += RoundUpToAlignment(allocation.size());
    }
  }
}

int64 TempBufferOffsets::GetOffset(BufferAllocation::Index index) const {
  return FindOrDie(buffer_index_to_offset_, index);
}

}  // namespace gpu
}  // namespace xla
