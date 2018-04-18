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

#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"

#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

HostToDeviceCopyThunk::HostToDeviceCopyThunk(
    const void* source_address,
    const BufferAllocation::Slice& destination_buffer, uint64 mem_size,
    const HloInstruction* hlo_instruction)
    : Thunk(Kind::kCopy, hlo_instruction),
      source_address_(source_address),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {}

tensorflow::Status HostToDeviceCopyThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  se::DeviceMemoryBase destination_data =
      buffer_allocations.GetDeviceAddress(destination_buffer_);
  stream->ThenMemcpy(&destination_data, source_address_, mem_size_);
  return tensorflow::Status::OK();
}

DeviceToDeviceCopyThunk::DeviceToDeviceCopyThunk(
    const BufferAllocation::Slice& source_buffer,
    const BufferAllocation::Slice& destination_buffer, uint64 mem_size,
    const HloInstruction* hlo_instruction)
    : Thunk(Kind::kCopy, hlo_instruction),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {}

tensorflow::Status DeviceToDeviceCopyThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream) {
  se::DeviceMemoryBase destination_data =
      buffer_allocations.GetDeviceAddress(destination_buffer_);
  se::DeviceMemoryBase source_data =
      buffer_allocations.GetDeviceAddress(source_buffer_);
  stream->ThenMemcpy(&destination_data, source_data, mem_size_);
  return tensorflow::Status::OK();
}
}  // namespace gpu
}  // namespace xla
