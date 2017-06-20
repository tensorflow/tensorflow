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

#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

InfeedThunk::InfeedThunk(const BufferAllocation::Slice& destination_buffer,
                         uint64 mem_size, const HloInstruction* hlo_instruction)
    : Thunk(Kind::kInfeed, hlo_instruction),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {}

tensorflow::Status InfeedThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations,
    perftools::gputools::Stream* stream) {
  VLOG(2) << "Infeeding to GPU ";
  perftools::gputools::DeviceMemoryBase destination_data =
      buffer_allocations.GetDeviceAddress(destination_buffer_);

  InfeedManager* infeed_manager = GetOrCreateInfeedManager();
  InfeedBuffer* buffer = infeed_manager->BlockingDequeueBuffer();
  CHECK_EQ(buffer->length(), mem_size_);
  stream->ThenMemcpy(&destination_data, *(buffer->device_memory()),
                     buffer->length());
  if (!stream->BlockHostUntilDone()) {
    return InternalError("Failed to complete data transfer on stream %p",
                         stream);
  }
  // Since Infeeds are totally ordered, no other infeed should sneak
  // in and we should be able to release the same buffer we dequeued.
  infeed_manager->ReleaseCurrentBuffer(buffer->device_memory());
  return tensorflow::Status::OK();
}

}  // namespace gpu
}  // namespace xla
