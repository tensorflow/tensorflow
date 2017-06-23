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

InfeedThunk::InfeedThunk(
    tensorflow::gtl::ArraySlice<BufferAllocation::Slice> tuple_element_buffers,
    const BufferAllocation::Slice& destination_buffer,
    const HloInstruction* hlo_instruction)
    : Thunk(Kind::kInfeed, hlo_instruction),
      tuple_element_buffers_(tuple_element_buffers.begin(),
                             tuple_element_buffers.end()),
      destination_buffer_(destination_buffer) {}

tensorflow::Status InfeedThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations,
    perftools::gputools::Stream* stream) {
  VLOG(2) << "Infeeding to GPU ";

  perftools::gputools::DeviceMemoryBase destination_address =
      buffer_allocations.GetDeviceAddress(destination_buffer_);

  InfeedManager* infeed_manager = GetOrCreateInfeedManager();
  std::vector<InfeedBuffer*> infeed_buffers;
  if (ShapeUtil::IsTuple(hlo_instruction()->shape())) {
    CHECK(!ShapeUtil::IsNestedTuple(hlo_instruction()->shape()));
    // Transfer the tuple elements first.
    std::vector<void*> tuple_element_addresses;
    for (BufferAllocation::Slice tuple_element_buffer :
         tuple_element_buffers_) {
      perftools::gputools::DeviceMemoryBase tuple_element_address =
          buffer_allocations.GetDeviceAddress(tuple_element_buffer);

      InfeedBuffer* buffer = infeed_manager->BlockingDequeueBuffer();
      infeed_buffers.push_back(buffer);
      stream->ThenMemcpy(&tuple_element_address, *(buffer->device_memory()),
                         buffer->length());
      tuple_element_addresses.push_back(tuple_element_address.opaque());
    }
    // Transfer the tuple outer buffer.
    auto host_size = tuple_element_addresses.size() * sizeof(void*);
    stream->ThenMemcpy(&destination_address, tuple_element_addresses.data(),
                       host_size);
  } else {
    InfeedBuffer* buffer = infeed_manager->BlockingDequeueBuffer();
    infeed_buffers.push_back(buffer);
    stream->ThenMemcpy(&destination_address, *(buffer->device_memory()),
                       buffer->length());
  }

  if (!stream->BlockHostUntilDone()) {
    return InternalError("Failed to complete data transfer on stream %p",
                         stream);
  }

  infeed_manager->ReleaseBuffers(infeed_buffers);

  VLOG(2) << "Infeeding to GPU complete";
  return tensorflow::Status::OK();
}

}  // namespace gpu
}  // namespace xla
