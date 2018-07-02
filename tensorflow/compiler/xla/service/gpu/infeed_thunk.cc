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

#include "tensorflow/compiler/xla/service/gpu/infeed_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

InfeedThunk::InfeedThunk(
    const ShapeTree<BufferAllocation::Slice>& infeed_slices,
    const HloInstruction* hlo_instruction)
    : Thunk(Kind::kInfeed, hlo_instruction), infeed_slices_(infeed_slices) {}

Status InfeedThunk::ExecuteOnStream(const BufferAllocations& buffer_allocations,
                                    se::Stream* stream,
                                    HloExecutionProfiler* profiler) {
  VLOG(2) << "Infeeding to GPU ";

  auto op_profiler = profiler->MakeScopedInstructionProfiler(hlo_instruction());
  // First copy the infeed data which is element 0 of the infeed instruction's
  // two-tuple output (the other element is a token).
  se::DeviceMemoryBase data_address =
      buffer_allocations.GetDeviceAddress(infeed_slices_.element({0}));
  InfeedManager* infeed_manager = GetOrCreateInfeedManager();
  std::vector<InfeedBuffer*> infeed_buffers;
  const Shape& data_shape =
      ShapeUtil::GetTupleElementShape(hlo_instruction()->shape(), 0);
  if (ShapeUtil::IsTuple(data_shape)) {
    CHECK(!ShapeUtil::IsNestedTuple(data_shape));
    // Transfer the tuple elements first.
    std::vector<void*> tuple_element_addresses;
    for (int i = 0; i < ShapeUtil::TupleElementCount(data_shape); ++i) {
      const BufferAllocation::Slice& tuple_element_buffer =
          infeed_slices_.element({0, i});
      se::DeviceMemoryBase tuple_element_address =
          buffer_allocations.GetDeviceAddress(tuple_element_buffer);

      InfeedBuffer* buffer = infeed_manager->BlockingDequeueBuffer();
      infeed_buffers.push_back(buffer);
      stream->ThenMemcpy(&tuple_element_address, *(buffer->device_memory()),
                         buffer->length());
      tuple_element_addresses.push_back(tuple_element_address.opaque());
    }
    // Transfer the tuple outer buffer.
    auto host_size = tuple_element_addresses.size() * sizeof(void*);
    stream->ThenMemcpy(&data_address, tuple_element_addresses.data(),
                       host_size);
  } else {
    InfeedBuffer* buffer = infeed_manager->BlockingDequeueBuffer();
    infeed_buffers.push_back(buffer);
    stream->ThenMemcpy(&data_address, *(buffer->device_memory()),
                       buffer->length());
  }

  // Construct top-level tuple of infeed containing the data and the token. Use
  // a nullptr for the token, it should never be dereferenced.
  std::vector<void*> infeed_addresses = {data_address.opaque(), nullptr};
  se::DeviceMemoryBase top_level_address =
      buffer_allocations.GetDeviceAddress(infeed_slices_.element({}));
  stream->ThenMemcpy(&top_level_address, infeed_addresses.data(),
                     2 * sizeof(void*));

  Status block_status = stream->BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         stream, block_status.error_message().c_str());
  }

  infeed_manager->ReleaseBuffers(infeed_buffers);

  VLOG(2) << "Infeeding to GPU complete";
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
