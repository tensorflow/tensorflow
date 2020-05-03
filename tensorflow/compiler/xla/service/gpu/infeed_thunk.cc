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

Status InfeedThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  VLOG(2) << "Infeeding to GPU: " << hlo_instruction()->ToString();

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  ShapeTree<InfeedBuffer> infeed_buffers =
      GetOrCreateInfeedManager()->BlockingGetNextDestination();

  // infeed_slices_'s shape should be a tuple of shape (buffers, token).
  const auto& infeed_shape = infeed_slices_.shape();
  TF_RET_CHECK(infeed_shape.IsTuple())
      << ShapeUtil::HumanStringWithLayout(infeed_shape);
  TF_RET_CHECK(infeed_shape.tuple_shapes().size() == 2)
      << ShapeUtil::HumanStringWithLayout(infeed_shape);
  TF_RET_CHECK(infeed_shape.tuple_shapes(1).IsToken())
      << ShapeUtil::HumanStringWithLayout(infeed_shape);
  TF_RET_CHECK(
      ShapeUtil::Equal(infeed_buffers.shape(), infeed_shape.tuple_shapes(0)))
      << "Expected infeed of shape "
      << ShapeUtil::HumanStringWithLayout(infeed_shape.tuple_shapes(0))
      << " but was "
      << ShapeUtil::HumanStringWithLayout(infeed_buffers.shape());

  {
    // The infeed buffer has an extra outer tuple with a token. Adjust the index
    // accordingly.
    ShapeIndex index = {0};
    std::function<void(std::vector<void*>*)> copy_tuple_contents =
        [&](std::vector<void*>* tuple_element_addresses) {
          const Shape& shape = ShapeUtil::GetSubshape(infeed_buffers.shape(),
                                                      ShapeIndexView(index, 1));
          // For the leaf buffers of the tuple copy the elements directly.
          if (shape.IsArray()) {
            const BufferAllocation::Slice& tuple_element_buffer =
                infeed_slices_.element(index);
            se::DeviceMemoryBase tuple_element_address =
                buffer_allocations.GetDeviceAddress(tuple_element_buffer);

            InfeedBuffer* buffer =
                infeed_buffers.mutable_element(ShapeIndexView(index, 1));
            stream.ThenMemcpy(&tuple_element_address,
                              *(buffer->device_memory()), buffer->length());
            tuple_element_addresses->push_back(tuple_element_address.opaque());
            return;
          }

          const int64 tuple_element_count = ShapeUtil::TupleElementCount(shape);
          index.push_back(0);
          std::vector<void*> inner_tuple_element_addresses;
          for (int64 i = 0; i < tuple_element_count; ++i) {
            index.back() = i;
            copy_tuple_contents(&inner_tuple_element_addresses);
          }
          index.pop_back();

          // Create a buffer of pointers for non-leaf buffers.
          CHECK_EQ(tuple_element_count, inner_tuple_element_addresses.size());
          auto host_size = inner_tuple_element_addresses.size() * sizeof(void*);
          se::DeviceMemoryBase tuple_address =
              buffer_allocations.GetDeviceAddress(
                  infeed_slices_.element(index));
          stream.ThenMemcpy(&tuple_address,
                            inner_tuple_element_addresses.data(), host_size);
          tuple_element_addresses->push_back(tuple_address.opaque());
        };

    std::vector<void*> tuple_element_addresses;
    copy_tuple_contents(&tuple_element_addresses);
    CHECK_EQ(1, tuple_element_addresses.size());
  }

  // Construct top-level tuple of infeed containing the data and the token. Use
  // a nullptr for the token, it should never be dereferenced.
  se::DeviceMemoryBase data_address =
      buffer_allocations.GetDeviceAddress(infeed_slices_.element({0}));
  void* infeed_addresses[] = {data_address.opaque(), nullptr};
  se::DeviceMemoryBase top_level_address =
      buffer_allocations.GetDeviceAddress(infeed_slices_.element({}));
  stream.ThenMemcpy(&top_level_address, infeed_addresses, 2 * sizeof(void*));

  Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         &stream, block_status.error_message());
  }

  VLOG(2) << "Infeeding to GPU complete";
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
