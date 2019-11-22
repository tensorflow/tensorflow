/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/outfeed_thunk.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/outfeed_manager.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

OutfeedThunk::OutfeedThunk(ShapeTree<BufferAllocation::Slice> outfeed_slices,
                           const HloInstruction* hlo_instruction)
    : Thunk(Kind::kOutfeed, hlo_instruction),
      outfeed_slices_(std::move(outfeed_slices)) {}

Status OutfeedThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto& stream = *params.stream;
  auto& buffer_allocations = *params.buffer_allocations;

  VLOG(2) << "Outfeeding from GPU: " << hlo_instruction()->ToString();

  auto op_profiler =
      params.profiler->MakeScopedInstructionProfiler(hlo_instruction());
  OutfeedManager* outfeed_manager = GetOrCreateOutfeedManager();
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* outfeed_buffers =
      outfeed_manager->BlockingGetNextDestination();

  // Nothing to be done for empty tuples.
  if (ShapeUtil::IsEmptyTuple(hlo_instruction()->operand(0)->shape())) {
    return Status::OK();
  }
  CHECK(ShapeUtil::Compatible(hlo_instruction()->operand(0)->shape(),
                              outfeed_buffers->shape()));

  TF_RETURN_IF_ERROR(outfeed_buffers->ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, std::unique_ptr<OutfeedBuffer>* buffer) {
        if (!*buffer) {  // Tuple pointers.
          return Status::OK();
        }

        BufferAllocation::Slice slice = outfeed_slices_.element(index);
        se::DeviceMemoryBase data_address;
        if (slice.allocation()) {
          // If we have a static allocation, read it from there. This avoids
          // synchronizing the host and device just to read a pointer.
          data_address = buffer_allocations.GetDeviceAddress(slice);
        } else {
          // Otherwise we have to read the tuple pointer first.
          CHECK(!index.empty());
          // Copy the parent buffer to the host.
          BufferAllocation::Slice tuple_slice =
              outfeed_slices_.element(ShapeIndexView(index).ConsumeFront());
          if (!tuple_slice.allocation()) {
            return Unimplemented(
                "Nested dynamic tuples are not supported on GPU");
          }
          se::DeviceMemoryBase tuple_address =
              buffer_allocations.GetDeviceAddress(tuple_slice);
          CHECK(tuple_slice.size() % sizeof(void*) == 0)
              << "Tuple size must be a multiple of pointer size";
          std::vector<void*> tuple_element_buffer_addresses(tuple_slice.size() /
                                                            sizeof(void*));
          stream.ThenMemcpy(tuple_element_buffer_addresses.data(),
                            tuple_address, tuple_slice.size());
          TF_RETURN_IF_ERROR(stream.BlockHostUntilDone());
          // The data address is specified by the element of the tuple pointer
          // buffer.
          data_address =
              se::DeviceMemoryBase(tuple_element_buffer_addresses[index.back()],
                                   (*buffer)->length());
        }

        // TODO(b/111309141): Run this on a separate stream so it doesn't block
        // the GPU from doing work during the transfer. This could be handled by
        // making StreamAssignment do something intelligent with outfeed thunks.
        stream
            .ThenMemcpy((*buffer)->destination()->untyped_data(), data_address,
                        (*buffer)->length())
            .ThenDoHostCallback([buffer]() { (*buffer)->Done(); });
        return Status::OK();
      }));

  Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return InternalError("Failed to complete data transfer on stream %p: %s",
                         &stream, block_status.error_message());
  }

  VLOG(2) << "Outfeeding from GPU complete";
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
