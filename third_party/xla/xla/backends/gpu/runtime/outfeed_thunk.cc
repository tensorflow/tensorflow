/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/outfeed_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/gpu_transfer_manager.h"
#include "xla/service/gpu/outfeed_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

OutfeedThunk::OutfeedThunk(ThunkInfo thunk_info,
                           std::vector<ShapedSlice> source_slices)
    : Thunk(Kind::kOutfeed, thunk_info),
      source_slices_(std::move(source_slices)) {}

absl::Status OutfeedThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream& stream = *params.stream;
  const BufferAllocations& buffer_allocations = *params.buffer_allocations;

  VLOG(2) << "Outfeeding from GPU";

  OutfeedManager* outfeed_manager =
      GpuTransferManager::GetOrCreateOutfeedManager(stream.parent());
  ShapeTree<std::unique_ptr<OutfeedBuffer>>* output_buffers =
      outfeed_manager->BlockingGetNextDestination();

  // Nothing to be done for an outfeed with no inputs.
  // Note: Cannot do this before `BlockingGetNextDestination` above to dequeue
  // an entry from the outfeed manager.
  if (source_slices_.empty()) {
    return absl::OkStatus();
  }

  const int64_t leaf_count = output_buffers->leaf_count();
  TF_RET_CHECK(source_slices_.size() == leaf_count)
      << "Mismatch between number of outfeed inputs (" << source_slices_.size()
      << ") and outputs (" << leaf_count << ")";

  auto output_leaf_it = output_buffers->leaf_begin();
  for (int64_t index = 0; index < leaf_count; ++index) {
    // Assert that the shapes are compatible.
    const ShapeIndex& shape_index = output_leaf_it->first;
    std::unique_ptr<OutfeedBuffer>& buffer = output_leaf_it->second;

    // NOTE: This code needs deal with the `output_buffers` object getting
    // deleted when its executing. Specifically, objects in the outfeed queue
    // are pointers to instance of stack allocated objects in
    // `GpuTransferManager::TransferLiteralFromOutfeed`. When all leaf node
    // buffers are notified via "buffer->Done()" below in the stream host
    // callback, `TransferLiteralFromOutfeed` deletes this stack allocated
    // object when it returns. This means that its possible that during the last
    // iteration, after the call to "buffer->Done()" is scheduled onto the
    // stream, the `output_buffers` object might get deleted, so we should avoid
    // accessing the object after that.
    //
    // To achieve that, increment the leaf iterator here before the last "Done"
    // is enqueued, instead of in the loop increment, which would be after the
    // "Done" is scheduled.
    ++output_leaf_it;
    const Shape& output_shape =
        ShapeUtil::GetSubshape(output_buffers->shape(), shape_index);
    TF_RET_CHECK(
        ShapeUtil::ReshapeIsBitcast(source_slices_[index].shape, output_shape))
        << "Mismatch between outfeed output buffer shape "
        << ShapeUtil::HumanStringWithLayout(output_shape)
        << " and outfeed source buffer shape "
        << ShapeUtil::HumanStringWithLayout(source_slices_[index].shape);

    BufferAllocation::Slice source_slice = source_slices_[index].slice;
    if (!source_slice.allocation())
      return Internal("outfeed source missing buffer allocation");
    se::DeviceMemoryBase data_address =
        buffer_allocations.GetDeviceAddress(source_slice);

    // TODO(b/111309141): Run this on a separate stream so it doesn't block
    // the GPU from doing work during the transfer.
    TF_RETURN_IF_ERROR(stream.Memcpy(buffer->destination()->untyped_data(),
                                     data_address, buffer->length()));
    TF_RETURN_IF_ERROR(stream.DoHostCallback([&buffer]() { buffer->Done(); }));
  }

  absl::Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return Internal("Failed to complete data transfer on stream %p: %s",
                    &stream, block_status.message());
  }

  VLOG(2) << "Outfeeding from GPU complete";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
