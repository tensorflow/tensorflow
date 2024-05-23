/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/runtime/infeed_thunk.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/gpu/infeed_manager.h"
#include "xla/service/gpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {
namespace gpu {

InfeedThunk::InfeedThunk(ThunkInfo thunk_info,
                         std::vector<ShapedSlice> dest_slices)
    : Thunk(Kind::kInfeed, thunk_info), dest_slices_(std::move(dest_slices)) {}

absl::Status InfeedThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::Stream& stream = *params.stream;
  const BufferAllocations& buffer_allocations = *params.buffer_allocations;

  VLOG(2) << "Infeeding to GPU";
  ShapeTree<se::DeviceMemoryHandle> source_buffers =
      GetOrCreateInfeedManager(stream.parent())->BlockingGetNextDestination();

  size_t index = 0;
  for (auto& source : source_buffers.leaves()) {
    // Assert that the shapes are compatible.
    const ShapeIndex& shape_index = source.first;
    se::DeviceMemoryHandle& buffer = source.second;
    const Shape& source_shape =
        ShapeUtil::GetSubshape(source_buffers.shape(), shape_index);
    TF_RET_CHECK(
        ShapeUtil::ReshapeIsBitcast(dest_slices_[index].shape, source_shape))
        << "Mismatch between infeed source buffer shape "
        << ShapeUtil::HumanStringWithLayout(source_shape)
        << " and infeed dest buffer shape "
        << ShapeUtil::HumanStringWithLayout(dest_slices_[index].shape);
    se::DeviceMemoryBase dest_address =
        buffer_allocations.GetDeviceAddress(dest_slices_[index++].slice);
    TF_RETURN_IF_ERROR(
        stream.Memcpy(&dest_address, buffer.memory(), buffer.memory().size()));
  }

  // Make sure that all dest slices have been copied into.
  CHECK_EQ(index, dest_slices_.size())
      << "Infeed did not populate all destination buffers";

  absl::Status block_status = stream.BlockHostUntilDone();
  if (!block_status.ok()) {
    return Internal("Failed to complete data transfer on stream %p: %s",
                    &stream, block_status.message());
  }

  VLOG(2) << "Infeeding to GPU complete";
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla
