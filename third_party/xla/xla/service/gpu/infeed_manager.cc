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

#include "xla/service/gpu/infeed_manager.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/literal.h"
#include "xla/service/gpu/xfeed_queue.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/xla_executor_state.h"
#include "xla/stream_executor/gpu/gpu_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace xla {
namespace gpu {

constexpr int kMaxInfeedsInFlight = 8;

InfeedManager::InfeedManager(se::StreamExecutor* executor)
    : BlockingXfeedQueue(/*max_pending_xfeeds=*/kMaxInfeedsInFlight),
      stream_(executor->CreateStream().value()) {}

static absl::StatusOr<se::ScopedDeviceMemory<uint8_t>> CopyBufferToDevice(
    se::Stream* stream, int64_t size, const void* source) {
  if (size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument("GPU infeed of %d bytes exceeds maximum of %d bytes",
                           size, std::numeric_limits<int32_t>::max());
  }

  if (size == 0) {
    return InvalidArgument("Infeed shape needs 0 bytes");
  }

  se::StreamExecutor* executor = stream->parent();
  se::ScopedDeviceMemory<uint8_t> buffer(
      executor, executor->AllocateArray<uint8_t>(size));
  TF_RETURN_IF_ERROR(stream->Memcpy(buffer.ptr(), source, size));

  return std::move(buffer);
}

absl::Status InfeedManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& literal_shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(literal_shape);

  BlockUntilEnqueueSlotAvailable();

  // For a tuple, we transfer each of its elements to the device and enqueue the
  // resulting destination device addresses with the infeed manager.
  ShapeTree<se::ScopedDeviceMemory<uint8_t>> buffer_tree(literal_shape);
  for (auto& leaf : buffer_tree.leaves()) {
    const Shape& sub_shape = ShapeUtil::GetSubshape(literal_shape, leaf.first);
    CHECK(sub_shape.IsArray()) << ShapeUtil::HumanStringWithLayout(sub_shape);
    TF_ASSIGN_OR_RETURN(
        leaf.second,
        CopyBufferToDevice(stream(), ShapeUtil::ByteSizeOf(sub_shape),
                           literal.untyped_data(leaf.first)));
  }

  // TODO(b/30467474): Since this stream is shared across different infeed
  // requests, blocking on the stream might be heavy-handed. Figure out if
  // finer-grained acknowledgement is possible.
  absl::Status block_status = stream()->BlockHostUntilDone();
  if (!block_status.ok()) {
    return Internal("Failed to complete data transfer on stream %p: %s",
                    stream(), block_status.message());
  }

  EnqueueDestination(std::move(buffer_tree));
  return absl::OkStatus();
}

InfeedManager* GetOrCreateInfeedManager(se::StreamExecutor* executor) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  stream_executor::gpu::GpuExecutor* gpu_executor =
      stream_executor::gpu::ExtractGpuExecutor(executor);
  auto* xla_state =
      gpu_executor->getOrCreateXLAState<GpuExecutorXLAState>(executor);
  return xla_state->getOrCreateInfeedManager(executor);
#else   // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return nullptr;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace gpu
}  // namespace xla
