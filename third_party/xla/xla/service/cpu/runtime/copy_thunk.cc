/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/cpu/runtime/copy_thunk.h"

#define EIGEN_USE_THREADS

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/pjrt/transpose.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/util.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<CopyThunk>> CopyThunk::Create(
    Info info, BufferAllocation::Slice source_buffer, const Shape& source_shape,
    BufferAllocation::Slice destination_buffer,
    const Shape& destination_shape) {
  if (!ShapeUtil::Compatible(source_shape, destination_shape)) {
    return InvalidArgument(
        "Source shape %s must be compatible with destination shape %s",
        source_shape.ToString(true), destination_shape.ToString(true));
  }

  return absl::WrapUnique(new CopyThunk(std::move(info), source_buffer,
                                        source_shape, destination_buffer,
                                        destination_shape));
}

CopyThunk::CopyThunk(Info info, BufferAllocation::Slice source_buffer,
                     const Shape& source_shape,
                     BufferAllocation::Slice destination_buffer,
                     const Shape& destination_shape)
    : Thunk(Kind::kCopy, std::move(info)),
      source_buffer_(source_buffer),
      source_shape_(source_shape),
      destination_buffer_(destination_buffer),
      destination_shape_(destination_shape) {
  if (source_shape_ != destination_shape_) {
    TransposePlan::Options options;
    options.elem_size_in_bytes =
        ShapeUtil::ByteSizeOfPrimitiveType(source_shape_.element_type());
    options.dims = source_shape_.dimensions();

    auto byte_strides = ShapeUtil::ByteStrides(source_shape_);
    options.input_layout = TransposePlan::Striding{*byte_strides};

    absl::InlinedVector<int64_t, 4> permutation(options.dims.size());
    absl::c_reverse_copy(destination_shape_.layout().minor_to_major(),
                         permutation.begin());
    options.permutation = permutation;

    transpose_plan_ = TransposePlan::Create(options).value();
  }
}

namespace {
struct ParallelBlockInfo {
  int64_t size_in_bytes;
  int64_t block_size;
  int64_t block_count;
};
}  // namespace

// Prefer single-threaded memcpy for small copies.
static constexpr int64_t kMinParallelCopySize = 1024 * 1024;

// Make block size a multiple of 1024 to match AVX2/AVX512 vector sizes.
static constexpr int64_t kBlockSizeAlign = 1024;

// Do not run more than 8 parallel copy blocks at a time.
static constexpr int64_t kMaxParallelCopyBlocks = 8;

static ParallelBlockInfo ComputeParallelBlockInfo(int64_t size_in_bytes) {
  int64_t num_blocks = std::min(
      kMaxParallelCopyBlocks, CeilOfRatio(size_in_bytes, kMinParallelCopySize));

  if (num_blocks == 1) {
    return ParallelBlockInfo{size_in_bytes, size_in_bytes, 1};
  }

  int64_t block_size = RoundUpTo(size_in_bytes / num_blocks, kBlockSizeAlign);
  return ParallelBlockInfo{size_in_bytes, block_size,
                           CeilOfRatio(size_in_bytes, block_size)};
}

static std::tuple<void*, void*, int64_t> GetBlockCopyParameters(
    const ParallelBlockInfo& info, int64_t block_index,
    se::DeviceMemoryBase destination, se::DeviceMemoryBase source) {
  CHECK_LT(block_index, info.block_count);

  int64_t offset = block_index * info.block_size;
  CHECK_LT(offset, info.size_in_bytes);

  int64_t size = std::min(info.block_size, info.size_in_bytes - offset);
  CHECK_LE(size, info.block_size);

  std::byte* dst = reinterpret_cast<std::byte*>(destination.opaque());
  std::byte* src = reinterpret_cast<std::byte*>(source.opaque());

  return {dst + offset, src + offset, size};
}

tsl::AsyncValueRef<Thunk::ExecuteEvent> CopyThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase source_data,
      params.buffer_allocations->GetDeviceAddress(source_buffer_));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase destination_data,
      params.buffer_allocations->GetDeviceAddress(destination_buffer_));

  // TODO(ezhulenev): Add extra checks for buffer aliasing, as we rely on the
  // fact that buffers do not alias each other to run copy in parallel.

  VLOG(3) << absl::StreamFormat("Copy buffer: use_transpose=%s",
                                transpose_plan_ ? "true" : "false");
  VLOG(3) << absl::StreamFormat(
      "  src: %s in slice %s (%p)", source_shape_.ToString(true),
      source_buffer_.ToString(), source_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  dst: %s in slice %s (%p)", destination_shape_.ToString(true),
      destination_buffer_.ToString(), destination_data.opaque());

  // Skip no-op copies.
  if (source_data.size() == 0 && destination_data.size() == 0) {
    return OkExecuteEvent();
  }

  // Use prepared transpose plan to copy data if copy requires changing layout.
  if (transpose_plan_) {
    transpose_plan_->Execute(source_data.opaque(), destination_data.opaque(),
                             [](std::function<void()> fn) { fn(); });
    return OkExecuteEvent();
  }

  // Decide if we want to execute copy in parallel.
  int64_t size_in_bytes = ShapeUtil::ByteSizeOf(source_shape_);
  ParallelBlockInfo info = ComputeParallelBlockInfo(size_in_bytes);

  if (params.intra_op_threadpool && info.block_count > 1) {
    auto counter = std::make_shared<std::atomic<int64_t>>(info.block_count);
    auto event = tsl::MakeConstructedAsyncValueRef<ExecuteEvent>();

    auto execute = [=](int64_t block_index) {
      auto [dst, src, size] = GetBlockCopyParameters(
          info, block_index, destination_data, source_data);

      std::memcpy(dst, src, size);

      if (counter->load() == 1 || counter->fetch_sub(1) == 1) {
        event.SetStateConcrete();
      }
    };

    for (int64_t i = 1; i < info.block_count; ++i) {
      params.intra_op_threadpool->getPool()->Schedule(
          [i, execute] { execute(i); });
    }
    execute(0);

    return event;
  }

  // Fallback to single-threaded memcpy.
  std::memcpy(destination_data.opaque(), source_data.opaque(), size_in_bytes);
  return OkExecuteEvent();
}

}  // namespace xla::cpu
