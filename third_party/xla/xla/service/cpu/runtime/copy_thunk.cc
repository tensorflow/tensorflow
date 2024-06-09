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

#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "xla/pjrt/transpose.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

CopyThunk::CopyThunk(Info info, BufferAllocation::Slice source_buffer,
                     const Shape& source_shape,
                     BufferAllocation::Slice destination_buffer,
                     const Shape& destination_shape)
    : Thunk(Kind::kCopy, std::move(info)),
      source_buffer_(source_buffer),
      source_shape_(source_shape),
      destination_buffer_(destination_buffer),
      destination_shape_(destination_shape) {
  // TODO(ezhulenev): Use factory constructor instead of CHECK.
  CHECK(ShapeUtil::Compatible(source_shape_, destination_shape_))
      << "Source shape " << source_shape_.ToString(true)
      << " must be compatble with destination shape "
      << destination_shape_.ToString(true);

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

absl::Status CopyThunk::Execute(const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase source_data,
      params.buffer_allocations->GetDeviceAddress(source_buffer_));

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase destination_data,
      params.buffer_allocations->GetDeviceAddress(destination_buffer_));

  VLOG(3) << absl::StreamFormat("Copy buffer: use_transpose=%s",
                                transpose_plan_ ? "true" : "false");
  VLOG(3) << absl::StreamFormat(
      "  src: %s in slice %s (%p)", source_shape_.ToString(true),
      source_buffer_.ToString(), source_data.opaque());
  VLOG(3) << absl::StreamFormat(
      "  dst: %s in slice %s (%p)", destination_shape_.ToString(true),
      destination_buffer_.ToString(), destination_data.opaque());

  // TODO(ezhulenev): Add benchmarks for copy thunk and add support for
  // running it on multiple threads.
  // TODO(ezhulenev): Use StreamExecutor API instead of std::memcpy? This also
  // requires a benchmark and a multi-threaded implementation.
  // TODO(ezhulenev): Add extra checks for buffer overlap.

  if (transpose_plan_) {
    transpose_plan_->Execute(source_data.opaque(), destination_data.opaque(),
                             [](std::function<void()> fn) { fn(); });
  } else {
    size_t size_in_bytes = ShapeUtil::ByteSizeOf(source_shape_);
    std::memcpy(destination_data.opaque(), source_data.opaque(), size_in_bytes);
  }

  return absl::OkStatus();
}

}  // namespace xla::cpu
