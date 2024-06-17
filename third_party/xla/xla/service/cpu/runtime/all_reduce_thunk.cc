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

#include "xla/service/cpu/runtime/all_reduce_thunk.h"

#include <cstddef>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/runtime/buffer_use.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/cpu/runtime/thunk.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::cpu {

absl::StatusOr<std::unique_ptr<AllReduceThunk>> AllReduceThunk::Create(
    Info info, absl::Span<const BufferAllocation::Slice> source_buffers,
    absl::Span<const Shape> source_shapes,
    BufferAllocation::Slice destination_buffer,
    const Shape& destination_shape) {
  return absl::WrapUnique(new AllReduceThunk(std::move(info), source_buffers,
                                             source_shapes, destination_buffer,
                                             destination_shape));
}

AllReduceThunk::AllReduceThunk(
    Info info, absl::Span<const BufferAllocation::Slice> source_buffers,
    absl::Span<const Shape> source_shapes,
    BufferAllocation::Slice destination_buffer, const Shape& destination_shape)
    : Thunk(Kind::kAllReduce, info),
      source_buffers_(source_buffers.begin(), source_buffers.end()),
      source_shapes_(source_shapes.begin(), source_shapes.end()),
      destination_buffer_(destination_buffer),
      destination_shape_(destination_shape) {}

tsl::AsyncValueRef<AllReduceThunk::ExecuteEvent> AllReduceThunk::Execute(
    const ExecuteParams& params) {
  tsl::profiler::TraceMe trace([&] { return TraceMeEncode(); });

  size_t num_srcs = source_buffers_.size();
  VLOG(3) << absl::StreamFormat("AllReduce: #source_buffers=%d", num_srcs);

  absl::InlinedVector<se::DeviceMemoryBase, 4> source_data(num_srcs);
  for (int i = 0; i < num_srcs; ++i) {
    TF_ASSIGN_OR_RETURN(
        source_data[i],
        params.buffer_allocations->GetDeviceAddress(source_buffers_[i]));
    VLOG(3) << absl::StreamFormat(
        "  src: %s in slice %s (%p)", source_shapes_[i].ToString(true),
        source_buffers_[i].ToString(), source_data[i].opaque());
  }

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase destination_data,
      params.buffer_allocations->GetDeviceAddress(destination_buffer_));

  VLOG(3) << absl::StreamFormat(
      "  dst: %s in slice %s (%p)", destination_shape_.ToString(true),
      destination_buffer_.ToString(), destination_data.opaque());

  // Handle single-replica case by copying the source to the destination.
  if (num_srcs == 1) {
    DCHECK_EQ(source_data.size(), destination_data.size());
    std::memcpy(destination_data.opaque(), source_data[0].opaque(),
                destination_data.size());
    return OkExecuteEvent();
  }

  return absl::UnimplementedError("AllReduceThunk::Execute not implemented");
}

Thunk::BufferUses AllReduceThunk::buffer_uses() const {
  BufferUses uses;
  uses.reserve(source_buffers_.size() + 1);
  for (auto& source_buffer : source_buffers_) {
    uses.push_back(BufferUse::Read(source_buffer));
  }
  uses.push_back(BufferUse::Write(destination_buffer_));
  return uses;
}

}  // namespace xla::cpu
