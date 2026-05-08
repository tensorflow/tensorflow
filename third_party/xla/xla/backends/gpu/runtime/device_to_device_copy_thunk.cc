/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/buffer_allocations.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

DeviceToDeviceCopyThunk::DeviceToDeviceCopyThunk(
    ThunkInfo thunk_info, const ShapedSlice& source_buffer,
    const ShapedSlice& destination_buffer, int64_t mem_size)
    : Command(CommandType::kMemcpyDeviceToDeviceCmd, Kind::kCopy,
              std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {
  // TODO(b/460846009): Determine size based on shape.
  // Bounded dynamic shape contains extra header after data.
  // Header size needs to be accounted for.
  CHECK_EQ(ShapeUtil::ByteSizeOf(source_buffer_.shape),
           ShapeUtil::ByteSizeOf(destination_buffer_.shape));

  CHECK_GE(source_buffer_.slice.size(), mem_size);
  CHECK_GE(destination_buffer_.slice.size(), mem_size);
}

DeviceToDeviceCopyThunk::CopyAddresses
DeviceToDeviceCopyThunk::GetCopyAddresses(
    const BufferAllocations& allocations) const {
  return {allocations.GetDeviceAddress(destination_buffer_.slice),
          allocations.GetDeviceAddress(source_buffer_.slice)};
}

absl::Status DeviceToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  auto [dst, src] = GetCopyAddresses(*params.buffer_allocations);
  VLOG(3) << "Memcpy D2D of size " << size_bytes() << " from " << src.opaque()
          << " to " << dst.opaque();
  return params.stream->Memcpy(&dst, src, size_bytes());
}

absl::StatusOr<const se::CommandBuffer::Command*>
DeviceToDeviceCopyThunk::Record(const Thunk::ExecuteParams& execute_params,
                                const RecordParams& record_params,
                                RecordAction record_action,
                                se::CommandBuffer* command_buffer) {
  auto [dst, src] = GetCopyAddresses(*execute_params.buffer_allocations);

  VLOG(5) << "DeviceToDeviceCopyThunk::Record: num_bytes=" << mem_size_;
  VLOG(5) << "  Dst: " << destination_buffer_ << " (" << dst.opaque() << ")";
  VLOG(5) << "  Src: " << source_buffer_ << " (" << src.opaque() << ")";

  if (mem_size_ == 0) {
    VLOG(5) << "Skip recording DeviceToDeviceCopyThunk command of 0 bytes";
    return nullptr;
  }

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateMemcpyD2D(&dst, src, mem_size_,
                                           create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(
        command_buffer->UpdateMemcpyD2D(update->command, &dst, src, mem_size_));
    return update->command;
  }
  return Internal("Invalid record action");
}

absl::StatusOr<ThunkProto> DeviceToDeviceCopyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  DeviceToDeviceCopyThunkProto* d2d_copy_thunk_proto =
      proto.mutable_device_to_device_copy_thunk();
  CopyThunkProto* copy_thunk_proto = d2d_copy_thunk_proto->mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source_buffer_.ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination_buffer_.ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
}

absl::StatusOr<std::unique_ptr<DeviceToDeviceCopyThunk>>
DeviceToDeviceCopyThunk::FromProto(
    ThunkInfo thunk_info, const DeviceToDeviceCopyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.copy_thunk().source_buffer(),
                             buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      ShapedSlice dst_slice,
      ShapedSlice::FromProto(thunk_proto.copy_thunk().destination_buffer(),
                             buffer_allocations));
  if (ShapeUtil::ByteSizeOfElements(src_slice.shape) !=
      ShapeUtil::ByteSizeOfElements(dst_slice.shape)) {
    return absl::FailedPreconditionError(
        "DeviceToDeviceCopyThunkProto with incompatible shapes.");
  }
  return std::make_unique<DeviceToDeviceCopyThunk>(
      std::move(thunk_info), src_slice, dst_slice,
      thunk_proto.copy_thunk().mem_size());
}

}  // namespace gpu
}  // namespace xla
