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

#include "xla/backends/gpu/runtime/memset_thunk.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <variant>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/runtime/command.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/command_buffer.h"
#include "xla/stream_executor/device_address.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

absl::Status MemzeroThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::DeviceAddressBase dest_data =
      params.buffer_allocations->GetDeviceAddress(dest_.slice);
  return params.stream->MemZero(&dest_data, dest_data.size());
}

absl::StatusOr<const se::CommandBuffer::Command*> MemzeroThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dest_data =
      execute_params.buffer_allocations->GetDeviceAddress(dest_.slice);

  VLOG(5) << "MemzeroThunk::Record";
  VLOG(5) << "  dest: " << dest_ << " (" << dest_data.opaque() << ")";

  if (dest_.slice.size() == 0) {
    VLOG(5) << "Skip recording MemzeroThunk command of 0 bytes";
    return nullptr;
  }

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateMemset(&dest_data, uint8_t{0},
                                        /*num_elements=*/dest_.slice.size(),
                                        create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(
        command_buffer->UpdateMemset(update->command, &dest_data, uint8_t{0},
                                     /*num_elements=*/dest_.slice.size()));
    return update->command;
  }
  return Internal("Invalid record action");
}

absl::StatusOr<std::unique_ptr<MemzeroThunk>> MemzeroThunk::FromProto(
    ThunkInfo thunk_info, const MemzeroThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice dest,
      ShapedSlice::FromProto(thunk_proto.dest_buffer(), buffer_allocations));
  return std::make_unique<MemzeroThunk>(std::move(thunk_info), dest);
}

absl::StatusOr<ThunkProto> MemzeroThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  MemzeroThunkProto* memzero_thunk_proto = proto.mutable_memzero_thunk();
  TF_ASSIGN_OR_RETURN(*memzero_thunk_proto->mutable_dest_buffer(),
                      dest_.ToProto());
  return proto;
}

absl::Status Memset32BitValueThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceAddressBase dest_data =
      params.buffer_allocations->GetDeviceAddress(dest_);
  return params.stream->Memset32(&dest_data, value_, dest_data.size());
}

absl::StatusOr<const se::CommandBuffer::Command*> Memset32BitValueThunk::Record(
    const Thunk::ExecuteParams& execute_params,
    const RecordParams& record_params, RecordAction record_action,
    se::CommandBuffer* command_buffer) {
  se::DeviceAddressBase dest_data =
      execute_params.buffer_allocations->GetDeviceAddress(dest_);

  VLOG(5) << "Memset32BitValueThunk::Record: value=" << value_;
  VLOG(5) << "  dest: " << dest_ << " (" << dest_data.opaque() << ")";

  if (dest_.size() == 0) {
    VLOG(5) << "Skip recording Memset32BitValueThunk command of 0 bytes";
    return nullptr;
  }

  if (auto* create = std::get_if<RecordCreate>(&record_action)) {
    return command_buffer->CreateMemset(
        &dest_data, value_,
        /*num_elements=*/dest_.size() / sizeof(uint32_t), create->dependencies);
  }
  if (auto* update = std::get_if<RecordUpdate>(&record_action)) {
    RETURN_IF_ERROR(command_buffer->UpdateMemset(
        update->command, &dest_data, value_,
        /*num_elements=*/dest_.size() / sizeof(uint32_t)));
    return update->command;
  }
  return Internal("Invalid record action");
}

absl::StatusOr<std::unique_ptr<Memset32BitValueThunk>>
Memset32BitValueThunk::FromProto(
    ThunkInfo thunk_info, const Memset32BitValueThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dest,
                      BufferAllocation::Slice::FromProto(
                          thunk_proto.dest_buffer(), buffer_allocations));
  return std::make_unique<Memset32BitValueThunk>(std::move(thunk_info),
                                                 thunk_proto.value(), dest);
}

absl::StatusOr<ThunkProto> Memset32BitValueThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  Memset32BitValueThunkProto* memset_thunk_proto =
      proto.mutable_memset32bit_value_thunk();
  TF_ASSIGN_OR_RETURN(*memset_thunk_proto->mutable_dest_buffer(),
                      dest_.ToProto());
  memset_thunk_proto->set_value(value_);
  return proto;
}

}  // namespace gpu
}  // namespace xla
