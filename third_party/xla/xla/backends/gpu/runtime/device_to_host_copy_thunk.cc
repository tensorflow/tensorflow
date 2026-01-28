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

#include "xla/backends/gpu/runtime/device_to_host_copy_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/casts.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

DeviceToHostCopyThunk::DeviceToHostCopyThunk(
    ThunkInfo thunk_info, const ShapedSlice& source_buffer,
    const ShapedSlice& destination_buffer, int64_t mem_size,
    std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    const HloInstruction* instr)
    : CopyThunk(std::move(thunk_info), source_buffer, destination_buffer,
                mem_size),
      async_events_(std::move(async_events)),
      instr_(instr) {}

absl::Status DeviceToHostCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceAddressBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination().slice);
  se::DeviceAddressBase source_data =
      params.buffer_allocations->GetDeviceAddress(source().slice);
  void* cpu_dst = destination_data.opaque();
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  TF_RETURN_IF_ERROR(stream->Memcpy(cpu_dst, source_data, size_bytes()));
  if (stream == params.stream) {
    VLOG(2) << "Memcpy D2H from the main stream";
    return absl::OkStatus();
  }
  VLOG(2) << "Memcpy D2H from the other stream";
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto event, executor->CreateEvent());
  // Record memcpy operation completion.
  TF_RETURN_IF_ERROR(stream->RecordEvent(event.get()));
  VLOG(3) << "Emplace events: " << event.get()
          << " for instr: " << instr_->ToString();
  return async_events_->Emplace(executor, instr_, std::move(event));
}

absl::StatusOr<ThunkProto> DeviceToHostCopyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  DeviceToHostCopyThunkProto* d2h_copy_thunk_proto =
      proto.mutable_device_to_host_copy_thunk();
  CopyThunkProto* copy_thunk_proto = d2h_copy_thunk_proto->mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source().ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination().ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
}

absl::StatusOr<std::unique_ptr<DeviceToHostCopyThunk>>
DeviceToHostCopyThunk::FromProto(
    ThunkInfo thunk_info, const DeviceToHostCopyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.copy_thunk().source_buffer(),
                             buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      ShapedSlice dst_slice,
      ShapedSlice::FromProto(thunk_proto.copy_thunk().destination_buffer(),
                             buffer_allocations));
  return std::make_unique<DeviceToHostCopyThunk>(
      std::move(thunk_info), src_slice, dst_slice,
      thunk_proto.copy_thunk().mem_size(),
      /*events=*/nullptr,
      /*instr=*/nullptr);
}

std::optional<AsyncEventsUniqueId>
DeviceToHostCopyThunk::GetAsyncEventsUniqueId() const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

}  // namespace gpu
}  // namespace xla
