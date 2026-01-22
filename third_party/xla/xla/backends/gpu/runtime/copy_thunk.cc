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

#include "xla/backends/gpu/runtime/copy_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/node_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/copy_thunk.pb.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {

DeviceToDeviceCopyThunk::DeviceToDeviceCopyThunk(
    ThunkInfo thunk_info, const ShapedSlice& source_buffer,
    const ShapedSlice& destination_buffer, int64_t mem_size)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
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

absl::Status DeviceToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceAddressBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_.slice);
  se::DeviceAddressBase source_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_.slice);
  VLOG(3) << "Memcpy D2D of size " << size_bytes() << " from "
          << source_data.opaque() << " to " << destination_data.opaque();
  return params.stream->Memcpy(&destination_data, source_data, size_bytes());
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

//===----------------------------------------------------------------------===//
// CopyThunk
//===----------------------------------------------------------------------===//

CopyThunk::CopyThunk(ThunkInfo thunk_info, const ShapedSlice& source_buffer,
                     const ShapedSlice& destination_buffer, int64_t mem_size)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size) {
  CHECK_EQ(ShapeUtil::ByteSizeOfElements(source_buffer_.shape),
           ShapeUtil::ByteSizeOfElements(destination_buffer_.shape));

  CHECK_GE(source_buffer_.slice.size(), mem_size);
  CHECK_GE(destination_buffer_.slice.size(), mem_size);
}

absl::Status CopyThunk::ExecuteOnStream(const ExecuteParams& params) {
  return absl::OkStatus();
}

//===----------------------------------------------------------------------===//
// CopyAsyncEvents
//===----------------------------------------------------------------------===//

// Emplace() will insert {key, event} pair into the hash map,
// and return the event in order to do RecordEvent() for async memcpy.
absl::Status CopyThunk::AsyncEvents::Emplace(se::StreamExecutor* executor,
                                             const HloInstruction* instr,
                                             std::unique_ptr<se::Event> event) {
  Key key = {executor, instr};
  absl::MutexLock lock(mutex_);
  VLOG(3) << "Emplace event " << event.get();
  if (auto [it, inserted] = events_.try_emplace(key, std::move(event));
      inserted) {
    return absl::OkStatus();
  }
  return absl::InternalError("Async copy event already exists!");
}

// Retrieve a completion event started by copy-start instruction
// `instr`, and remove the event from the collection.
absl::StatusOr<std::unique_ptr<se::Event>> CopyThunk::AsyncEvents::Extract(
    se::StreamExecutor* executor, const HloInstruction* instr) {
  Key key = {executor, instr};
  absl::MutexLock lock(mutex_);
  if (auto event = events_.extract(key)) {
    VLOG(3) << "Extract event " << event.mapped().get();
    return std::move(event.mapped());
  }
  return absl::InternalError("Async copy event was not found!");
}

absl::StatusOr<ThunkProto> CopyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  CopyThunkProto* copy_thunk_proto = proto.mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source_buffer_.ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination_buffer_.ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
}

absl::StatusOr<std::unique_ptr<CopyThunk>> CopyThunk::FromProto(
    ThunkInfo thunk_info, const CopyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.source_buffer(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(ShapedSlice dst_slice,
                      ShapedSlice::FromProto(thunk_proto.destination_buffer(),
                                             buffer_allocations));
  if (ShapeUtil::ByteSizeOfElements(src_slice.shape) !=
      ShapeUtil::ByteSizeOfElements(dst_slice.shape)) {
    return absl::FailedPreconditionError(
        "DeviceToDeviceCopyThunkProto with incompatible shapes.");
  }

  return std::make_unique<CopyThunk>(std::move(thunk_info), src_slice,
                                     dst_slice, thunk_proto.mem_size());
}

//===----------------------------------------------------------------------===//
// DeviceToHostCopyThunk
//===----------------------------------------------------------------------===//
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

//===----------------------------------------------------------------------===//
// HostToDeviceCopyThunk
//===----------------------------------------------------------------------===//
HostToDeviceCopyThunk::HostToDeviceCopyThunk(
    ThunkInfo thunk_info, const ShapedSlice& source_buffer,
    const ShapedSlice& destination_buffer, int64_t mem_size,
    std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    const HloInstruction* instr)
    : CopyThunk(std::move(thunk_info), source_buffer, destination_buffer,
                mem_size),
      async_events_(std::move(async_events)),
      instr_(instr) {}

absl::Status HostToDeviceCopyThunk::ExecuteOnStream(
    const ExecuteParams& params) {
  se::DeviceAddressBase destination_data =
      params.buffer_allocations->GetDeviceAddress(destination().slice);
  se::DeviceAddressBase source_data =
      params.buffer_allocations->GetDeviceAddress(source().slice);
  void* cpu_src = source_data.opaque();
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  TF_RETURN_IF_ERROR(stream->Memcpy(&destination_data, cpu_src, size_bytes()));
  if (stream == params.stream) {
    VLOG(2) << "Memcpy H2D from the main stream";
    return absl::OkStatus();
  }
  VLOG(2) << "Memcpy H2D from the other stream";
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto event, executor->CreateEvent());
  // Record memcpy operation completion.
  TF_RETURN_IF_ERROR(stream->RecordEvent(event.get()));
  VLOG(3) << "Emplace events: " << event.get()
          << " for instr: " << instr_->ToString();
  return async_events_->Emplace(executor, instr_, std::move(event));
}

absl::StatusOr<ThunkProto> HostToDeviceCopyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  HostToDeviceCopyThunkProto* h2d_copy_thunk_proto =
      proto.mutable_host_to_device_copy_thunk();
  CopyThunkProto* copy_thunk_proto = h2d_copy_thunk_proto->mutable_copy_thunk();
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_source_buffer(),
                      source().ToProto());
  TF_ASSIGN_OR_RETURN(*copy_thunk_proto->mutable_destination_buffer(),
                      destination().ToProto());
  copy_thunk_proto->set_mem_size(size_bytes());
  return proto;
}

absl::StatusOr<std::unique_ptr<HostToDeviceCopyThunk>>
HostToDeviceCopyThunk::FromProto(
    ThunkInfo thunk_info, const HostToDeviceCopyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.copy_thunk().source_buffer(),
                             buffer_allocations));
  TF_ASSIGN_OR_RETURN(
      ShapedSlice dst_slice,
      ShapedSlice::FromProto(thunk_proto.copy_thunk().destination_buffer(),
                             buffer_allocations));
  return std::make_unique<HostToDeviceCopyThunk>(
      std::move(thunk_info), src_slice, dst_slice,
      thunk_proto.copy_thunk().mem_size(),
      /*events=*/nullptr,
      /*instr=*/nullptr);
}

std::optional<AsyncEventsUniqueId>
HostToDeviceCopyThunk::GetAsyncEventsUniqueId() const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

//===----------------------------------------------------------------------===//
// CopyDoneThunk
//===----------------------------------------------------------------------===//

CopyDoneThunk::CopyDoneThunk(
    Thunk::Kind kind, ThunkInfo thunk_info,
    std::shared_ptr<CopyThunk::AsyncEvents> async_events,
    const HloInstruction* copy_start_instr)
    : Thunk(kind, std::move(thunk_info)),
      async_events_(std::move(async_events)),
      copy_start_instr_(copy_start_instr) {}

absl::Status CopyDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "CopyDone thunk between a host and a device for: "
          << copy_start_instr_->ToString();
  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<se::Event> event,
                      async_events_->Extract(executor, copy_start_instr_));
  return params.stream->WaitFor(event.get());
}

std::optional<AsyncEventsUniqueId> CopyDoneThunk::GetAsyncEventsUniqueId()
    const {
  if (!async_events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to async_events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(async_events_.get());
}

//===----------------------------------------------------------------------===//
// DynamicMemcpyThunk
//===----------------------------------------------------------------------===//

DynamicMemcpyThunk::DynamicMemcpyThunk(ThunkInfo thunk_info,
                                       const ShapedSlice& source_buffer,
                                       const ShapedSlice& destination_buffer,
                                       uint64_t mem_size,
                                       DynamicMemcpyThunk::Offsets offsets)
    : Thunk(Kind::kCopy, std::move(thunk_info)),
      source_buffer_(source_buffer),
      destination_buffer_(destination_buffer),
      mem_size_(mem_size),
      offsets_(std::move(offsets)) {}

absl::Status DynamicMemcpyThunk::ExecuteOnStream(const ExecuteParams& params) {
  se::DeviceAddressBase src_data =
      params.buffer_allocations->GetDeviceAddress(source_buffer_.slice);
  se::DeviceAddressBase dst_data =
      params.buffer_allocations->GetDeviceAddress(destination_buffer_.slice);

  int64_t iteration_index = 0;
  if (offsets_.depends_on_loop) {
    TF_ASSIGN_OR_RETURN(iteration_index, WhileThunk::CurrentLoopIteration());
  }

  int64_t src_offset = offsets_.src_offsets[iteration_index];
  int64_t dst_offset = offsets_.dst_offsets[iteration_index];

  auto src_with_offset = src_data.GetByteSlice(src_offset, mem_size_);
  auto dst_with_offset = dst_data.GetByteSlice(dst_offset, mem_size_);
  VLOG(3) << "Memcpy of size " << mem_size_ << " from "
          << src_with_offset.opaque() << " (offset " << src_offset << ") to "
          << dst_with_offset.opaque() << " (offset " << dst_offset << ")";
  TF_ASSIGN_OR_RETURN(
      se::Stream * stream,
      GetStreamForExecution(Thunk::execution_stream_id(), params));
  return stream->Memcpy(&dst_with_offset, src_with_offset, mem_size_);
}

DynamicMemcpyThunkProto::Offsets DynamicMemcpyThunk::Offsets::ToProto() const {
  DynamicMemcpyThunkProto::Offsets proto;
  proto.set_depends_on_loop(depends_on_loop);
  proto.mutable_src_offsets()->Add(src_offsets.begin(), src_offsets.end());
  proto.mutable_dst_offsets()->Add(dst_offsets.begin(), dst_offsets.end());
  return proto;
}

absl::StatusOr<DynamicMemcpyThunk::Offsets>
DynamicMemcpyThunk::Offsets::FromProto(
    const DynamicMemcpyThunkProto::Offsets& proto) {
  Offsets offsets;
  offsets.depends_on_loop = proto.depends_on_loop();
  offsets.src_offsets = {proto.src_offsets().begin(),
                         proto.src_offsets().end()};
  offsets.dst_offsets = {proto.dst_offsets().begin(),
                         proto.dst_offsets().end()};
  return offsets;
}

absl::StatusOr<ThunkProto> DynamicMemcpyThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  DynamicMemcpyThunkProto* dynamic_memcpy_thunk_proto =
      proto.mutable_dynamic_memcpy_thunk();
  TF_ASSIGN_OR_RETURN(*dynamic_memcpy_thunk_proto->mutable_source_buffer(),
                      source_buffer_.ToProto());
  TF_ASSIGN_OR_RETURN(*dynamic_memcpy_thunk_proto->mutable_destination_buffer(),
                      destination_buffer_.ToProto());
  dynamic_memcpy_thunk_proto->set_mem_size(mem_size_);
  *dynamic_memcpy_thunk_proto->mutable_offsets() = offsets_.ToProto();
  return proto;
}

absl::StatusOr<std::unique_ptr<DynamicMemcpyThunk>>
DynamicMemcpyThunk::FromProto(
    ThunkInfo thunk_info, const DynamicMemcpyThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  TF_ASSIGN_OR_RETURN(
      ShapedSlice src_slice,
      ShapedSlice::FromProto(thunk_proto.source_buffer(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(ShapedSlice dst_slice,
                      ShapedSlice::FromProto(thunk_proto.destination_buffer(),
                                             buffer_allocations));
  TF_ASSIGN_OR_RETURN(Offsets offsets,
                      Offsets::FromProto(thunk_proto.offsets()));
  return std::make_unique<DynamicMemcpyThunk>(std::move(thunk_info), src_slice,
                                              dst_slice, thunk_proto.mem_size(),
                                              std::move(offsets));
}

}  // namespace gpu
}  // namespace xla
