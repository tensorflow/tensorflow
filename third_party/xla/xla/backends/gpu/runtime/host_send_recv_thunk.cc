/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/host_send_recv_thunk.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/runtime/device_id.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla::gpu {

using tsl::AsyncValueRef;
using tsl::profiler::TraceMe;
using tsl::profiler::TraceMeEncode;

// For sharded buffers we should execute Send/Recv operations only on devices
// with maximal sharding, and do nothing on every other device.
static absl::StatusOr<bool> ShouldSkip(
    absl::string_view operation, const Thunk::ExecuteParams& params,
    const std::optional<GlobalDeviceId>& device_constraint) {
  if (!device_constraint.has_value()) {
    return false;
  }

  GlobalDeviceId global_device_id = params.collective_params->global_device_id;
  bool skip = global_device_id != *device_constraint;
  if (skip) {
    VLOG(3) << "Skip " << operation << " as device id " << global_device_id
            << " doesn't match device id constraint " << *device_constraint;
  }

  return skip;
}

//===----------------------------------------------------------------------===//
// HostSendRecvAsyncEvents
//===----------------------------------------------------------------------===//

absl::Status HostSendRecvAsyncEvents::Emplace(
    se::StreamExecutor* executor, int32_t channel_id,
    tsl::AsyncValueRef<std::unique_ptr<se::Event>> event) {
  Key key = {executor, channel_id};

  absl::MutexLock lock(mutex_);
  if (auto it = events_.try_emplace(key, std::move(event)); it.second) {
    return absl::OkStatus();
  }

  return absl::InternalError(absl::StrFormat(
      "Async send/recv event already exists (channel_id=%d)", channel_id));
}

absl::StatusOr<AsyncValueRef<std::unique_ptr<se::Event>>>
HostSendRecvAsyncEvents::Extract(se::StreamExecutor* executor,
                                 int32_t channel_id) {
  Key key = {executor, channel_id};

  absl::MutexLock lock(mutex_);
  if (auto event = events_.extract(key)) {
    return std::move(event.mapped());
  }

  return absl::InternalError(absl::StrFormat(
      "Async send/recv event was not found (channel_id==%d)", channel_id));
}

//===----------------------------------------------------------------------===//
// HostSendThunk
//===----------------------------------------------------------------------===//

HostSendThunk::HostSendThunk(
    ThunkInfo thunk_info, Shape shape, BufferAllocation::Slice buffer,
    int64_t channel_id, std::shared_ptr<HostSendRecvAsyncEvents> events,
    absl::flat_hash_map<std::string, std::string> frontend_attrs,
    std::optional<GlobalDeviceId> device_constraint)
    : Thunk(Thunk::kHostSend, thunk_info),
      shape_(shape),
      buffer_(buffer),
      channel_id_(channel_id),
      events_(std::move(events)),
      frontend_attrs_(std::move(frontend_attrs)),
      device_constraint_(device_constraint) {}

absl::StatusOr<ThunkProto> HostSendThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  HostSendThunkProto& host_send_thunk_proto = *proto.mutable_host_send_thunk();
  *host_send_thunk_proto.mutable_shape() = shape_.ToProto();
  TF_ASSIGN_OR_RETURN(*host_send_thunk_proto.mutable_buffer(),
                      buffer_.ToProto());
  host_send_thunk_proto.set_channel_id(channel_id_);
  for (const auto& [key, value] : frontend_attrs_) {
    host_send_thunk_proto.mutable_frontend_attrs()->insert({key, value});
  }
  if (device_constraint_.has_value()) {
    host_send_thunk_proto.set_device_constraint(device_constraint_->value());
  }
  std::optional<AsyncEventsUniqueId> async_events_unique_id =
      GetAsyncEventsUniqueId();
  if (!async_events_unique_id.has_value()) {
    return absl::InternalError("HostSendThunk has no paired Done event");
  }
  host_send_thunk_proto.set_async_events_unique_id(
      async_events_unique_id.value().value());
  return proto;
}

absl::StatusOr<std::unique_ptr<HostSendThunk>> HostSendThunk::FromProto(
    ThunkInfo thunk_info, const HostSendThunkProto& proto,
    absl::Span<const BufferAllocation> allocations,
    HostSendRecvAsyncEventsMap& async_events_map) {
  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.shape()));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice buffer,
      BufferAllocation::Slice::FromProto(proto.buffer(), allocations));
  std::optional<GlobalDeviceId> device_constraint;
  if (proto.has_device_constraint()) {
    device_constraint = GlobalDeviceId(proto.device_constraint());
  }
  absl::flat_hash_map<std::string, std::string> frontend_attrs(
      proto.frontend_attrs().begin(), proto.frontend_attrs().end());

  auto [async_event_it, _] = async_events_map.try_emplace(
      AsyncEventsUniqueId(proto.async_events_unique_id()),
      std::make_shared<HostSendRecvAsyncEvents>());
  return std::make_unique<HostSendThunk>(
      thunk_info, std::move(shape), buffer, proto.channel_id(),
      async_event_it->second, std::move(frontend_attrs), device_constraint);
}

absl::Status HostSendThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Send buffer: channel_id=" << channel_id_
          << "; shape=" << shape_.ToString();

  TF_ASSIGN_OR_RETURN(bool skip,
                      ShouldSkip("sending buffer", params, device_constraint_));
  if (skip) {
    return absl::OkStatus();
  }

  TraceMe trace(
      [&] { return TraceMeEncode("Send", {{"channel_id", channel_id_}}); });

  // Use device_to_host stream if it is available.
  se::Stream* stream = params.device_to_host_stream;
  if (stream) {
    TF_RETURN_IF_ERROR(stream->WaitFor(params.stream));
  } else {
    stream = params.stream;
  }

  se::DeviceAddressBase src =
      params.buffer_allocations->GetDeviceAddress(buffer_);

  // Send buffer to a handler registered with the executable.
  if (auto* send = params.send_device_memory_function) {
    TF_ASSIGN_OR_RETURN(
        AsyncValueRef<std::unique_ptr<se::Event>> done,
        (*send)(channel_id_, stream, shape_, src, frontend_attrs_));
    return events_->Emplace(stream->parent(), channel_id_, std::move(done));
  }

  return absl::InvalidArgumentError(
      "SendDeviceMemoryFunction is not available");
}

std::optional<AsyncEventsUniqueId> HostSendThunk::GetAsyncEventsUniqueId()
    const {
  if (!events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(events_.get());
}

//===----------------------------------------------------------------------===//
// HostSendDoneThunk
//===----------------------------------------------------------------------===//

HostSendDoneThunk::HostSendDoneThunk(
    ThunkInfo thunk_info, int64_t channel_id,
    std::shared_ptr<HostSendRecvAsyncEvents> events,
    std::optional<GlobalDeviceId> device_constraint)
    : Thunk(Thunk::kHostSendDone, thunk_info),
      channel_id_(channel_id),
      events_(std::move(events)),
      device_constraint_(device_constraint) {}

absl::StatusOr<ThunkProto> HostSendDoneThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  HostSendDoneThunkProto& host_send_done_thunk_proto =
      *proto.mutable_host_send_done_thunk();
  host_send_done_thunk_proto.set_channel_id(channel_id_);
  if (device_constraint_.has_value()) {
    host_send_done_thunk_proto.set_device_constraint(
        device_constraint_->value());
  }
  std::optional<AsyncEventsUniqueId> async_events_unique_id =
      GetAsyncEventsUniqueId();
  if (!async_events_unique_id.has_value()) {
    return absl::InternalError("HostSendDoneThunk has no paired Start event");
  }
  host_send_done_thunk_proto.set_async_events_unique_id(
      async_events_unique_id.value().value());
  return proto;
}

absl::StatusOr<std::unique_ptr<HostSendDoneThunk>> HostSendDoneThunk::FromProto(
    ThunkInfo thunk_info, const HostSendDoneThunkProto& proto,
    absl::Span<const BufferAllocation> allocations,
    HostSendRecvAsyncEventsMap& async_events_map) {
  std::optional<GlobalDeviceId> device_constraint;
  if (proto.has_device_constraint()) {
    device_constraint = GlobalDeviceId(proto.device_constraint());
  }

  auto [async_event_it, _] = async_events_map.try_emplace(
      AsyncEventsUniqueId(proto.async_events_unique_id()),
      std::make_shared<HostSendRecvAsyncEvents>());

  return std::make_unique<HostSendDoneThunk>(thunk_info, proto.channel_id(),
                                             std::move(async_event_it->second),
                                             device_constraint);
}

absl::Status HostSendDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Wait for send completion: channel_id=" << channel_id_;

  TF_ASSIGN_OR_RETURN(bool skip, ShouldSkip("waiting for send completion",
                                            params, device_constraint_));
  if (skip) {
    return absl::OkStatus();
  }

  TraceMe trace(
      [&] { return TraceMeEncode("SendDone", {{"channel_id", channel_id_}}); });

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto done_event, events_->Extract(executor, channel_id_));

  // Wait until send handler will record an event on the stream.
  BlockUntilReady(done_event.GetAsyncValue());
  if (done_event.IsError()) {
    return done_event.GetError();
  }

  VLOG(5) << "Completed Send operation: channel_id=" << channel_id_;

  // Once event is recorded we can add a stream dependency.
  return params.stream->WaitFor(done_event.get().get());
}

std::optional<AsyncEventsUniqueId> HostSendDoneThunk::GetAsyncEventsUniqueId()
    const {
  if (!events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(events_.get());
}

//===----------------------------------------------------------------------===//
// HostRecvThunk
//===----------------------------------------------------------------------===//

HostRecvThunk::HostRecvThunk(
    ThunkInfo thunk_info, Shape shape, BufferAllocation::Slice buffer,
    int64_t channel_id, std::shared_ptr<HostSendRecvAsyncEvents> events,
    absl::flat_hash_map<std::string, std::string> frontend_attrs,
    std::optional<GlobalDeviceId> device_constraint)
    : Thunk(Thunk::kHostRecv, thunk_info),
      shape_(shape),
      buffer_(buffer),
      channel_id_(channel_id),
      events_(std::move(events)),
      frontend_attrs_(std::move(frontend_attrs)),
      device_constraint_(device_constraint) {}

absl::StatusOr<ThunkProto> HostRecvThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  HostRecvThunkProto& host_recv_thunk_proto = *proto.mutable_host_recv_thunk();
  *host_recv_thunk_proto.mutable_shape() = shape_.ToProto();
  TF_ASSIGN_OR_RETURN(*host_recv_thunk_proto.mutable_buffer(),
                      buffer_.ToProto());
  host_recv_thunk_proto.set_channel_id(channel_id_);
  for (const auto& [key, value] : frontend_attrs_) {
    host_recv_thunk_proto.mutable_frontend_attrs()->insert({key, value});
  }
  if (device_constraint_.has_value()) {
    host_recv_thunk_proto.set_device_constraint(device_constraint_->value());
  }
  std::optional<AsyncEventsUniqueId> async_events_unique_id =
      GetAsyncEventsUniqueId();
  if (!async_events_unique_id.has_value()) {
    return absl::InternalError("HostRecvThunk has no paired Done event");
  }
  host_recv_thunk_proto.set_async_events_unique_id(
      async_events_unique_id.value().value());
  return proto;
}

absl::StatusOr<std::unique_ptr<HostRecvThunk>> HostRecvThunk::FromProto(
    ThunkInfo thunk_info, const HostRecvThunkProto& proto,
    absl::Span<const BufferAllocation> allocations,
    HostSendRecvAsyncEventsMap& async_events_map) {
  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.shape()));
  TF_ASSIGN_OR_RETURN(
      BufferAllocation::Slice buffer,
      BufferAllocation::Slice::FromProto(proto.buffer(), allocations));
  std::optional<GlobalDeviceId> device_constraint;
  if (proto.has_device_constraint()) {
    device_constraint = GlobalDeviceId(proto.device_constraint());
  }
  absl::flat_hash_map<std::string, std::string> frontend_attrs(
      proto.frontend_attrs().begin(), proto.frontend_attrs().end());

  auto [async_event_it, _] = async_events_map.try_emplace(
      AsyncEventsUniqueId(proto.async_events_unique_id()),
      std::make_shared<HostSendRecvAsyncEvents>());
  return std::make_unique<HostRecvThunk>(
      thunk_info, std::move(shape), buffer, proto.channel_id(),
      async_event_it->second, std::move(frontend_attrs), device_constraint);
}

absl::Status HostRecvThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Recv buffer: channel_id=" << channel_id_
          << "; shape=" << shape_.ToString();

  TF_ASSIGN_OR_RETURN(
      bool skip, ShouldSkip("receiving buffer", params, device_constraint_));
  if (skip) {
    return absl::OkStatus();
  }

  TraceMe trace(
      [&] { return TraceMeEncode("Recv", {{"channel_id", channel_id_}}); });

  // Use host_to_device stream if it is available.
  se::Stream* stream = params.host_to_device_stream;
  if (stream) {
    TF_RETURN_IF_ERROR(stream->WaitFor(params.stream));
  } else {
    stream = params.stream;
  }

  se::DeviceAddressBase dst =
      params.buffer_allocations->GetDeviceAddress(buffer_);

  // Recv buffer from a handler registered with the run options.
  if (auto* recv = params.recv_device_memory_function) {
    TF_ASSIGN_OR_RETURN(
        AsyncValueRef<std::unique_ptr<se::Event>> done,
        (*recv)(channel_id_, stream, shape_, &dst, frontend_attrs_));
    return events_->Emplace(stream->parent(), channel_id_, std::move(done));
  }

  return absl::InvalidArgumentError(
      "RecvDeviceMemoryFunction is not available");
}

std::optional<AsyncEventsUniqueId> HostRecvThunk::GetAsyncEventsUniqueId()
    const {
  if (!events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(events_.get());
}

//===----------------------------------------------------------------------===//
// HostRecvDoneThunk
//===----------------------------------------------------------------------===//

HostRecvDoneThunk::HostRecvDoneThunk(
    ThunkInfo thunk_info, int64_t channel_id,
    std::shared_ptr<HostSendRecvAsyncEvents> events,
    std::optional<GlobalDeviceId> device_constraint)
    : Thunk(Thunk::kHostRecvDone, thunk_info),
      channel_id_(channel_id),
      events_(std::move(events)),
      device_constraint_(device_constraint) {}

absl::StatusOr<ThunkProto> HostRecvDoneThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();
  HostRecvDoneThunkProto& host_recv_done_thunk_proto =
      *proto.mutable_host_recv_done_thunk();
  host_recv_done_thunk_proto.set_channel_id(channel_id_);
  if (device_constraint_.has_value()) {
    host_recv_done_thunk_proto.set_device_constraint(
        device_constraint_->value());
  }
  std::optional<AsyncEventsUniqueId> async_events_unique_id =
      GetAsyncEventsUniqueId();
  if (!async_events_unique_id.has_value()) {
    return absl::InternalError("HostRecvDoneThunk has no paired Start event");
  }
  host_recv_done_thunk_proto.set_async_events_unique_id(
      async_events_unique_id.value().value());
  return proto;
}

absl::StatusOr<std::unique_ptr<HostRecvDoneThunk>> HostRecvDoneThunk::FromProto(
    ThunkInfo thunk_info, const HostRecvDoneThunkProto& proto,
    absl::Span<const BufferAllocation> allocations,
    HostSendRecvAsyncEventsMap& async_events_map) {
  std::optional<GlobalDeviceId> device_constraint;
  if (proto.has_device_constraint()) {
    device_constraint = GlobalDeviceId(proto.device_constraint());
  }

  auto [async_event_it, _] = async_events_map.try_emplace(
      AsyncEventsUniqueId(proto.async_events_unique_id()),
      std::make_shared<HostSendRecvAsyncEvents>());

  return std::make_unique<HostRecvDoneThunk>(thunk_info, proto.channel_id(),
                                             std::move(async_event_it->second),
                                             device_constraint);
}

absl::Status HostRecvDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Wait for recv completion: channel_id=" << channel_id_;

  TF_ASSIGN_OR_RETURN(bool skip, ShouldSkip("waiting for recv completion",
                                            params, device_constraint_));
  if (skip) {
    return absl::OkStatus();
  }

  TraceMe trace(
      [&] { return TraceMeEncode("RecvDone", {{"channel_id", channel_id_}}); });

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto done_event, events_->Extract(executor, channel_id_));

  // Wait until send handler will record an event on the stream.
  BlockUntilReady(done_event.GetAsyncValue());
  if (done_event.IsError()) {
    return done_event.GetError();
  }

  VLOG(5) << "Completed Recv operation: channel=" << channel_id_;

  // Once event is recorded we can add a stream dependency.
  return params.stream->WaitFor(done_event.get().get());
}

std::optional<AsyncEventsUniqueId> HostRecvDoneThunk::GetAsyncEventsUniqueId()
    const {
  if (!events_) {
    return std::nullopt;
  }
  // We rely on the fact that the pointer to events_ is unique.
  return absl::bit_cast<AsyncEventsUniqueId>(events_.get());
}

}  // namespace xla::gpu
