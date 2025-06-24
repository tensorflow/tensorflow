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

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/global_device_id.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory.h"
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
  if (!device_constraint.has_value()) return false;

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

  absl::MutexLock lock(&mutex_);
  if (auto it = events_.try_emplace(key, std::move(event)); it.second)
    return absl::OkStatus();

  return absl::InternalError(absl::StrFormat(
      "Async send/recv event already exists (channel_id=%d)", channel_id));
}

absl::StatusOr<AsyncValueRef<std::unique_ptr<se::Event>>>
HostSendRecvAsyncEvents::Extract(se::StreamExecutor* executor,
                                 int32_t channel_id) {
  Key key = {executor, channel_id};

  absl::MutexLock lock(&mutex_);
  if (auto event = events_.extract(key)) return std::move(event.mapped());

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

absl::Status HostSendThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Send buffer: channel_id=" << channel_id_
          << "; shape=" << shape_.ToString();

  TF_ASSIGN_OR_RETURN(bool skip,
                      ShouldSkip("sending buffer", params, device_constraint_));
  if (skip) return absl::OkStatus();

  TraceMe trace(
      [&] { return TraceMeEncode("Send", {{"channel_id", channel_id_}}); });

  // Use device_to_host stream if it is available.
  se::Stream* stream = params.device_to_host_stream;
  if (stream) {
    TF_RETURN_IF_ERROR(stream->WaitFor(params.stream));
  } else {
    stream = params.stream;
  }

  se::DeviceMemoryBase src =
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

absl::Status HostSendDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Wait for send completion: channel_id=" << channel_id_;

  TF_ASSIGN_OR_RETURN(bool skip, ShouldSkip("waiting for send completion",
                                            params, device_constraint_));
  if (skip) return absl::OkStatus();

  TraceMe trace(
      [&] { return TraceMeEncode("SendDone", {{"channel_id", channel_id_}}); });

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto done_event, events_->Extract(executor, channel_id_));

  // Wait until send handler will record an event on the stream.
  BlockUntilReady(done_event.GetAsyncValue());
  if (done_event.IsError()) return done_event.GetError();

  VLOG(5) << "Completed Send operation: channel_id=" << channel_id_;

  // Once event is recorded we can add a stream dependency.
  return params.stream->WaitFor(done_event.get().get());
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

absl::Status HostRecvThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Recv buffer: channel_id=" << channel_id_
          << "; shape=" << shape_.ToString();

  TF_ASSIGN_OR_RETURN(
      bool skip, ShouldSkip("receiving buffer", params, device_constraint_));
  if (skip) return absl::OkStatus();

  TraceMe trace(
      [&] { return TraceMeEncode("Recv", {{"channel_id", channel_id_}}); });

  // Use host_to_device stream if it is available.
  se::Stream* stream = params.host_to_device_stream;
  if (stream) {
    TF_RETURN_IF_ERROR(stream->WaitFor(params.stream));
  } else {
    stream = params.stream;
  }

  se::DeviceMemoryBase dst =
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

//===----------------------------------------------------------------------===//
// HostRecvDoneThunk
//===----------------------------------------------------------------------===//

HostRecvDoneThunk::HostRecvDoneThunk(
    ThunkInfo thunk_info, int64_t channel_id,
    std::shared_ptr<HostSendRecvAsyncEvents> events,
    std::optional<GlobalDeviceId> device_constraint)
    : Thunk(Thunk::kHostRecvDone, thunk_info),
      channel_id_(channel_id),
      events_(std::move(events)) {}

absl::Status HostRecvDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  VLOG(3) << "Wait for recv completion: channel_id=" << channel_id_;

  TF_ASSIGN_OR_RETURN(bool skip, ShouldSkip("waiting for recv completion",
                                            params, device_constraint_));
  if (skip) return absl::OkStatus();

  TraceMe trace(
      [&] { return TraceMeEncode("RecvDone", {{"channel_id", channel_id_}}); });

  se::StreamExecutor* executor = params.stream->parent();
  TF_ASSIGN_OR_RETURN(auto done_event, events_->Extract(executor, channel_id_));

  // Wait until send handler will record an event on the stream.
  BlockUntilReady(done_event.GetAsyncValue());
  if (done_event.IsError()) return done_event.GetError();

  VLOG(5) << "Completed Recv operation: channel=" << channel_id_;

  // Once event is recorded we can add a stream dependency.
  return params.stream->WaitFor(done_event.get().get());
}

}  // namespace xla::gpu
