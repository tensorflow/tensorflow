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

#include "xla/backends/gpu/runtime/async_thunk.h"

#include <functional>
#include <memory>
#include <utility>

#include "absl/base/casts.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xla/backends/gpu/runtime/async_execution.h"
#include "xla/backends/gpu/runtime/collective_params.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/thunk_executor.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// AsyncStartThunk
//===----------------------------------------------------------------------===//

AsyncStartThunk::AsyncStartThunk(ThunkInfo thunk_info, AsyncKind async_kind,
                                 ThunkSequence thunks)
    : Thunk(Thunk::kAsyncStart, std::move(thunk_info)),
      async_kind_(async_kind),
      executor_(std::move(thunks)),
      async_execution_(std::make_shared<AsyncExecution>(this)) {}

AsyncStartThunk::AsyncStartThunk(
    ThunkInfo thunk_info, AsyncKind async_kind, ThunkSequence thunks,
    std::shared_ptr<AsyncExecution> async_execution)
    : Thunk(Thunk::kAsyncStart, std::move(thunk_info)),
      async_kind_(async_kind),
      executor_(std::move(thunks)),
      async_execution_(std::move(async_execution)) {}

absl::Status AsyncStartThunk::Prepare(const PrepareParams& params) {
  return executor_.Prepare(params);
}

absl::Status AsyncStartThunk::Initialize(const InitializeParams& params) {
  RETURN_IF_ERROR(executor_.Initialize(params));
  return async_execution_->Initialize(params.execution_scoped_state,
                                      params.executor);
}

absl::Status AsyncStartThunk::ExecuteOnStream(const ExecuteParams& params) {
  auto get_async_stream = [&]() -> absl::StatusOr<se::Stream*> {
    switch (async_kind_) {
      case AsyncKind::kCompute:
        return GetStreamForExecution(execution_stream_id(), params);
      case AsyncKind::kCommunication:
        return params.collective_params->async_streams.at(
            execution_stream_id().value());
    }
  };

  ASSIGN_OR_RETURN(se::Stream * async_stream, std::invoke(get_async_stream));
  XLA_VLOG_DEVICE(1, async_stream->parent()->device_ordinal())
      << absl::StreamFormat(
             "Execute async %s for `%s`: stream_id=%v, stream=%p",
             async_kind_ == AsyncKind::kCompute ? "compute" : "communication",
             profile_annotation(), execution_stream_id(), async_stream);

  // Execute the nested thunks on the async stream. The guard will record the
  // completion event when it goes out of scope.
  ASSIGN_OR_RETURN(auto guard,
                   async_execution_->Start(params.execution_scoped_state,
                                           params.stream, async_stream));
  return executor_.ExecuteOnStream(params.WithComputeStream(async_stream));
}

absl::Status AsyncStartThunk::WalkNested(Walker callback) {
  return executor_.thunks().WalkNested(callback);
}

absl::Status AsyncStartThunk::TransformNested(Transformer callback) {
  return executor_.thunks().TransformNested(callback);
}

AsyncExecutionId AsyncStartThunk::async_execution_id() const {
  return absl::bit_cast<AsyncExecutionId>(async_execution_.get());
}

std::shared_ptr<AsyncExecution> AsyncStartThunk::async_execution() const {
  return async_execution_;
}

//===----------------------------------------------------------------------===//
// AsyncDoneThunk
//===----------------------------------------------------------------------===//

AsyncDoneThunk::AsyncDoneThunk(ThunkInfo thunk_info,
                               std::shared_ptr<AsyncExecution> async_execution)
    : Thunk(Thunk::kAsyncDone, std::move(thunk_info)),
      async_execution_(std::move(async_execution)) {}

absl::Status AsyncDoneThunk::ExecuteOnStream(const ExecuteParams& params) {
  return async_execution_->Done(params.execution_scoped_state, params.stream);
}

AsyncExecutionId AsyncDoneThunk::async_execution_id() const {
  return absl::bit_cast<AsyncExecutionId>(async_execution_.get());
}

std::shared_ptr<AsyncExecution> AsyncDoneThunk::async_execution() const {
  return async_execution_;
}

//===----------------------------------------------------------------------===//
// AsyncStartThunk/AsyncDoneThunk serialization
//===----------------------------------------------------------------------===//

static AsyncStartThunkProto::AsyncKind ToProtoAsyncKind(
    AsyncStartThunk::AsyncKind kind) {
  switch (kind) {
    case AsyncStartThunk::AsyncKind::kCommunication:
      return AsyncStartThunkProto::ASYNC_KIND_COMMUNICATION;
    case AsyncStartThunk::AsyncKind::kCompute:
      return AsyncStartThunkProto::ASYNC_KIND_COMPUTE;
  }
}

static absl::StatusOr<AsyncStartThunk::AsyncKind> FromProtoAsyncKind(
    AsyncStartThunkProto::AsyncKind kind) {
  switch (kind) {
    case AsyncStartThunkProto::ASYNC_KIND_COMMUNICATION:
      return AsyncStartThunk::AsyncKind::kCommunication;
    case AsyncStartThunkProto::ASYNC_KIND_COMPUTE:
      return AsyncStartThunk::AsyncKind::kCompute;
    default:
      return Internal("Unknown async kind: %d", kind);
  }
}

absl::StatusOr<ThunkProto> AsyncStartThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AsyncStartThunkProto* start_proto = proto.mutable_async_start_thunk();
  start_proto->set_async_execution_id(async_execution_id().value());
  start_proto->set_async_kind(ToProtoAsyncKind(async_kind_));
  start_proto->mutable_thunks();
  for (const auto& thunk : executor_.thunks()) {
    ASSIGN_OR_RETURN(*start_proto->mutable_thunks()->add_thunks(),
                     thunk->ToProto());
  }
  return proto;
}

absl::StatusOr<std::unique_ptr<AsyncStartThunk>> AsyncStartThunk::FromProto(
    ThunkInfo thunk_info, const AsyncStartThunkProto& proto,
    const Deserializer& deserializer, AsyncExecutionMap& async_executions) {
  ASSIGN_OR_RETURN(AsyncStartThunk::AsyncKind async_kind,
                   FromProtoAsyncKind(proto.async_kind()));

  ThunkSequence nested;
  for (const auto& thunk_proto : proto.thunks().thunks()) {
    ASSIGN_OR_RETURN(std::unique_ptr<Thunk> thunk, deserializer(thunk_proto));
    nested.push_back(std::move(thunk));
  }

  auto start_thunk = std::make_unique<AsyncStartThunk>(
      std::move(thunk_info), async_kind, std::move(nested));

  AsyncExecutionId id(proto.async_execution_id());
  async_executions[id] = start_thunk->async_execution();

  return start_thunk;
}

absl::StatusOr<ThunkProto> AsyncDoneThunk::ToProto() const {
  ThunkProto proto;
  *proto.mutable_thunk_info() = thunk_info().ToProto();

  AsyncDoneThunkProto* done_proto = proto.mutable_async_done_thunk();
  done_proto->set_async_execution_id(async_execution_id().value());
  return proto;
}

absl::StatusOr<std::unique_ptr<AsyncDoneThunk>> AsyncDoneThunk::FromProto(
    ThunkInfo thunk_info, const AsyncDoneThunkProto& proto,
    AsyncExecutionMap& async_executions) {
  AsyncExecutionId id(proto.async_execution_id());
  auto it = async_executions.find(id);
  if (it == async_executions.end()) {
    return Internal(
        "Async execution not found for id %d. AsyncStartThunk must be "
        "deserialized before AsyncDoneThunk.",
        id.value());
  }
  return std::make_unique<AsyncDoneThunk>(std::move(thunk_info), it->second);
}

}  // namespace xla::gpu
