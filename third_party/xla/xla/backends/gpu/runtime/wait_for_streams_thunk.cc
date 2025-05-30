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

#include "xla/backends/gpu/runtime/wait_for_streams_thunk.h"

#include <memory>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/stream_executor/stream.h"
#include "tsl/platform/statusor.h"

namespace xla::gpu {

absl::Status WaitForStreamsThunk::ExecuteOnStream(const ExecuteParams& params) {
  TF_ASSIGN_OR_RETURN(se::Stream * stream,
                      Thunk::GetStreamForExecution(stream_id_, params));

  VLOG(5) << "Waiting for stream id: " << wait_for_stream_id_;
  TF_ASSIGN_OR_RETURN(
      se::Stream * wait_on_stream,
      Thunk::GetStreamForExecution(wait_for_stream_id_, params));

  return stream->WaitFor(wait_on_stream);
}

absl::StatusOr<ThunkProto> WaitForStreamsThunk::ToProto() const {
  TF_ASSIGN_OR_RETURN(ThunkProto proto, Thunk::ToProto());
  WaitForStreamsThunkProto* wait_for_streams_thunk_proto =
      proto.mutable_wait_for_streams_thunk();
  wait_for_streams_thunk_proto->set_stream_id(stream_id_.value());
  wait_for_streams_thunk_proto->set_wait_for_stream_id(
      wait_for_stream_id_.value());
  return proto;
}

absl::StatusOr<std::unique_ptr<WaitForStreamsThunk>>
WaitForStreamsThunk::FromProto(ThunkInfo thunk_info,
                               const WaitForStreamsThunkProto& proto) {
  if (proto.stream_id() < 0) {
    return absl::InvalidArgumentError(
        "Failed to deserialize WaitForStreamsThunkProto: stream_id must be "
        "non-negative.");
  }
  if (proto.wait_for_stream_id() < 0) {
    return absl::InvalidArgumentError(
        "Failed to deserialize WaitForStreamsThunkProto: wait_for_stream_id "
        "must be non-negative.");
  }
  return std::make_unique<WaitForStreamsThunk>(
      thunk_info, ExecutionStreamId(proto.stream_id()),
      ExecutionStreamId(proto.wait_for_stream_id()));
}

}  // namespace xla::gpu
