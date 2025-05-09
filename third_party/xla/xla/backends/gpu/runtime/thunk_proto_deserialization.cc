/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/runtime/thunk_proto_deserialization.h"

#include <memory>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProto(
    const ThunkProto& thunk_proto) {
  if (thunk_proto.thunk_info().execution_stream_id() < 0) {
    return absl::InvalidArgumentError(
        "The thunk execution stream ID must be non-negative.");
  }

  Thunk::ThunkInfo thunk_info;
  thunk_info.execution_stream_id =
      thunk_proto.thunk_info().execution_stream_id();
  thunk_info.profile_annotation = thunk_proto.thunk_info().profile_annotation();

  if (thunk_proto.has_sequential_thunk()) {
    return SequentialThunk::FromProto(
        thunk_info, thunk_proto.sequential_thunk(), DeserializeThunkProto);
  }

  return absl::InvalidArgumentError("Unknown thunk type found in ThunkProto.");
}

}  // namespace xla::gpu
