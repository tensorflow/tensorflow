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
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/gemm_thunk.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/service/buffer_assignment.h"

namespace xla::gpu {

absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProto(
    const ThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  if (thunk_proto.thunk_info().execution_stream_id() < 0) {
    return absl::InvalidArgumentError(
        "The thunk execution stream ID must be non-negative.");
  }

  Thunk::ThunkInfo thunk_info;
  thunk_info.execution_stream_id =
      thunk_proto.thunk_info().execution_stream_id();
  thunk_info.profile_annotation = thunk_proto.thunk_info().profile_annotation();

  if (thunk_proto.has_sequential_thunk()) {
    auto deserializer = [&buffer_allocations](const ThunkProto& thunk_proto) {
      return DeserializeThunkProto(thunk_proto, buffer_allocations);
    };

    return SequentialThunk::FromProto(
        std::move(thunk_info), thunk_proto.sequential_thunk(), deserializer);
  }
  if (thunk_proto.has_copy_thunk()) {
    return CopyThunk::FromProto(std::move(thunk_info), thunk_proto.copy_thunk(),
                                buffer_allocations);
  }
  if (thunk_proto.has_device_to_host_copy_thunk()) {
    return DeviceToHostCopyThunk::FromProto(
        std::move(thunk_info), thunk_proto.device_to_host_copy_thunk(),
        buffer_allocations);
  }
  if (thunk_proto.has_host_to_device_copy_thunk()) {
    return HostToDeviceCopyThunk::FromProto(
        std::move(thunk_info), thunk_proto.host_to_device_copy_thunk(),
        buffer_allocations);
  }
  if (thunk_proto.has_gemm_thunk()) {
    return GemmThunk::FromProto(std::move(thunk_info), thunk_proto.gemm_thunk(),
                                buffer_allocations);
  }
  return absl::InvalidArgumentError("Unknown thunk type found in ThunkProto.");
}

}  // namespace xla::gpu
