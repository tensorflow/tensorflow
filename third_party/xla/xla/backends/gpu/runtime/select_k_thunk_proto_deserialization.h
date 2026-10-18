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

#ifndef XLA_BACKENDS_GPU_RUNTIME_SELECT_K_THUNK_PROTO_DESERIALIZATION_H_
#define XLA_BACKENDS_GPU_RUNTIME_SELECT_K_THUNK_PROTO_DESERIALIZATION_H_

#include <memory>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Deserializes a SelectKThunkProto into a CustomCallThunk since SelectK
// is no longer a separate thunk kind.
// TODO: Remove this case on Apr 30, 2027
absl::StatusOr<std::unique_ptr<CustomCallThunk>> DeserializeSelectKThunkProto(
    Thunk::ThunkInfo thunk_info, const SelectKThunkProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    absl::string_view platform_name,
    const stream_executor::GpuComputeCapability& gpu_compute_capability);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_SELECT_K_THUNK_PROTO_DESERIALIZATION_H_
