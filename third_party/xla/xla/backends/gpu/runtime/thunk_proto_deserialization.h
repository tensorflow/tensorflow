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

#ifndef XLA_BACKENDS_GPU_RUNTIME_THUNK_PROTO_DESERIALIZATION_H_
#define XLA_BACKENDS_GPU_RUNTIME_THUNK_PROTO_DESERIALIZATION_H_

#include <memory>
#include <optional>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/buffer_assignment.h"
#include "xla/stream_executor/kernel_spec.h"

namespace xla::gpu {

// Deserializes the given `thunk_proto` into a Thunk.
// - `buffer_allocations` is used to deserialize buffer slices.
// - `hlo_module` is used to deserialize thunks that reference HLO instructions.
// - `platform_name` is used to look up platform-specific kernels in the
//   GpuKernelRegistry.
// - `symbol_resolver` is used to deserialize custom kernels where the kernel is
//   not inlined in the proto, but rather loaded at runtime via symbol
//   resolution.
absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProto(
    const ThunkProto& thunk_proto,
    absl::Span<const BufferAllocation> buffer_allocations,
    const HloModule* absl_nullable hlo_module, absl::string_view platform_name,
    const se::GpuComputeCapability& gpu_compute_capability,
    const std::optional<stream_executor::KernelLoaderSpec::SymbolResolver>&
        symbol_resolver = std::nullopt);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_PROTO_DESERIALIZATION_H_
