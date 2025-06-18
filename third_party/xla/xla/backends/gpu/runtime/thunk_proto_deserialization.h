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

#include "absl/status/statusor.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"

namespace xla::gpu {

// Deserializes the given `thunk_proto` into a Thunk.
absl::StatusOr<std::unique_ptr<Thunk>> DeserializeThunkProto(
    const ThunkProto& thunk_proto);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_THUNK_PROTO_DESERIALIZATION_H_
