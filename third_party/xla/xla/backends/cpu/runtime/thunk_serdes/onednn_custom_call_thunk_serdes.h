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

#ifndef XLA_BACKENDS_CPU_RUNTIME_THUNK_SERDES_ONEDNN_CUSTOM_CALL_THUNK_SERDES_H_
#define XLA_BACKENDS_CPU_RUNTIME_THUNK_SERDES_ONEDNN_CUSTOM_CALL_THUNK_SERDES_H_

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

// SerDes for oneDNN custom call thunks (`OneDnnOpThunk`).
//
// This header has two implementations: `onednn_custom_call_thunk_serdes.cc`
// used when XLA is built with the oneDNN async runtime, and
// `onednn_custom_call_thunk_serdes_stub.cc` used otherwise.

// Serializes `thunk` into `proto` if it is a `OneDnnOpThunk` and returns
// `std::nullopt` otherwise. Returns an error if the thunk's custom call target
// is not a oneDNN target known to the thunk runtime, or if its backend config
// does not match the target.
std::optional<absl::Status> TrySerializeOneDnnCustomCall(const Thunk& thunk,
                                                         ThunkProto& proto);

// Deserializes a thunk previously serialized by `TrySerializeOneDnnCustomCall`.
absl::StatusOr<std::unique_ptr<Thunk>> OneDnnCustomCallFromProto(
    const OneDnnCustomCallProto& onednn_proto, Thunk::Info info,
    const std::vector<BufferAllocation>& buffer_allocations);

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_RUNTIME_THUNK_SERDES_ONEDNN_CUSTOM_CALL_THUNK_SERDES_H_
