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

// Stub implementation of the oneDNN custom call thunk SerDes used when XLA is
// built without the oneDNN async runtime, where no `OneDnnOpThunk` exists.

#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/cpu/runtime/thunk.h"
#include "xla/backends/cpu/runtime/thunk.pb.h"
#include "xla/backends/cpu/runtime/thunk_serdes/onednn_custom_call_thunk_serdes.h"
#include "xla/service/buffer_assignment.h"

namespace xla::cpu {

std::optional<absl::Status> TrySerializeOneDnnCustomCall(const Thunk& thunk,
                                                         ThunkProto& proto) {
  return std::nullopt;
}

absl::StatusOr<std::unique_ptr<Thunk>> OneDnnCustomCallFromProto(
    const OneDnnCustomCallProto& onednn_proto, Thunk::Info info,
    const std::vector<BufferAllocation>& buffer_allocations) {
  return absl::UnimplementedError(
      "Cannot deserialize a OneDNN custom call thunk: XLA was built without "
      "oneDNN async runtime support.");
}

}  // namespace xla::cpu
