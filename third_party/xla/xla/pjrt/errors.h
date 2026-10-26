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

#ifndef XLA_PJRT_ERRORS_H_
#define XLA_PJRT_ERRORS_H_

#include <optional>

#include "absl/status/status.h"

namespace xla {

// Returns true if the status has the compilation error payload.
bool HasCompilationErrorPayload(const absl::Status& status);

// Sets the payload of the compilation error status to the compilation error
// payload. Useful to denote compilation errors separately from other errors.
absl::Status SetCompilationErrorWithPayload(absl::Status status);

// Returns true if the status has the OOM error payload.
bool HasOomErrorPayload(const absl::Status& status);

// Reason for the OOM error.
enum class OomReason {
  // The requested size is larger than the total free HBM.
  kOutOfMemory,
  // Enough HBM is free in total, but no contiguous region is large enough.
  kFragmentation,
  // Catch-all.
  kUnknown,
};

template <typename Sink>
void AbslStringify(Sink& sink, OomReason reason) {
  switch (reason) {
    case OomReason::kOutOfMemory:
      sink.Append("out_of_memory");
      break;
    case OomReason::kFragmentation:
      sink.Append("fragmentation");
      break;
    case OomReason::kUnknown:
      sink.Append("unknown");
      break;
  }
}

// Sets the payload of the status to denote an out-of-memory error, with the
// `reason` why memory ran out as the payload's value.
absl::Status SetOomErrorWithPayload(absl::Status status, OomReason reason);

// Returns the reason passed to SetOomErrorWithPayload(), OomReason::kUnknown if
// the payload's value isn't a known reason, or std::nullopt if the status
// doesn't have the OOM error payload.
std::optional<OomReason> GetOomErrorReason(const absl::Status& status);
}  // namespace xla

#endif  // XLA_PJRT_ERRORS_H_
