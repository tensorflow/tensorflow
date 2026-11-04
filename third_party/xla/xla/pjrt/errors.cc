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

#include "xla/pjrt/errors.h"

#include <optional>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {
// The payload attached to the absl::Status returned by the compilation
// service when the compilation fails due to compilation errors.
inline constexpr absl::string_view kCompilationErrorPayload =
    "type.googleapis.com/xla.CompilationError";

bool HasCompilationErrorPayload(const absl::Status& status) {
  return status.GetPayload(kCompilationErrorPayload).has_value();
}

absl::Status SetCompilationErrorWithPayload(absl::Status status) {
  status.SetPayload(kCompilationErrorPayload, absl::Cord(""));
  return status;
}

// The payload attached to the absl::Status when execution or allocation
// fails due to out-of-memory errors.
inline constexpr absl::string_view kOomErrorPayload =
    "type.googleapis.com/pjrt.OOM";

bool HasOomErrorPayload(const absl::Status& status) {
  return status.GetPayload(kOomErrorPayload).has_value();
}

absl::Status SetOomErrorWithPayload(absl::Status status, OomReason reason) {
  status.SetPayload(kOomErrorPayload, absl::Cord(absl::StrCat(reason)));
  return status;
}

std::optional<OomReason> GetOomErrorReason(const absl::Status& status) {
  std::optional<absl::Cord> payload = status.GetPayload(kOomErrorPayload);
  if (!payload.has_value()) {
    return std::nullopt;
  }

  for (OomReason reason : {OomReason::kOutOfMemory, OomReason::kFragmentation,
                           OomReason::kUnknown}) {
    if (*payload == absl::StrCat(reason)) {
      return reason;
    }
  }

  return OomReason::kUnknown;
}
}  // namespace xla
