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

#include "xla/error/debug_me_context_util.h"

#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/debug_me_context.h"
#include "tsl/platform/platform.h"

namespace xla::error {

std::string DebugMeContextToErrorMessageString() {
  if (!tsl::DebugMeContext<DebugMeContextKey>::HasAnyValues()) {
    return "";
  }

  std::string error_message = "DebugMeContext:\n";
  {
    const std::vector<std::string> compiler_values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(
            DebugMeContextKey::kCompiler);
    if (!compiler_values.empty()) {
      absl::StrAppend(&error_message,
                      "Compiler: ", absl::StrJoin(compiler_values, "/"), "\n");
    }
  }
  {
    const std::vector<std::string> hlo_pass_values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(
            DebugMeContextKey::kHloPass);
    if (!hlo_pass_values.empty()) {
      absl::StrAppend(&error_message,
                      "HLO Passes: ", absl::StrJoin(hlo_pass_values, "/"),
                      "\n");
    }
  }
  {
    const std::vector<std::string> hlo_instruction_values =
        tsl::DebugMeContext<DebugMeContextKey>::GetValues(
            DebugMeContextKey::kHloInstruction);
    if (!hlo_instruction_values.empty()) {
      absl::StrAppend(&error_message, "HLO Instructions: ",
                      absl::StrJoin(hlo_instruction_values, "/"), "\n");
    }
  }
  return error_message;
}

void AttachDebugMeContextPayload(absl::Status& status) {
  if (!status.ok()) {
    std::string error_message_string = DebugMeContextToErrorMessageString();
    if (!error_message_string.empty()) {
      status.SetPayload(kDebugContextPayloadUrl,
                        absl::Cord(error_message_string));
    }
  }
}

absl::Status FlattenDebugPayloadIntoMessage(const absl::Status& status) {
  if (status.ok()) {
    return status;
  }

  std::optional<absl::Cord> debug_context_payload =
      status.GetPayload(kDebugContextPayloadUrl);
  if (!debug_context_payload.has_value()) {
    return status;
  }

  std::string new_message =
      absl::StrCat(status.message(), "\n", debug_context_payload.value());
#if defined(PLATFORM_GOOGLE)
  absl::Status new_status(status.code(), new_message,
                          status.GetSourceLocations().front());
#else   // ndef PLATFORM_GOOGLE
  absl::Status new_status(status.code(), new_message);
#endif  // ndef PLATFORM_GOOGLE

  // Copy all other payloads from the old status to the new one.
  status.ForEachPayload([&](absl::string_view type_url, const absl::Cord& p) {
    if (type_url != kDebugContextPayloadUrl) {
      new_status.SetPayload(type_url, p);
    }
  });

  // Replace the original status with our new, updated one.
  return new_status;
}

}  // namespace xla::error
