/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/error_util.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/errors.h"

namespace xla {

absl::Status WrapWithPythonStacktrace(absl::Status status,
                                      absl::string_view stack_trace) {
  if (!status.ok() && !stack_trace.empty()) {
    if (!status.GetPayload(kPythonStackTracePayloadKey).has_value()) {
      absl::string_view truncated_trace = stack_trace;
      int newline_count = 0;
      size_t pos = 0;
      while (newline_count < 50) {
        size_t next_newline = stack_trace.find('\n', pos);
        if (next_newline == absl::string_view::npos) {
          break;
        }
        newline_count++;
        pos = next_newline + 1;
      }

      bool truncated = false;
      if (newline_count == 50 && pos < stack_trace.size()) {
        truncated_trace = stack_trace.substr(0, pos);
        truncated = true;
      }

      if (truncated) {
        tsl::errors::AppendToMessage(
            &status,
            absl::StrFormat("\n\nSuspected Python Code Location:\n%s[... "
                            "truncated to 50 lines ...]",
                            truncated_trace));
      } else {
        tsl::errors::AppendToMessage(
            &status, absl::StrFormat("\n\nSuspected Python Code Location:\n%s",
                                     stack_trace));
      }
      status.SetPayload(kPythonStackTracePayloadKey, absl::Cord(""));
    }
  }
  return status;
}

}  // namespace xla
