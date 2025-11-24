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

#ifndef XLA_ERROR_INTERNAL_CHECK_HELPER_H_
#define XLA_ERROR_INTERNAL_CHECK_HELPER_H_

#include <cstdint>
#include <ostream>
#include <sstream>
#include <string>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/error/debug_me_context_util.h"
#include "xla/error/error_codes.h"

namespace xla::error {

// Enum to control the termination behavior of the CheckHelper.
enum class CheckType : std::uint8_t { kFatal, kQFatal };

// Helper class to pretty print the debug context and user message when a check
// fails.
class CheckHelper {
 public:
  CheckHelper(const char* absl_nonnull file, int line,
              absl::string_view condition_text,
              CheckType check_type = CheckType::kFatal)
      : file_(absl::string_view(file)),
        line_(line),
        condition_text_(condition_text),
        check_type_(check_type) {}

  // Non-copyable/movable.
  CheckHelper(const CheckHelper&) = delete;
  CheckHelper& operator=(const CheckHelper&) = delete;

  // *always* terminates via LOG(FATAL).
  [[noreturn]] ~CheckHelper() {
    std::string error_message = absl::StrCat(
        GetErrorCodeAndName(ErrorCode::kInternal),
        ": Check failed: ", condition_text_, " ", user_stream_.str(), "\n",
        DebugMeContextToErrorMessageString());

    switch (check_type_) {
      case CheckType::kFatal:
        LOG(FATAL).AtLocation(file_, line_) << error_message;
        break;
      case CheckType::kQFatal:
        LOG(QFATAL).AtLocation(file_, line_) << error_message;
        break;
    }
  }

  std::ostream& InternalStream() { return user_stream_; }

 private:
  absl::string_view file_;
  int line_;
  absl::string_view condition_text_;
  CheckType check_type_;
  std::ostringstream user_stream_;
};

}  // namespace xla::error

#endif  // XLA_ERROR_INTERNAL_CHECK_HELPER_H_
