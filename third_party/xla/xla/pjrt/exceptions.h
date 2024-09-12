/* Copyright 2022 The OpenXLA Authors.

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
#ifndef XLA_PJRT_EXCEPTIONS_H_
#define XLA_PJRT_EXCEPTIONS_H_

#include <cstdlib>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"

namespace xla {

// Custom exception type used such that we can raise XlaRuntimeError in
// Python code instead of RuntimeError.
class XlaRuntimeError : public std::runtime_error {
 public:
  explicit XlaRuntimeError(absl::Status status)
      : std::runtime_error(StatusToString(status)), status_(status) {
    CHECK(!status_->ok());
  }

  explicit XlaRuntimeError(const std::string what) : std::runtime_error(what) {}

  std::optional<absl::Status> status() const { return status_; }

 private:
  static std::string StatusToString(const absl::Status& st) {
    if (!ShowStackTraces()) {
      return st.ToString(absl::StatusToStringMode::kWithNoExtraData);
    }
    std::stringstream ss;
    ss << st;
    return ss.str();
  }

  static bool ShowStackTraces() {
    if (char* env = getenv("JAX_TRACEBACK_FILTERING")) {
      return absl::string_view(env) == "off";
    }
    return false;
  }

  std::optional<absl::Status> status_;
};

}  // namespace xla

#endif  // XLA_PJRT_EXCEPTIONS_H_
