/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_TSL_PLATFORM_LOGGING_H_
#define XLA_TSL_PLATFORM_LOGGING_H_

#include "absl/log/absl_log.h"
#include "absl/log/check.h"       // IWYU pragma: export
#include "absl/log/log.h"         // IWYU pragma: export
#include "absl/log/vlog_is_on.h"  // IWYU pragma: export
#include "absl/strings/string_view.h"

namespace tsl {
namespace internal {

#ifndef CHECK_NOTNULL
template <typename T>
T&& CheckNotNull(absl::string_view file, int line, absl::string_view exprtext,
                 T&& t) {
  if (t == nullptr) {
    // Use ABSL_LOG instead of LOG to avoid conflicts if downstream
    // projects (e.g. pytorch) define their own LOG macro.
    ABSL_LOG(FATAL).AtLocation(file, line) << exprtext;
  }
  return std::forward<T>(t);
}

#define CHECK_NOTNULL(val)                          \
  ::tsl::internal::CheckNotNull(__FILE__, __LINE__, \
                                "'" #val "' Must be non NULL", (val))
#endif  // CHECK_NOTNULL

}  // namespace internal

void UpdateLogVerbosityIfDefined(absl::string_view env_var);

}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_LOGGING_H_
