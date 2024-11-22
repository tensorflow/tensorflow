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

#ifndef TENSORFLOW_TSL_PLATFORM_LOGGING_H_
#define TENSORFLOW_TSL_PLATFORM_LOGGING_H_

#include <string>

#include "absl/base/log_severity.h"
#include "absl/log/check.h"       // IWYU pragma: export
#include "absl/log/log.h"         // IWYU pragma: export
#include "absl/log/vlog_is_on.h"  // IWYU pragma: export

namespace tsl {

// These constants are also useful outside of LOG statements, e.g. in mocked
// logs.
using base_logging::ERROR;
using base_logging::FATAL;
using base_logging::INFO;
using base_logging::NUM_SEVERITIES;
using base_logging::WARNING;

namespace internal {
inline void LogString(const char* fname, int line, absl::LogSeverity severity,
                      const std::string& message) {
  LOG(LEVEL(severity)).AtLocation(fname, line) << message;
}

#ifndef CHECK_NOTNULL
template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LOG(FATAL).AtLocation(file, line) << std::string(exprtext);  // Crash OK
  }
  return std::forward<T>(t);
}

#define CHECK_NOTNULL(val)                          \
  ::tsl::internal::CheckNotNull(__FILE__, __LINE__, \
                                "'" #val "' Must be non NULL", (val))
#endif  // CHECK_NOTNULL

}  // namespace internal

// Change verbose level of pre-defined files if envorionment
// variable `env_var` is defined. This is currently a no op in OSS.
void UpdateLogVerbosityIfDefined(const char* env_var);

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_LOGGING_H_
