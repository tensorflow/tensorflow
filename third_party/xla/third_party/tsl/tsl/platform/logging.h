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
#include <vector>

#include "tsl/platform/platform.h"

#if defined(PLATFORM_GOOGLE) || defined(PLATFORM_GOOGLE_ANDROID) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(GOOGLE_LOGGING) ||      \
    defined(__EMSCRIPTEN__) || defined(PLATFORM_CHROMIUMOS)
#include "tsl/platform/google/logging.h"  // IWYU pragma: export
#else
#include "tsl/platform/default/logging.h"  // IWYU pragma: export
#endif

#include "absl/log/check.h"       // IWYU pragma: export
#include "absl/log/log.h"         // IWYU pragma: export
#include "absl/log/vlog_is_on.h"  // IWYU pragma: export

namespace tsl {

// These constants are also useful outside of LOG statements, e.g. in mocked
// logs.
const int INFO = 0;
const int WARNING = 1;
const int ERROR = 2;
const int FATAL = 3;
const int NUM_SEVERITIES = 4;

// Adapt Absl LogSink interface to the TF interface.
using TFLogSink = ::absl::LogSink;
using TFLogEntry = ::absl::LogEntry;

// Add or remove a `LogSink` as a consumer of logging data.  Thread-safe.
void TFAddLogSink(TFLogSink* sink);
void TFRemoveLogSink(TFLogSink* sink);

// Get all the log sinks.  Thread-safe.
std::vector<TFLogSink*> TFGetLogSinks();

// Change verbose level of pre-defined files if environment
// variable `env_var` is defined.
void UpdateLogVerbosityIfDefined(const char* env_var);

namespace internal {
inline void LogString(const char* fname, int line, int severity,
                      const std::string& message) {
  LOG(LEVEL(severity)).AtLocation(fname, line) << message;
}

#ifndef CHECK_NOTNULL
template <typename T>
T&& CheckNotNull(const char* file, int line, const char* exprtext, T&& t) {
  if (t == nullptr) {
    LOG(FATAL).AtLocation(file, line) << std::string(exprtext);
  }
  return std::forward<T>(t);
}

#define CHECK_NOTNULL(val)                          \
  ::tsl::internal::CheckNotNull(__FILE__, __LINE__, \
                                "'" #val "' Must be non NULL", (val))
#endif  // CHECK_NOTNULL

}  // namespace internal

}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_LOGGING_H_
