// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/logging.h"

#include <cstdarg>

#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace litert {
namespace internal {

namespace {

inline tflite::LogSeverity ConvertSeverity(LogSeverity severity) {
  return static_cast<tflite::LogSeverity>(severity);
}

inline LogSeverity ConvertSeverity(tflite::LogSeverity severity) {
  return static_cast<LogSeverity>(severity);
}

}  // namespace

void Logger::Log(LogSeverity severity, const char* format, ...) {
  va_list args;
  va_start(args, format);
  tflite::logging_internal::MinimalLogger::LogFormatted(
      ConvertSeverity(severity), format, args);
  va_end(args);
}

LogSeverity Logger::GetMinimumSeverity() {
  return ConvertSeverity(
      tflite::logging_internal::MinimalLogger::GetMinimumLogSeverity());
}

LogSeverity Logger::SetMinimumSeverity(LogSeverity new_severity) {
  return ConvertSeverity(
      tflite::logging_internal::MinimalLogger::SetMinimumLogSeverity(
          ConvertSeverity(new_severity)));
}

}  // namespace internal
}  // namespace litert
