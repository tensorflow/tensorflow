/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include <stdarg.h>

#include <cstdio>

#include "tensorflow/lite/minimal_logging.h"

namespace tflite {
namespace logging_internal {

#ifndef NDEBUG
// In debug builds, default is VERBOSE.
LogSeverity MinimalLogger::minimum_log_severity_ = TFLITE_LOG_VERBOSE;
#else
// In prod builds, default is INFO.
LogSeverity MinimalLogger::minimum_log_severity_ = TFLITE_LOG_INFO;
#endif

void MinimalLogger::LogFormatted(LogSeverity severity, const char* format,
                                 va_list args) {
  if (severity >= MinimalLogger::minimum_log_severity_) {
    fprintf(stderr, "%s: ", GetSeverityName(severity));
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
    vfprintf(stderr, format, args);
#pragma clang diagnostic pop
    fputc('\n', stderr);
  }
}

}  // namespace logging_internal
}  // namespace tflite
