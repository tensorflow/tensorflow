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

#include "tensorflow/lite/minimal_logging.h"

#include <cstdarg>

namespace tflite {
namespace logging_internal {

void MinimalLogger::Log(LogSeverity severity, const char* format, ...) {
  va_list args;
  va_start(args, format);
  VLog(severity, format, args);
  va_end(args);
}

const char* MinimalLogger::GetSeverityName(LogSeverity severity) {
  switch (severity) {
    case TFLITE_LOG_INFO:
      return "INFO";
    case TFLITE_LOG_WARNING:
      return "WARNING";
    case TFLITE_LOG_ERROR:
      return "ERROR";
    default:
      return "<Unknown severity>";
  }
}

}  // namespace logging_internal
}  // namespace tflite
