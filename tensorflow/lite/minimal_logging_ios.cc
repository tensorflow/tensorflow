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

#include <syslog.h>
#include <cstdarg>

namespace tflite {
namespace logging_internal {
namespace {

int GetPlatformSeverity(LogSeverity severity) {
  switch (severity) {
    case TFLITE_LOG_INFO:
      return LOG_INFO;
    case TFLITE_LOG_WARNING:
      return LOG_WARNING;
    case TFLITE_LOG_ERROR:
      return LOG_ERR;
    default:
      return LOG_DEBUG;
  }
}

}  // namespace

void MinimalLogger::VLog(LogSeverity severity, const char* format,
                         va_list args) {
  // TODO(b/123704468): Use os_log when available.
  vsyslog(GetPlatformSeverity(severity), format, args);
}

}  // namespace logging_internal
}  // namespace tflite
