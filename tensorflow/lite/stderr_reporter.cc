/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/stderr_reporter.h"

#include <stdarg.h>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"

namespace tflite {

int StderrReporter::Report(const char* format, va_list args) {
  logging_internal::MinimalLogger::LogFormatted(TFLITE_LOG_ERROR, format, args);
  return 0;
}

ErrorReporter* DefaultErrorReporter() {
  static StderrReporter* error_reporter = new StderrReporter;
  return error_reporter;
}

}  // namespace tflite
