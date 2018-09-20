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
#ifndef TENSORFLOW_CONTRIB_LITE_STDERR_REPORTER_H_
#define TENSORFLOW_CONTRIB_LITE_STDERR_REPORTER_H_

#include <cstdarg>
#include "tensorflow/contrib/lite/c/c_api_internal.h"
#include "tensorflow/contrib/lite/core/api/error_reporter.h"

namespace tflite {

// An error reporter that simplify writes the message to stderr.
struct StderrReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override;
};

// Return the default error reporter (output to stderr).
ErrorReporter* DefaultErrorReporter();

}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_STDERR_REPORTER_H_
