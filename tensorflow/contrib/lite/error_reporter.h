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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_ERROR_REPORTER_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_ERROR_REPORTER_H_

#include <cstdarg>
#include "tensorflow/contrib/lite/context.h"

namespace tflite {

// A functor that reports error to supporting system. Invoked similar to
// printf.
//
// Usage:
//  ErrorReporter foo;
//  foo.Report("test %d", 5);
// or
//  va_list args;
//  foo.Report("test %d", args); // where args is va_list
//
// Sublclass ErrorReporter to provide another reporting destination.
// For example, if you have a GUI program, you might redirect to a buffer
// that drives a GUI error log box.
class ErrorReporter {
 public:
  virtual ~ErrorReporter();
  virtual int Report(const char* format, va_list args) = 0;
  int Report(const char* format, ...);
  int ReportError(void*, const char* format, ...);
};

// An error reporter that simplify writes the message to stderr.
struct StderrReporter : public ErrorReporter {
  int Report(const char* format, va_list args) override;
};

// Return the default error reporter (output to stderr).
ErrorReporter* DefaultErrorReporter();

}  // namespace tflite

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_ERROR_REPORTER_H_
