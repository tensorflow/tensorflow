/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_MICRO_MICRO_ERROR_REPORTER_H_
#define TENSORFLOW_LITE_MICRO_MICRO_ERROR_REPORTER_H_

#include <cstdarg>

#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/micro/compatibility.h"

#if !defined(TF_LITE_STRIP_ERROR_STRINGS)
// This function can be used independent of the MicroErrorReporter to get
// printf-like functionalitys and are common to all target platforms.
void MicroPrintf(const char* format, ...);
#else
// We use a #define to ensure that the strings are completely stripped, to
// prevent an unnecessary increase in the binary size.
#define MicroPrintf(format, ...)
#endif

namespace tflite {

// Get a pointer to a singleton global error reporter.
ErrorReporter* GetMicroErrorReporter();

class MicroErrorReporter : public ErrorReporter {
 public:
  ~MicroErrorReporter() override {}
  int Report(const char* format, va_list args) override;

 private:
  TF_LITE_REMOVE_VIRTUAL_DELETE
};

}  // namespace tflite

#endif  // TENSORFLOW_LITE_MICRO_MICRO_ERROR_REPORTER_H_
