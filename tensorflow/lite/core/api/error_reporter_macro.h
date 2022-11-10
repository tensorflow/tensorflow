/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_CORE_API_ERROR_REPORTER_MACRO_H_
#define TENSORFLOW_LITE_CORE_API_ERROR_REPORTER_MACRO_H_

#include "tensorflow/lite/core/api/error_reporter.h"

// This Macro is intended to be used only by the code that TFLite Micro
// shares with TFLite. In the code that is shared with TFLM you should
// not make bare calls to the error reporter, instead use the
// TF_LITE_REPORT_ERROR macro, since this allows message strings to be
// stripped when the binary size has to be optimized. If you are looking to
// reduce binary size, define TF_LITE_STRIP_ERROR_STRINGS when compiling and
// every call will be stubbed out, taking no memory.
#ifndef TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_REPORT_ERROR(reporter, ...)                             \
  do {                                                                  \
    static_cast<tflite::ErrorReporter*>(reporter)->Report(__VA_ARGS__); \
  } while (false)
#else  // TF_LITE_STRIP_ERROR_STRINGS
#define TF_LITE_REPORT_ERROR(reporter, ...)
#endif  // TF_LITE_STRIP_ERROR_STRINGS

#endif  // TENSORFLOW_LITE_CORE_API_ERROR_REPORTER_MACRO_H_
