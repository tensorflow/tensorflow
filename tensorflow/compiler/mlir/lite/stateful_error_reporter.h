/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_STATEFUL_ERROR_REPORTER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_STATEFUL_ERROR_REPORTER_H_

// LINT.IfChange
#include <string>

#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

namespace tflite_migration {

// Similar to tflite::ErrorReporter, except that it allows callers to get the
// last error message.
class StatefulErrorReporter : public tflite::ErrorReporter {
 public:
  // Returns last error message. Returns empty string if no error is reported.
  virtual std::string message() = 0;
};

}  // namespace tflite_migration
// LINT.ThenChange(//tensorflow/lite/stateful_error_reporter.h)

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_STATEFUL_ERROR_REPORTER_H_
