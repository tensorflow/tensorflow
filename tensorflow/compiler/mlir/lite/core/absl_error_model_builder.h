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
#ifndef TENSORFLOW_COMPILER_MLIR_LITE_CORE_ABSL_ERROR_MODEL_BUILDER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_CORE_ABSL_ERROR_MODEL_BUILDER_H_

#include <cstdarg>

#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"
#include "tensorflow/compiler/mlir/lite/core/model_builder_base.h"

namespace mlir::TFL {

// An error reporter that uses absl logging.
class AbslErrorReporter : public tflite::ErrorReporter {
  int Report(const char* format, va_list args) override;
};

tflite::ErrorReporter* GetAbslErrorReporter();

class FlatBufferModelAbslError
    : public tflite::impl::FlatBufferModelBase<FlatBufferModelAbslError> {
 public:
  // Use stderr_reporter as the default error reporter.
  static tflite::ErrorReporter* GetDefaultErrorReporter() {
    return GetAbslErrorReporter();
  }

  // Inherit all constructors from FlatBufferModelBase since inherited factory
  // methods refer to them.
  using FlatBufferModelBase<FlatBufferModelAbslError>::FlatBufferModelBase;
};

}  // namespace mlir::TFL

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_CORE_ABSL_ERROR_MODEL_BUILDER_H_
