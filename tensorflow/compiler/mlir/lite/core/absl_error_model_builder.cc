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
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"

#include <cstdarg>
#include <cstdio>

#include "absl/log/log.h"
#include "tensorflow/compiler/mlir/lite/core/api/error_reporter.h"

namespace mlir::TFL {

int AbslErrorReporter::Report(const char* format, va_list args) {
  char buffer[1024];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wformat-nonliteral"
  vsprintf(buffer, format, args);
#pragma clang diagnostic pop
  LOG(ERROR) << buffer;
  return 0;
}

tflite::ErrorReporter* GetAbslErrorReporter() {
  static AbslErrorReporter* error_reporter = new AbslErrorReporter;
  return error_reporter;
}

}  // namespace mlir::TFL
