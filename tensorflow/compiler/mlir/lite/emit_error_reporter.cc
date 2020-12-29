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

#include "tensorflow/compiler/mlir/lite/emit_error_reporter.h"

namespace tflite {

int EmitErrorReporter::Report(const char* format, va_list args) {
  std::vector<char> buf(1 + snprintf(nullptr, 0, format, args));
  std::vsnprintf(buf.data(), buf.size(), format, args);
  module_.emitError() << std::string(buf.begin(), buf.end());
  return 0;
}

}  // namespace tflite
