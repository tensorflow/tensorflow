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

#ifndef TENSORFLOW_LITE_PROFILING_MODEL_RUNTIME_INFO_H_
#define TENSORFLOW_LITE_PROFILING_MODEL_RUNTIME_INFO_H_

#include "absl/strings/string_view.h"
#include "tensorflow/lite/core/interpreter.h"

namespace tflite {
namespace profiling {

// Generates a ModelRuntimeInfo proto for the given interpreter and writes it to
// the given output file path.
TfLiteStatus GenerateModelRuntimeInfo(const Interpreter &interpreter,
                                      absl::string_view output_file_path);
}  // namespace profiling
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_MODEL_RUNTIME_INFO_H_
