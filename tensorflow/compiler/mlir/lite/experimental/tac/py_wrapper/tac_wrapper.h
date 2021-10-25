/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_PY_WRAPPER_TAC_WRAPPER_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_PY_WRAPPER_TAC_WRAPPER_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

// Place `<locale>` before <Python.h> to avoid build failures in macOS.
#include <locale>

// The empty line above is on purpose as otherwise clang-format will
// automatically move <Python.h> before <locale>.
#include <Python.h>

namespace tflite {

// Run target-aware-conversion for the given tflite model with the given device
// specs.
// Warning: The API is experimental and subject to change.
bool run_tac(const std::string& model_file_path,
             const std::vector<std::string>& device_specs,
             const std::string& model_output_path);

}  // namespace tflite

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_TAC_PY_WRAPPER_TAC_WRAPPER_H_
