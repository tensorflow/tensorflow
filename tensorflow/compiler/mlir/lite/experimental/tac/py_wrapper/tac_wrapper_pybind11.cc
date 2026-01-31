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

#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/compiler/mlir/lite/experimental/tac/py_wrapper/tac_wrapper.h"

// Warning: The API is experimental and subject to change.
PYBIND11_MODULE(_pywrap_tac_wrapper, m) {
  m.def(
      "run_tac",
      [](const std::string& model_file_path,
         const std::vector<std::string>& device_specs,
         const std::string& model_output_path) {
        return ::tflite::run_tac(model_file_path, device_specs,
                                 model_output_path);
      },
      R"pbdoc(
    Run target-aware-conversion with the given device specs.
  )pbdoc");
}
