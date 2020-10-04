/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// Python wrapper to modify model interface.

#include "tensorflow/lite/tools/optimize/modify_model_interface.h"

#include <string>

#include "pybind11/pybind11.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace pybind11 {

PYBIND11_MODULE(_pywrap_modify_model_interface, m) {
  // An anonymous function that invokes the C++ function
  // after applying transformations to the python function arguments
  m.def("modify_model_interface",
        [](const std::string& input_file, const std::string& output_file,
           const int input_type, const int output_type) -> int {
          return tflite::optimize::ModifyModelInterface(
              input_file, output_file,
              static_cast<tflite::TensorType>(input_type),
              static_cast<tflite::TensorType>(output_type));
        });
}

}  // namespace pybind11
