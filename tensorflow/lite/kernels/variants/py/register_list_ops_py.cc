/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
#include <cstdint>

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/lite/kernels/variants/register_list_ops.h"
#include "tensorflow/lite/mutable_op_resolver.h"

PYBIND11_MODULE(register_list_ops_py, m) {
  m.doc() = R"pbdoc(
    Bindings to register list ops with python interpreter.
  )pbdoc";
  m.def(
      "TFLRegisterListOps",
      [](uintptr_t resolver) {
        ::tflite::variants::ops::RegisterListOps(
            reinterpret_cast<::tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
        Register all custom list ops.
      )pbdoc");
}
