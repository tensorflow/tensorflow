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
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/lite/python/testdata/test_registerer.h"

PYBIND11_MODULE(_pywrap_test_registerer, m) {
  m.doc() = R"pbdoc(
    _pywrap_test_registerer
    -----
  )pbdoc";
  m.def("get_num_test_registerer_calls", &tflite::get_num_test_registerer_calls,
        R"pbdoc(
          Returns the num_test_registerer_calls counter and re-sets it.
        )pbdoc");
  m.def(
      "TF_TestRegisterer",
      [](uintptr_t resolver) {
        tflite::TF_TestRegisterer(
            reinterpret_cast<tflite::MutableOpResolver*>(resolver));
      },
      R"pbdoc(
        Dummy registerer function with the correct signature. Registers a fake
        custom op needed by test models. Increments the
        num_test_registerer_calls counter by one.
      )pbdoc");
}
