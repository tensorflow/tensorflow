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

#include "pybind11/pybind11.h"  // from @pybind11
#include "tensorflow/python/lib/core/pybind11_lib.h"
#include "tensorflow/python/util/nest.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_nest, m) {
  m.doc() = R"pbdoc(
    _pywrap_nest
    -----
  )pbdoc";
  m.def(
      "FlattenDictItems",
      [](const py::handle& dict) {
        return tensorflow::PyoOrThrow(tensorflow::FlattenDictItems(dict.ptr()));
      },
      R"pbdoc(
    Returns a dictionary with flattened keys and values.
  )pbdoc");
}
