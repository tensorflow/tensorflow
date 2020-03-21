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
#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"
#include "tensorflow/lite/python/optimize/sparsification_wrapper.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;
using tflite::sparsification_wrapper::SparsificationWrapper;

PYBIND11_MODULE(_pywrap_tensorflow_lite_sparsification_wrapper, m) {
  m.doc() = R"pbdoc(
    _pywrap_tensorflow_lite_sparsification_wrapper
    -----
  )pbdoc";
  py::class_<SparsificationWrapper>(m, "SparsificationWrapper")
      .def(py::init([](py::handle& data) {
        return ::SparsificationWrapper::CreateWrapperCPPFromBuffer(data.ptr());
      }))
      .def("SparsifyModel", [](SparsificationWrapper& self) {
        return tensorflow::pyo_or_throw(self.SparsifyModel());
      });
}
