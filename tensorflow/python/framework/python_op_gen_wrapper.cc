
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

#include <Python.h>

#include "pybind11/pybind11.h"
#include "tensorflow/python/framework/python_op_gen.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_python_op_gen, m) {
  m.def("GetPythonWrappers", [](py::bytes input) {
    char* c_string;
    Py_ssize_t py_size;
    if (PyBytes_AsStringAndSize(input.ptr(), &c_string, &py_size) == -1) {
      throw py::error_already_set();
    }
    return py::bytes(
        tensorflow::GetPythonWrappers(c_string, static_cast<size_t>(py_size)));
  });
};
