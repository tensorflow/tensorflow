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

#include <array>

#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_py_exception_registry, m) {
  m.def("PyExceptionRegistry_Init", [](py::object& code_to_exc_type_map) {
    tensorflow::PyExceptionRegistry::Init(code_to_exc_type_map.ptr());
  });
  m.def("PyExceptionRegistry_Lookup",
        [](TF_Code code) { tensorflow::PyExceptionRegistry::Lookup(code); });
};
