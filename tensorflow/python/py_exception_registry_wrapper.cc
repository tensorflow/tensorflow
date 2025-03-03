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

#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "tensorflow/c/tf_status.h"
#include "tensorflow/python/lib/core/py_exception_registry.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_py_exception_registry, m) {
  py::enum_<TF_Code>(m, "TF_Code", py::module_local())
      .value("TF_OK", TF_OK)
      .value("TF_CANCELLED", TF_CANCELLED)
      .value("TF_UNKNOWN", TF_UNKNOWN)
      .value("TF_INVALID_ARGUMENT", TF_INVALID_ARGUMENT)
      .value("TF_DEADLINE_EXCEEDED", TF_DEADLINE_EXCEEDED)
      .value("TF_PERMISSION_DENIED", TF_PERMISSION_DENIED)
      .value("TF_UNAUTHENTICATED", TF_UNAUTHENTICATED)
      .value("TF_RESOURCE_EXHAUSTED", TF_RESOURCE_EXHAUSTED)
      .value("TF_FAILED_PRECONDITION", TF_FAILED_PRECONDITION)
      .value("TF_ABORTED", TF_ABORTED)
      .value("TF_OUT_OF_RANGE", TF_OUT_OF_RANGE)
      .value("TF_UNIMPLEMENTED", TF_UNIMPLEMENTED)
      .value("TF_INTERNAL", TF_INTERNAL)
      .value("TF_DATA_LOSS", TF_DATA_LOSS)
      .export_values();

  m.def("PyExceptionRegistry_Init", [](py::object& code_to_exc_type_map) {
    tensorflow::PyExceptionRegistry::Init(code_to_exc_type_map.ptr());
  });
  m.def("PyExceptionRegistry_Lookup",
        [](TF_Code code) { tensorflow::PyExceptionRegistry::Lookup(code); });
};
