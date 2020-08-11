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
#include "pybind11/pybind11.h"
#include "tensorflow/python/framework/op_def_util.h"

namespace py = pybind11;

namespace {

py::handle ConvertAttr(py::handle value, std::string attr_type) {
  tensorflow::Safe_PyObjectPtr result =
      ::tensorflow::ConvertPyObjectToAttributeType(
          value.ptr(), ::tensorflow::AttributeTypeFromName(attr_type));
  if (!result) {
    throw py::error_already_set();
  }
  Py_INCREF(result.get());
  return result.release();
}

}  // namespace

// Expose ConvertPyObjectToAttributeType via Python.  Note: this is done to
// simplify testing; ConvertPyObjectToAttributeType is expected to be called
// directly from c++.
PYBIND11_MODULE(_op_def_util, m) {
  m.def("ConvertPyObjectToAttributeType", ConvertAttr, py::arg("value"),
        py::arg("attr_type_enum"));
}
