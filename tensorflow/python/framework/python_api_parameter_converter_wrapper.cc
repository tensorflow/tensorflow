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
// Note: This library is only used by python_api_parameter_converter_test.  It
// is not meant to be used in other circumstances.

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/python/framework/python_api_parameter_converter.h"

namespace py = pybind11;

namespace tensorflow {
namespace {

PythonAPIInfo::InferredAttributes Convert(
    const PythonAPIInfo& api_info,
    const PythonTensorConverter& tensor_converter, py::handle arg_list) {
  PythonAPIInfo::InferredAttributes inferred_attrs;

  PyObject* args_fast = PySequence_Fast(arg_list.ptr(), "Expected a list");
  absl::Span<PyObject*> args_raw(PySequence_Fast_ITEMS(args_fast),
                                 PySequence_Fast_GET_SIZE(args_fast));

  if (!CopyPythonAPITensorLists(api_info, args_raw)) {
    throw py::error_already_set();
  }
  if (!ConvertPythonAPIParameters(api_info, tensor_converter, args_raw,
                                  &inferred_attrs)) {
    throw py::error_already_set();
  }

  return inferred_attrs;
}

}  // namespace
}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_python_api_parameter_converter, m) {
  m.def("Convert", tensorflow::Convert);
}
