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

#include "absl/types/span.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/python/framework/python_api_info.h"
#include "tensorflow/python/framework/python_api_parameter_converter.h"
#include "tensorflow/python/framework/python_tensor_converter.h"

namespace py = pybind11;

namespace tensorflow {
namespace {

PythonAPIInfo::InferredAttributes Convert(
    const PythonAPIInfo& api_info,
    const PythonTensorConverter& tensor_converter, py::handle arg_list) {
  PythonAPIInfo::InferredAttributes inferred_attrs;

  Py_ssize_t size = PySequence_Size(arg_list.ptr());
  if (size < 0) {
    throw py::error_already_set();
  }

  std::vector<PyObject*> args_raw_vec(size);
  for (Py_ssize_t i = 0; i < size; ++i) {
    PyObject* item = PySequence_GetItem(arg_list.ptr(), i);
    if (!item) {
      for (Py_ssize_t j = 0; j < i; ++j) {
        Py_XDECREF(args_raw_vec[j]);
      }
      throw py::error_already_set();
    }
    args_raw_vec[i] = item;
  }

  absl::Span<PyObject*> args_raw(args_raw_vec.data(), args_raw_vec.size());

  int max_index = GetPythonAPIMaxIndex(api_info);
  if (static_cast<int>(args_raw.size()) <= max_index) {
    for (PyObject* item : args_raw_vec) {
      Py_XDECREF(item);
    }
    PyErr_SetString(PyExc_ValueError,
                    "Parameters span size is smaller than expected");
    throw py::error_already_set();
  }

  if (!CopyPythonAPITensorLists(api_info, args_raw)) {
    for (PyObject* item : args_raw_vec) {
      Py_XDECREF(item);
    }
    throw py::error_already_set();
  }
  if (!ConvertPythonAPIParameters(api_info, tensor_converter, args_raw,
                                  &inferred_attrs)) {
    for (PyObject* item : args_raw_vec) {
      Py_XDECREF(item);
    }
    throw py::error_already_set();
  }

  if (PyList_Check(arg_list.ptr())) {
    for (Py_ssize_t i = 0; i < size; ++i) {
      PyObject* new_item = args_raw_vec[i];
      PyList_SET_ITEM(arg_list.ptr(), i, new_item);
    }
  } else {
    for (Py_ssize_t i = 0; i < size; ++i) {
      Py_DECREF(args_raw_vec[i]);
    }
  }

  return inferred_attrs;
}

}  // namespace
}  // namespace tensorflow

PYBIND11_MODULE(_pywrap_python_api_parameter_converter, m) {
  m.def("Convert", tensorflow::Convert);
}
