/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/python/interpreter_wrapper/python_utils.h"

#include <cstddef>

namespace mlirlite {
namespace python_utils {

int ConvertFromPyString(PyObject* obj, char** data, Py_ssize_t* length) {
#if PY_MAJOR_VERSION >= 3
  if (PyUnicode_Check(obj)) {
    // const_cast<> is for CPython 3.7 finally adding const to the API.
    *data = const_cast<char*>(PyUnicode_AsUTF8AndSize(obj, length));
    return *data == nullptr ? -1 : 0;
  }
  return PyBytes_AsStringAndSize(obj, data, length);
#else
  return PyString_AsStringAndSize(obj, data, length);
#endif
}

PyObject* ConvertToPyString(const char* data, size_t length) {
#if PY_MAJOR_VERSION >= 3
  return PyBytes_FromStringAndSize(data, length);
#else
  return PyString_FromStringAndSize(data, length);
#endif
}

}  // namespace python_utils
}  // namespace mlirlite
