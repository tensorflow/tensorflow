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

#include "include/pybind11/pybind11.h"
#include "include/pybind11/pytypes.h"

#ifndef TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_LIB_H_
#define TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_LIB_H_

namespace py = pybind11;

// SWIG struct so pybind11 can handle SWIG objects returned by tf_session
// until that is converted over to pybind11.
// This type is intended to be layout-compatible with an initial sequence of
// certain objects pointed to by a PyObject pointer. The intended use is to
// first check dynamically that a given PyObject* py has the correct type,
// and then use `reinterpret_cast<SwigPyObject*>(py)` to retrieve the member
// `ptr` for further, custom use. SWIG wrapped objects' layout is documented
// here: http://www.swig.org/Doc4.0/Python.html#Python_nn28
typedef struct {
  PyObject_HEAD void* ptr;  // This is the pointer to the actual C++ obj.
  void* ty;
  int own;
  PyObject* next;
  PyObject* dict;
} SwigPyObject;

namespace tensorflow {

// Convert PyObject* to py::object with no error handling.

inline py::object pyo(PyObject* ptr) {
  return py::reinterpret_steal<py::object>(ptr);
}

// Raise an exception if the PyErrOcurred flag is set or else return the Python
// object.

inline py::object pyo_or_throw(PyObject* ptr) {
  if (PyErr_Occurred() || ptr == nullptr) {
    throw py::error_already_set();
  }
  return pyo(ptr);
}

void throwTypeError(const char* error_message) {
  PyErr_SetString(PyExc_TypeError, error_message);
  throw pybind11::error_already_set();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_LIB_CORE_PYBIND11_LIB_H_
