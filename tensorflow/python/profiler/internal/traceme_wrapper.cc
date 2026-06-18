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

#include "xla/python/profiler/internal/traceme_wrapper.h"

#include "pybind11/attr.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "xla/python/profiler/internal/traceme_state.h"

namespace py = ::pybind11;

using ::xla::profiler::TraceMeWrapper;

// Returns true if TraceMe is enabled.
// This is a low-overhead function that can be called frequently.
static PyObject* traceme_enabled(PyObject* self, PyObject* args) {
  if (xla::profiler::traceme_enabled) {
    Py_RETURN_TRUE;
  }
  Py_RETURN_FALSE;
}

static PyMethodDef traceme_method_def = {"traceme_enabled", traceme_enabled,
                                         METH_NOARGS,
                                         "Returns true if TraceMe is enabled."};

PYBIND11_MODULE(_pywrap_traceme, m) {
  py::class_<TraceMeWrapper>(m, "TraceMe", py::module_local())
      .def(py::init<const py::str&, const py::kwargs&>())
      .def("SetMetadata", &TraceMeWrapper::SetMetadata)
      .def("Stop", &TraceMeWrapper::Stop);

  py::object module_name = m.attr("__name__");
  m.attr("traceme_enabled") =
      py::reinterpret_steal<py::object>(PyCFunction_NewEx(
          &traceme_method_def, /*self=*/nullptr, module_name.ptr()));
};
