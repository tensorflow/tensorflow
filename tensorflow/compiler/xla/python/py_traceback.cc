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

#include "tensorflow/compiler/xla/python/py_traceback.h"

#include "absl/strings/str_format.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "tensorflow/compiler/xla/python/traceback.h"

namespace xla {

namespace py = pybind11;

void BuildTracebackSubmodule(py::module& m) {
  py::class_<Traceback::Frame>(m, "Frame")
      .def_readonly("file_name", &Traceback::Frame::file_name)
      .def_readonly("function_name", &Traceback::Frame::function_name)
      .def_readonly("function_start_line",
                    &Traceback::Frame::function_start_line)
      .def_readonly("line_num", &Traceback::Frame::line_num)
      .def("__repr__", [](const Traceback::Frame& frame) {
        return absl::StrFormat("%s;%s:%d", frame.function_name, frame.file_name,
                               frame.line_num);
      });

  py::class_<Traceback, std::shared_ptr<Traceback>> traceback(
      m, "Traceback", "Represents a Python stack trace.");
  traceback.def_property_static(
      "enabled", [](py::object /* cls */) { return Traceback::enabled(); },
      [](py::object /* cls */, bool enabled) {
        return Traceback::SetEnabled(enabled);
      });
  traceback.def_static(
      "get_traceback", []() { return Traceback::Get(); },
      R"doc(
    Returns a :class:`Traceback` for the current thread.

    If ``Traceback.enabled`` is ``True``, returns a :class:`Traceback` object
    that describes the Python stack of the calling thread. Stack trace
    collection has a small overhead, so it is disabled by default. If traceback
    collection is disabled, returns ``None``.
    )doc");
  traceback.def_property_readonly("frames", &Traceback::Frames);
  traceback.def("__str__", &Traceback::ToString);
}

}  // namespace xla
