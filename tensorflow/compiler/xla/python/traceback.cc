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

#include "tensorflow/compiler/xla/python/traceback.h"

#include <stdexcept>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/python_ref_manager.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

namespace py = pybind11;

bool Traceback::enabled_ = true;

Traceback::~Traceback() {
  // We want Traceback objects to be safe to destroy without holding the
  // GIL, so we defer destruction of the strings.
  GlobalPyRefManager()->AddGarbage(frames_);
}

std::string Traceback::Frame::ToString() const {
  return absl::StrFormat("%s:%d (%s)", file_name, line_num, function_name);
}

std::string Traceback::ToString() const {
  std::vector<std::string> frame_strs;
  frame_strs.reserve(frames_.size());
  for (const Frame& frame : Frames()) {
    frame_strs.push_back(frame.ToString());
  }
  return absl::StrJoin(frame_strs, "\n");
}

std::vector<Traceback::Frame> Traceback::Frames() const {
  // We require the GIL because we manipulate Python strings.
  CHECK(PyGILState_Check());
  std::vector<Traceback::Frame> frames;
  frames.reserve(frames_.size());
  for (const auto& frame : frames_) {
    frames.push_back(Frame{
        std::string(py::reinterpret_borrow<py::str>(frame.first->co_filename)),
        std::string(py::reinterpret_borrow<py::str>(frame.first->co_name)),
        frame.first->co_firstlineno,
        PyCode_Addr2Line(frame.first, frame.second)});
  }
  return frames;
}

std::shared_ptr<Traceback> Traceback::Get() {
  DCHECK(PyGILState_Check());
  if (!enabled_) {
    return nullptr;
  }
  auto tb = std::make_shared<Traceback>();
  const PyThreadState* thread_state = PyThreadState_GET();
  for (PyFrameObject* py_frame = thread_state->frame; py_frame != nullptr;
       py_frame = py_frame->f_back) {
    Py_INCREF(py_frame->f_code);
    tb->frames_.emplace_back(py_frame->f_code, py_frame->f_lasti);
  }
  return tb;
}

void Traceback::SetEnabled(bool enabled) { enabled_ = enabled; }

#if PY_VERSION_HEX < 0x03070000

// Traceback objects cannot be constructed from the type in Python 3.6.
static py::object MakePythonTraceback(py::object tb_next, py::object tb_frame,
                                      int tb_lasti, int tb_lineno) {
  PyTracebackObject* tb;
  if (tb_next.ptr() && tb_next != Py_None &&
      !PyTraceBack_Check(tb_next.ptr())) {
    throw std::runtime_error("tb_next argument must be a traceback");
  }
  if (!tb_frame.ptr() || !PyFrame_Check(tb_frame.ptr())) {
    throw std::runtime_error("tb_frame argument must be a frame");
  }
  tb = PyObject_GC_New(PyTracebackObject, &PyTraceBack_Type);
  if (tb) {
    tb->tb_next =
        tb_next == Py_None
            ? nullptr
            : reinterpret_cast<PyTracebackObject*>(tb_next.release().ptr());
    tb->tb_frame = reinterpret_cast<PyFrameObject*>(tb_frame.release().ptr());
    tb->tb_lasti = tb_lasti;
    tb->tb_lineno = tb_lineno;
    PyObject_GC_Track(tb);
  }
  return py::reinterpret_steal<py::object>(reinterpret_cast<PyObject*>(tb));
}
#else

static py::object MakePythonTraceback(py::object tb_next, py::object tb_frame,
                                      int tb_lasti, int tb_lineno) {
  py::handle traceback_type(reinterpret_cast<PyObject*>(&PyTraceBack_Type));
  return traceback_type(tb_next, tb_frame, tb_lasti, tb_lineno);
}

#endif  // PY_VERSION_HEX < 0x3070000

py::object Traceback::AsPythonTraceback() const {
  py::object traceback = py::none();
  py::dict globals;
  for (const std::pair<PyCodeObject*, int>& frame : frames_) {
    PyFrameObject* py_frame = PyFrame_New(PyThreadState_Get(), frame.first,
                                          globals.ptr(), /*locals=*/nullptr);

    traceback = MakePythonTraceback(
        /*tb_next=*/std::move(traceback),
        /*tb_frame=*/
        py::reinterpret_steal<py::object>(
            reinterpret_cast<PyObject*>(py_frame)),
        /*tb_lasti=*/frame.second,
        /*tb_lineno=*/PyCode_Addr2Line(frame.first, frame.second));
  }
  return traceback;
}

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
  traceback.def("as_python_traceback", &Traceback::AsPythonTraceback);

#if PY_VERSION_HEX < 0x03070000
  m.def("make_python_traceback", &MakePythonTraceback, py::arg("tb_next"),
        py::arg("tb_frame"), py::arg("tb_lasti"), py::arg("tb_lineno"));
#endif  // PY_VERSION_HEX < 0x3070000

  // This function replaces the exception traceback associated with the current
  // Python thread.
#if PY_VERSION_HEX < 0x03070000
  m.def(
      "replace_thread_exc_traceback",
      [](py::object tb) {
        if (!PyTraceBack_Check(tb.ptr())) {
          throw std::runtime_error("argument must be a traceback object");
        }
        PyThreadState* thread_state = PyThreadState_Get();
        if (!thread_state->exc_traceback) {
          throw std::runtime_error(
              "Current thread does not have an active "
              "exception traceback");
        }
        PyObject* old_exc_traceback = thread_state->exc_traceback;
        thread_state->exc_traceback = tb.release().ptr();
        Py_XDECREF(old_exc_traceback);
      },
      py::arg("traceback"));
#else   // PY_VERSION_HEX < 0x3070000
  m.def(
      "replace_thread_exc_traceback",
      [](py::object tb) {
        if (!PyTraceBack_Check(tb.ptr())) {
          throw std::runtime_error("argument must be a traceback object");
        }
        PyThreadState* thread_state = PyThreadState_Get();
        if (!thread_state->exc_info->exc_traceback) {
          throw std::runtime_error(
              "Current thread does not have an active "
              "exception traceback");
        }
        PyObject* old_exc_traceback = thread_state->exc_info->exc_traceback;
        thread_state->exc_info->exc_traceback = tb.release().ptr();
        Py_XDECREF(old_exc_traceback);
      },
      py::arg("traceback"));
#endif  // PY_VERSION_HEX < 0x3070000
}

}  // namespace xla
