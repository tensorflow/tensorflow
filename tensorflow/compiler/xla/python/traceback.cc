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
#include <string>
#include <utility>

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "pybind11/pytypes.h"
#include "tensorflow/compiler/xla/python/exceptions.h"
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

py::object Traceback::AsPythonTraceback() const {
  py::object traceback = py::none();
  py::dict globals;
  py::handle traceback_type(reinterpret_cast<PyObject*>(&PyTraceBack_Type));
  for (const std::pair<PyCodeObject*, int>& frame : frames_) {
    PyFrameObject* py_frame = PyFrame_New(PyThreadState_Get(), frame.first,
                                          globals.ptr(), /*locals=*/nullptr);

    traceback = traceback_type(
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
  traceback.def("raw_frames", [](const Traceback& tb) -> py::tuple {
    // We return a tuple of lists, rather than a list of tuples, because it
    // is cheaper to allocate only three Python objects for everything rather
    // than one per frame.
    py::list out_code(tb.raw_frames().size());
    py::list out_lasti(tb.raw_frames().size());
    for (size_t i = 0; i < tb.raw_frames().size(); ++i) {
      const auto& frame = tb.raw_frames()[i];
      out_code[i] = py::reinterpret_borrow<py::object>(
          reinterpret_cast<PyObject*>(frame.first));
      out_lasti[i] = py::int_(frame.second);
    }
    return py::make_tuple(out_code, out_lasti);
  });
  traceback.def("__str__", &Traceback::ToString);
  traceback.def("__eq__",
                [](const Traceback& a, const Traceback& b) { return a == b; });
  traceback.def("__hash__",
                [](const Traceback& tb) { return absl::HashOf(tb); });
  traceback.def("as_python_traceback", &Traceback::AsPythonTraceback);

  traceback.def_static(
      "code_addr2line",
      [](py::handle code, int lasti) {
        if (!PyCode_Check(code.ptr())) {
          throw xla::XlaRuntimeError("code argument must be a code object");
        }
        return PyCode_Addr2Line(reinterpret_cast<PyCodeObject*>(code.ptr()),
                                lasti);
      },
      "Python wrapper around the Python C API function PyCode_Addr2Line");

  // This function replaces the exception traceback associated with the current
  // Python thread.
  m.def(
      "replace_thread_exc_traceback",
      [](py::object tb) {
        if (!tb.is_none() && !PyTraceBack_Check(tb.ptr())) {
          throw xla::XlaRuntimeError(
              "argument must be a traceback object or None");
        }
        PyThreadState* thread_state = PyThreadState_Get();
        if (!thread_state->exc_info->exc_traceback) {
          throw xla::XlaRuntimeError(
              "Current thread does not have an active "
              "exception traceback");
        }
        PyObject* old_exc_traceback = thread_state->exc_info->exc_traceback;
        PyObject* new_tb = tb.is_none() ? nullptr : tb.release().ptr();
        thread_state->exc_info->exc_traceback = new_tb;
        Py_XDECREF(old_exc_traceback);
      },
      py::arg("traceback"));
}

}  // namespace xla
