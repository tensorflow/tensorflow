/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/python/traceback.h"

#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/stl/optional.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/string_view.h"  // IWYU pragma: keep
#include "third_party/nanobind/include/nanobind/stl/vector.h"  // IWYU pragma: keep
#include "xla/pjrt/exceptions.h"
#include "xla/python/nb_class_ptr.h"
#include "xla/python/python_ref_manager.h"
#include "tsl/platform/platform.h"

#ifdef PLATFORM_GOOGLE
#define Py_BUILD_CORE
#include "internal/pycore_frame.h"
#undef Py_BUILD_CORE
#endif  // PLATFORM_GOOGLE

namespace xla {

namespace nb = nanobind;

bool Traceback::enabled_ = true;

Traceback::Traceback() {
  DCHECK(PyGILState_Check());
  PyThreadState* thread_state = PyThreadState_GET();

#if PY_VERSION_HEX < 0x030b0000
  // The representation of frame->f_lasti changed from bytes to words in Python
  // 3.10, see https://docs.python.org/3/whatsnew/3.10.html#changes-in-the-c-api
  // This should match sizeof(_Py_CODEUNIT) which is unfortunately private.
#if PY_VERSION_HEX < 0x030a0000
  constexpr int kLastiWordBytes = 1;
#else   // PY_VERSION_HEX < 0x030a0000
  constexpr int kLastiWordBytes = 2;
#endif  // PY_VERSION_HEX < 0x030a0000

  for (PyFrameObject* py_frame = thread_state->frame; py_frame != nullptr;
       py_frame = py_frame->f_back) {
    Py_INCREF(py_frame->f_code);
    frames_.emplace_back(py_frame->f_code, py_frame->f_lasti * kLastiWordBytes);
  }
#else  // PY_VERSION_HEX < 0x030b0000

#ifdef PLATFORM_GOOGLE
  // This code is equivalent to the version using public APIs, but it saves us
  // an allocation of one object per stack frame. However, this is definitely
  // violating the API contract of CPython, so we only use this where we can be
  // confident we know exactly which CPython we are using (internal to Google).
  // Feel free to turn this on if you like, but it might break at any time!
  for (_PyInterpreterFrame* f = thread_state->cframe->current_frame;
       f != nullptr; f = f->previous) {
    if (_PyFrame_IsIncomplete(f)) continue;
    Py_INCREF(f->f_code);
    frames_.emplace_back(f->f_code,
                         _PyInterpreterFrame_LASTI(f) * sizeof(_Py_CODEUNIT));
  }
#else   // PLATFORM_GOOGLE
  PyFrameObject* next;
  for (PyFrameObject* py_frame = PyThreadState_GetFrame(thread_state);
       py_frame != nullptr; py_frame = next) {
    frames_.emplace_back(PyFrame_GetCode(py_frame), PyFrame_GetLasti(py_frame));
    next = PyFrame_GetBack(py_frame);
    Py_XDECREF(py_frame);
  }
#endif  // PLATFORM_GOOGLE

#endif  // PY_VERSION_HEX < 0x030b0000
}

Traceback::~Traceback() {
  for (auto& frame : frames_) {
    DCHECK(PyGILState_Check());
    Py_DECREF(frame.first);
  }
}

Traceback::Traceback(Traceback&& other) : frames_(std::move(other.frames_)) {
  // absl::InlinedVector does not always clear itself if moved. Since we rely on
  // its empty() method to destroy Traceback differently, we explicitly clear
  // here.
  other.frames_.clear();
}

std::string Traceback::Frame::ToString() const {
  return absl::StrFormat("%s:%d (%s)", nb::cast<std::string_view>(file_name),
                         line_num, nb::cast<std::string_view>(function_name));
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
    frames.push_back(Frame{nb::borrow<nb::str>(frame.first->co_filename),
                           nb::borrow<nb::str>(frame.first->co_name),
                           frame.first->co_firstlineno,
                           PyCode_Addr2Line(frame.first, frame.second)});
  }
  return frames;
}

std::optional<nb_class_ptr<Traceback>> Traceback::Get() {
  DCHECK(PyGILState_Check());
  if (!enabled_) {
    return std::nullopt;
  }
  return make_nb_class<Traceback>();
}

void Traceback::SetEnabled(bool enabled) { enabled_ = enabled; }

nb::object Traceback::AsPythonTraceback() const {
  nb::object traceback = nb::none();
  nb::dict globals;
  nb::handle traceback_type(reinterpret_cast<PyObject*>(&PyTraceBack_Type));
  for (const std::pair<PyCodeObject*, int>& frame : frames_) {
    int lineno = PyCode_Addr2Line(frame.first, frame.second);
    // Under Python 3.11 we observed crashes when using a fake PyFrameObject
    // with a real PyCodeObject (https://github.com/google/jax/issues/16027).
    // because the frame does not have fields necessary to compute the locals,
    // notably the closure object, leading to crashes in CPython in
    // _PyFrame_FastToLocalsWithError
    // https://github.com/python/cpython/blob/deaf509e8fc6e0363bd6f26d52ad42f976ec42f2/Objects/frameobject.c#LL1116C2-L1116C2
    // We therefore always build a fake code object to go along with our fake
    // frame.
    PyCodeObject* py_code =
        PyCode_NewEmpty(PyUnicode_AsUTF8(frame.first->co_filename),
                        PyUnicode_AsUTF8(frame.first->co_name), lineno);
    PyFrameObject* py_frame = PyFrame_New(PyThreadState_Get(), py_code,
                                          globals.ptr(), /*locals=*/nullptr);
    Py_DECREF(py_code);

    traceback = traceback_type(
        /*tb_next=*/std::move(traceback),
        /*tb_frame=*/
        nb::steal<nb::object>(reinterpret_cast<PyObject*>(py_frame)),
        /*tb_lasti=*/0,
        /*tb_lineno=*/
        PyCode_Addr2Line(frame.first, frame.second));
  }
  return traceback;
}

void BuildTracebackSubmodule(nb::module_& m) {
  nb::class_<Traceback::Frame>(m, "Frame")
      .def_ro("file_name", &Traceback::Frame::file_name)
      .def_ro("function_name", &Traceback::Frame::function_name)
      .def_ro("function_start_line", &Traceback::Frame::function_start_line)
      .def_ro("line_num", &Traceback::Frame::line_num)
      .def("__repr__", [](const Traceback::Frame& frame) {
        return absl::StrFormat(
            "%s;%s:%d", nb::cast<std::string_view>(frame.function_name),
            nb::cast<std::string_view>(frame.file_name), frame.line_num);
      });

  nb::class_<Traceback> traceback(m, "Traceback",
                                  "Represents a Python stack trace.");
  traceback.def_prop_rw_static(
      "enabled", [](nb::object /* cls */) { return Traceback::enabled(); },
      [](nb::object /* cls */, bool enabled) {
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
  traceback.def_prop_ro("frames", &Traceback::Frames);
  traceback.def("raw_frames", [](const Traceback& tb) -> nb::tuple {
    // We return a tuple of lists, rather than a list of tuples, because it
    // is cheaper to allocate only three Python objects for everything rather
    // than one per frame.
    nb::list out_code = nb::steal<nb::list>(PyList_New(tb.raw_frames().size()));
    nb::list out_lasti =
        nb::steal<nb::list>(PyList_New(tb.raw_frames().size()));
    for (size_t i = 0; i < tb.raw_frames().size(); ++i) {
      const auto& frame = tb.raw_frames()[i];
      PyObject* code = reinterpret_cast<PyObject*>(frame.first);
      Py_INCREF(code);
      PyList_SET_ITEM(out_code.ptr(), i, code);
      PyList_SET_ITEM(out_lasti.ptr(), i,
                      nb::int_(frame.second).release().ptr());
    }
    return nb::make_tuple(out_code, out_lasti);
  });
  traceback.def("__str__", &Traceback::ToString);
  traceback.def("__eq__",
                [](const Traceback& a, const Traceback& b) { return a == b; });
  traceback.def("__hash__",
                [](const Traceback& tb) { return absl::HashOf(tb); });
  traceback.def("as_python_traceback", &Traceback::AsPythonTraceback);

  traceback.def_static(
      "code_addr2line",
      [](nb::handle code, int lasti) {
        if (!PyCode_Check(code.ptr())) {
          throw xla::XlaRuntimeError("code argument must be a code object");
        }
        return PyCode_Addr2Line(reinterpret_cast<PyCodeObject*>(code.ptr()),
                                lasti);
      },
      "Python wrapper around the Python C API function PyCode_Addr2Line");

#if PY_VERSION_HEX >= 0x030b0000
  traceback.def_static(
      "code_addr2location",
      [](nb::handle code, int lasti) {
        if (!PyCode_Check(code.ptr())) {
          throw xla::XlaRuntimeError("code argument must be a code object");
        }
        int start_line, start_column, end_line, end_column;
        if (!PyCode_Addr2Location(reinterpret_cast<PyCodeObject*>(code.ptr()),
                                  lasti, &start_line, &start_column, &end_line,
                                  &end_column)) {
          throw nb::python_error();
        }
        return nb::make_tuple(start_line, start_column, end_line, end_column);
      },
      "Python wrapper around the Python C API function PyCode_Addr2Location");
#endif  // PY_VERSION_HEX >= 0x030b0000

#if PY_VERSION_HEX < 0x030b0000
  // This function replaces the exception traceback associated with the current
  // Python thread.
  m.def(
      "replace_thread_exc_traceback",
      [](nb::object tb) {
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
      nb::arg("traceback").none());
#endif  // PY_VERSION_HEX < 0x30b0000
}
}  // namespace xla
