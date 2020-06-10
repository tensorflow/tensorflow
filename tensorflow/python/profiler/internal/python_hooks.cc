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
#include "tensorflow/python/profiler/internal/python_hooks.h"

#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/platform/path.h"

namespace tensorflow {
namespace profiler {

namespace py = ::pybind11;

template <typename T>
int ProfileFunction(PyObject* obj, PyFrameObject* frame, int what,
                    PyObject* arg) {
  T::GetSingleton()->ProfileFast(frame, what, arg);
  return 0;
}

void SysSetProfileNone() {
  py::object setprofile = py::module::import("sys").attr("setprofile");
  setprofile(py::none());
}

void ThreadingSetProfile(const py::object& callback) {
  py::object setprofile = py::module::import("threading").attr("setprofile");
  setprofile(callback);
}

PythonHooks* PythonHooks::GetSingleton() {
  static PythonHooks* singleton = new PythonHooks;
  return singleton;
}

void PythonHooks::Start(const PythonHooksOptions& option) {
  if (option.enable_python_traceme || option.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (option.enable_trace_python_function) {
      SetProfilerInAllThreads();
    }
    if (option.enable_python_traceme) {
      EnableTraceMe(true);
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHooks::Stop(const PythonHooksOptions& option) {
  if (option.enable_python_traceme || option.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (option.enable_trace_python_function) {
      ClearProfilerInAllThreads();
    }
    if (option.enable_python_traceme) {
      EnableTraceMe(false);
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHooks::Finalize() { tracemes_.clear(); }

void PythonHooks::ProfileSlow(const py::object& frame, const string& event,
                              const py::object& arg) {
  int what;
  absl::string_view event_name(event);

  if (absl::ConsumePrefix(&event_name, "c_")) {
    if (event_name == "call") {
      what = PyTrace_C_CALL;
    } else if (event_name == "return") {
      what = PyTrace_C_RETURN;
    } else if (event_name == "exception") {
      what = PyTrace_C_EXCEPTION;
    } else {
      return;
    }
  } else {
    if (event_name == "call") {
      what = PyTrace_CALL;
    } else if (event_name == "return") {
      what = PyTrace_RETURN;
    } else if (event_name == "exception") {
      what = PyTrace_EXCEPTION;
    } else {
      return;
    }
  }

  ProfileFast(reinterpret_cast<PyFrameObject*>(frame.ptr()), what, arg.ptr());
}

void PythonHooks::ProfileFast(PyFrameObject* frame, int what, PyObject* arg) {
  const int64 thread_id = PyThread_get_thread_ident();

  if (what == PyTrace_CALL) {
    PyCodeObject* f_code = frame->f_code;
    string filename(py::reinterpret_borrow<py::str>(f_code->co_filename));
    int line_no = frame->f_lineno;

    string function;
    if (f_code->co_name == nullptr) {
      function = "<unknown>";
    } else {
      function = py::reinterpret_borrow<py::str>(f_code->co_name);
    }

    tracemes_[thread_id].push_back(absl::make_unique<TraceMe>(absl::StrCat(
        "$", io::Basename(filename), ":", line_no, " ", function)));
  } else if (what == PyTrace_C_CALL && PyCFunction_Check(arg)) {
    // Python stack does not have a filename/line_no for native calls.
    auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
    PyObject* module = func->m_module;
    string filename;
    bool filename_ok;
#if PY_MAJOR_VERSION < 3
    filename_ok = (module != nullptr && PyString_Check(module));
#else
    filename_ok = (module != nullptr && PyUnicode_Check(module));
#endif
    if (filename_ok) {
      filename = py::reinterpret_borrow<py::str>(module);
    } else {
      filename = "<unknown>";
    }

    string function(func->m_ml->ml_name);
    tracemes_[thread_id].push_back(absl::make_unique<TraceMe>(
        absl::StrCat(filename, " ", func->m_ml->ml_name)));
  } else if (what == PyTrace_RETURN || what == PyTrace_C_RETURN ||
             what == PyTrace_EXCEPTION || what == PyTrace_C_EXCEPTION) {
    auto& thread_tracemes = tracemes_[thread_id];
    if (!thread_tracemes.empty()) {
      thread_tracemes.pop_back();
    }
  }
}

void PythonHooks::SetProfilerInAllThreads() {
  // We also want any new threads started to use our profiler.
  // NOTE: threading does not provide a C API equivalent to
  // `threading.setprofile` so we are forced to go via Python to setup the
  // profile when a new thread is created. After the first callback in that
  // thread we unregister the Python profile function and use
  // `PyEval_SetProfile` to register a C profiler which has significantly less
  // overhead (>2x faster).
  py::cpp_function callback =
      py::cpp_function([this](const py::object& frame, const string& event,
                              const py::object& arg) {
        ProfileSlow(frame, event, arg);
        SysSetProfileNone();
        PyEval_SetProfile(ProfileFunction<PythonHooks>, nullptr);
      });

  ThreadingSetProfile(callback);

  // NOTE: This must be after `threading.setprofile` otherwise we
  // end up recording that in our trace.
  PyThreadState* curr_thread = PyThreadState_Get();
  PyThreadState* next_thread = curr_thread;
  while (next_thread != nullptr) {
    VLOG(1) << "Setting profiler in " << next_thread->thread_id;
    PyThreadState_Swap(next_thread);
    PyEval_SetProfile(ProfileFunction<PythonHooks>, nullptr);
    next_thread = next_thread->next;
  }
  PyThreadState_Swap(curr_thread);
}

void PythonHooks::ClearProfilerInAllThreads() {
  PyThreadState* curr_thread = PyThreadState_Get();
  PyThreadState* next_thread = curr_thread;
  while (next_thread != nullptr) {
    VLOG(1) << "Clearing profiler in " << next_thread->thread_id;
    PyThreadState_Swap(next_thread);
    PyEval_SetProfile(nullptr, nullptr);
    next_thread = next_thread->next;
  }
  PyThreadState_Swap(curr_thread);

  // And notify the threading library that we're done.
  ThreadingSetProfile(py::none());
}

void PythonHooks::EnableTraceMe(bool enable) {
  const char* kModuleName =
      "tensorflow.python.profiler.internal._pywrap_traceme";
  auto trace_module = py::module::import(kModuleName);
  trace_module.attr("enabled") = enable;
}

}  // namespace profiler
}  // namespace tensorflow
