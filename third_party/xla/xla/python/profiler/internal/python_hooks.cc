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
#include "xla/python/profiler/internal/python_hooks.h"

#include <atomic>
#include <cstdint>
#include <string>

#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
namespace profiler {

namespace py = ::pybind11;

namespace {

void SysSetProfileNone() {
  py::object setprofile = py::module::import("sys").attr("setprofile");
  setprofile(py::none());
}

void ThreadingSetProfile(const py::object& callback) {
  py::object setprofile = py::module::import("threading").attr("setprofile");
  setprofile(callback);
}

std::string GetEventName(PyObject* co_filename, PyObject* co_name,
                         int co_firstlineno) {
  std::string filename(py::reinterpret_borrow<py::str>(co_filename));
  std::string function;
  if (co_name == nullptr) {
    function = "<unknown>";
  } else {
    function = py::reinterpret_borrow<py::str>(co_name);
  }

  return absl::StrCat("$", tsl::io::Basename(filename), ":", co_firstlineno,
                      " ", function);
}

std::string GetEventName(absl::string_view method_name, PyObject* module) {
  // Python stack does not have a filename/line_no for native calls.
  // Use module name and function/method name instead.
  std::string filename;
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
  if (!method_name.empty()) {
    return absl::StrCat("$", filename, " ", method_name);
  }
  return "$<unknown>";
}

void AddEventToXLine(const PythonTraceEntry& event,
                     tsl::profiler::XLineBuilder* line,
                     tsl::profiler::XPlaneBuilder* plane) {
  // TODO(jiesun): maybe add full filename as event stats.
  auto xevent = line->AddEvent(*plane->GetOrCreateEventMetadata(event.Name()));
  xevent.SetTimestampNs(event.start_time_ns);
  xevent.SetEndTimestampNs(event.end_time_ns);
}

#if PY_VERSION_HEX < 0x030C0000
template <typename ForEachThreadFunc>
void ForEachThread(PyThreadState* curr_thread, ForEachThreadFunc&& callback) {
  // Note: PyThreadState's interp is not accessible in open source due to
  // Py_LIMITED_API definition nuances. We can not iterate all threads through
  // that PyInterpreterState.
#ifndef NDEBUG
  // If debug version of python runtime is used, (e.g. monolithic binaries in
  // g3). PyGILState_Check will fail because current thread's PyThreadState
  // is not the one that holding GIL (after PyThreadState_Swap). This extra
  // check in PyEval_SetProfile is not useful, but will sporadic crash if user
  // use debug version of python runtime. In this case, we fallback to only
  // set up profile hooks in current threads.
  // In OSS, the python runtime and tensorflow profiler are built separately.
  // So this workaround doesn't apply.
  callback(curr_thread);
#else
  for (PyThreadState* p = curr_thread; p != nullptr; p = p->next) {
    PyThreadState_Swap(p);
    callback(p);
  }
  for (PyThreadState* p = curr_thread->prev; p != nullptr; p = p->prev) {
    PyThreadState_Swap(p);
    callback(p);
  }
#endif
}

#endif  // PY_VERSION_HEX

}  // namespace

/*static*/ PythonHookContext* PythonHooks::e2e_context_ = nullptr;

std::string PythonTraceEntry::Name() const {
  if (co_filename) {
    return GetEventName(co_filename, co_name, co_firstlineno);
  }
  return GetEventName(method_name, m_module);
}

PythonHooks* PythonHooks::GetSingleton() {
  static PythonHooks* singleton = new PythonHooks;
  return singleton;
}

void PythonHookContext::Start(const PythonHooksOptions& options) {
  if (!Py_IsInitialized()) return;

#if PY_MAJOR_VERSION < 3 || (PY_MAJOR_VERSION == 3 && PY_MINOR_VERSION < 7)
  // Before Python 3.7, the GIL is created on demand by PyEval_InitThreads().
  // When a thread was not started by Python (e.g., when starting profiling via
  // RPC) there might be no GIL. Before Python 3.6, PyGILState_Ensure would
  // crash. The crash was fixed in Python 3.6 but the fix introduced a race for
  // GIL creation. Calling PyEval_InitThreads() prevents the race. This is a
  // no-op when called for a second time so it is innocuous. See
  // https://vstinner.github.io/python37-gil-change.html for details.
  PyEval_InitThreads();
#endif

  options_ = options;
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  if (options_.enable_python_traceme || options_.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_python_traceme) {
      EnableTraceMe(true);
    }
    if (options_.enable_trace_python_function) {
      SetProfilerInAllThreads();
    }
    if (options_.end_to_end_mode) {
      // When end to end mode is used, Stop() and Finalize() i.e. symbolization
      // and data collection happens during C's atexit(), when Py_FinalizeEx()
      // already called.
      try {
        auto atexit = py::module::import("atexit");
        atexit.attr("register")(py::cpp_function([]() {
          PythonHooks* singleton = PythonHooks::GetSingleton();
          auto e2e_context = singleton->Stop();
          // Serialize into internal storage before the tracked PyCodeObjects
          // went out of scope.
          if (e2e_context) {
            e2e_context->CollectData(nullptr);
            PythonHooks::set_e2e_context(e2e_context.release());
          }
        }));
      } catch (const py::error_already_set& e) {
        LOG(ERROR) << "Can't install atexit handler for e2e mode." << e.what();
      }
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHookContext::Stop() {
  if (!Py_IsInitialized()) return;
  if (options_.enable_python_traceme || options_.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_trace_python_function) {
      ClearProfilerInAllThreads();
    }
    if (options_.enable_python_traceme) {
      EnableTraceMe(false);
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHookContext::CollectData(tensorflow::profiler::XPlane* raw_plane) {
  if (raw_plane == nullptr) {
    end_to_end_xplane_.emplace();
    raw_plane = &*end_to_end_xplane_;
  }
  tsl::profiler::XPlaneBuilder plane(raw_plane);
  for (auto& it : entries_) {
    int64_t thread_id = it.first;
    auto& thread_events = it.second;
    VLOG(1) << "Collecting " << thread_events.completed.size() << ":"
            << thread_events.active.size() << " events on thread " << thread_id;
    auto line = plane.GetOrCreateLine(thread_id);
    line.SetTimestampNs(start_timestamp_ns_);
    for (const auto& event : thread_events.completed) {
      AddEventToXLine(event, &line, &plane);
    }
    if (options_.include_incomplete_events) {
      uint64_t now = tsl::profiler::GetCurrentTimeNanos();
      while (!thread_events.active.empty()) {
        auto& event = thread_events.active.top();
        event.end_time_ns = now;
        AddEventToXLine(event, &line, &plane);
        thread_events.active.pop();
      }
    }
  }
  PyGILState_STATE gil_state = PyGILState_Ensure();
  entries_.clear();
  PyGILState_Release(gil_state);
}

void PythonHookContext::Finalize(tensorflow::profiler::XSpace* space) {
  if (space && options_.enable_trace_python_function) {
    tensorflow::profiler::XPlane* plane =
        tsl::profiler::FindOrAddMutablePlaneWithName(
            space, tsl::profiler::kPythonTracerPlaneName);
    if (options_.end_to_end_mode) {
      if (end_to_end_xplane_) {
        end_to_end_xplane_->set_name(plane->name());
        plane->Swap(&*end_to_end_xplane_);
        end_to_end_xplane_.reset();
      }
    } else {
      PyGILState_STATE gil_state = PyGILState_Ensure();
      CollectData(plane);
      PyGILState_Release(gil_state);
    }
  }
}

/*static*/ int PythonHooks::ProfileFunction(PyObject* obj, PyFrameObject* frame,
                                            int what, PyObject* arg) {
  GetSingleton()->ProfileFast(frame, what, arg);
  return 0;
}

void PythonHooks::ProfileSlow(const py::object& frame, const std::string& event,
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

void PythonHookContext::ProfileFast(PyFrameObject* frame, int what,
                                    PyObject* arg) {
  const int64_t thread_id = tsl::Env::Default()->GetCurrentThreadId();
  uint64_t now = tsl::profiler::GetCurrentTimeNanos();
  auto& thread_traces = entries_[thread_id];

  switch (what) {
    case PyTrace_CALL: {
#if PY_VERSION_HEX < 0x030b0000
      PyCodeObject* f_code = frame->f_code;
      thread_traces.active.emplace(now, 0, f_code);
#else   // PY_VERSION_HEX < 0x030b0000
      PyCodeObject* f_code = PyFrame_GetCode(frame);
      thread_traces.active.emplace(now, 0, f_code);
      Py_XDECREF(f_code);
#endif  // PY_VERSION_HEX < 0x030b0000
      break;
    }
    case PyTrace_RETURN:
    case PyTrace_EXCEPTION: {
      if (!thread_traces.active.empty()) {
        auto& entry = thread_traces.active.top();
        entry.end_time_ns = now;
        thread_traces.completed.emplace_back(std::move(entry));
        thread_traces.active.pop();
      } else if (options_.include_incomplete_events) {
#if PY_VERSION_HEX < 0x030b0000
        PyCodeObject* f_code = frame->f_code;
        thread_traces.completed.emplace_back(start_timestamp_ns_, now, f_code);
#else   // PY_VERSION_HEX < 0x030b0000
        PyCodeObject* f_code = PyFrame_GetCode(frame);
        thread_traces.completed.emplace_back(start_timestamp_ns_, now, f_code);
        Py_XDECREF(f_code);
#endif  // PY_VERSION_HEX < 0x030b0000
      }
      break;
    }
    case PyTrace_C_CALL: {
      if (PyCFunction_Check(arg)) {
        // Python stack does not have a filename/line_no for native calls.
        auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
        entries_[thread_id].active.emplace(now, 0, func);
      }
      break;
    }
    case PyTrace_C_RETURN:
    case PyTrace_C_EXCEPTION: {
      if (PyCFunction_Check(arg)) {
        if (!thread_traces.active.empty()) {
          auto& entry = thread_traces.active.top();
          entry.end_time_ns = now;
          thread_traces.completed.emplace_back(std::move(entry));
          thread_traces.active.pop();
        } else if (options_.include_incomplete_events) {
          // Only the end of the events is recorded, use profiler start as
          // start timestamp of the new event.
          auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
          entries_[thread_id].completed.emplace_back(start_timestamp_ns_, now,
                                                     func);
        }
      }
      break;
    }
    default:
      break;
  }
}

/*static*/ void PythonHookContext::SetProfilerInAllThreads() {
  // We also want any new threads started to use our profiler.
  // NOTE: threading does not provide a C API equivalent to
  // `threading.setprofile` so we are forced to go via Python to setup the
  // profile when a new thread is created. After the first callback in that
  // thread we unregister the Python profile function and use
  // `PyEval_SetProfile` to register a C profiler which has significantly less
  // overhead (>2x faster).
  PythonHooks* singleton = PythonHooks::GetSingleton();
  py::cpp_function callback = py::cpp_function(
      [singleton](const py::object& frame, const std::string& event,
                  const py::object& arg) {
        singleton->ProfileSlow(frame, event, arg);
        SysSetProfileNone();
        PyEval_SetProfile(&PythonHooks::ProfileFunction, nullptr);
      });

  ThreadingSetProfile(callback);

  // NOTE: This must be after `threading.setprofile` otherwise we
  // end up recording that in our trace.
#if PY_VERSION_HEX < 0x030C0000
  PyThreadState* curr_thread = PyThreadState_Get();
  ForEachThread(curr_thread, [](PyThreadState* thread) {
    VLOG(1) << "Setting profiler in " << thread->thread_id;
    PyEval_SetProfile(&PythonHooks::ProfileFunction, nullptr);
  });
  PyThreadState_Swap(curr_thread);
#else   // PY_VERSION_HEX >= 0x030C0000
  PyEval_SetProfileAllThreads(&PythonHooks::ProfileFunction, nullptr);
#endif  // PY_VERSION_HEX >= 0x030C0000
}

/*static*/ void PythonHookContext::ClearProfilerInAllThreads() {
#if PY_VERSION_HEX < 0x030C0000
  PyThreadState* curr_thread = PyThreadState_Get();
  ForEachThread(curr_thread, [](PyThreadState* thread) {
    VLOG(1) << "Clearing profiler in " << thread->thread_id;
    PyEval_SetProfile(nullptr, nullptr);
  });
  PyThreadState_Swap(curr_thread);
#else   // PY_VERSION_HEX >= 0x030C0000
  PyEval_SetProfileAllThreads(nullptr, nullptr);
#endif  // PY_VERSION_HEX >= 0x030C0000

  // And notify the threading library that we're done.
  ThreadingSetProfile(py::none());
}

/*static*/ void PythonHookContext::EnableTraceMe(bool enable) {
  const char* kModuleName =
      "tensorflow.python.profiler.trace";
  try {
    auto trace_module = py::module::import(kModuleName);
    trace_module.attr("enabled") = py::bool_(enable);
  } catch (const py::error_already_set& e) {
    LOG(ERROR) << "Can't import " << kModuleName;
  }
}

}  // namespace profiler
}  // namespace xla
