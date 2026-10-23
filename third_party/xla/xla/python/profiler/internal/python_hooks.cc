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
#include <cstddef>
#include <cstdint>
#include <deque>
#include <initializer_list>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "xla/python/profiler/internal/traceme_state.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/profiler/utils/time_utils.h"
#include "xla/tsl/profiler/utils/xplane_builder.h"
#include "xla/tsl/profiler/utils/xplane_schema.h"
#include "xla/tsl/profiler/utils/xplane_utils.h"
#include "tsl/platform/path.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

#if PY_VERSION_HEX < 0x030C0000
#include "tsl/platform/platform.h"
#endif  // PY_VERSION_HEX < 0x030C0000

#ifdef Py_GIL_DISABLED
#include "absl/synchronization/mutex.h"
#endif  // Py_GIL_DISABLED

namespace xla {
namespace profiler {

namespace {

bool SysSetProfileNone() {
  PyObject* sys_mod = PyImport_ImportModule("sys");
  if (sys_mod == nullptr) {
    return false;
  }
  PyObject* res = PyObject_CallMethod(sys_mod, "setprofile", "O", Py_None);
  Py_DECREF(sys_mod);
  Py_XDECREF(res);
  return res != nullptr;
}

bool ThreadingSetProfile(PyObject* callback) {
  PyObject* threading_mod = PyImport_ImportModule("threading");
  if (threading_mod == nullptr) {
    return false;
  }
  PyObject* res = PyObject_CallMethod(threading_mod, "setprofile", "O",
                                      callback != nullptr ? callback : Py_None);
  Py_DECREF(threading_mod);
  Py_XDECREF(res);
  return res != nullptr;
}

// Returns the UTF-8 text of `obj`, or "<unknown>" if `obj` is not a str.
// Requires the GIL; the result is valid while `obj` is alive.
absl::string_view PyUnicodeToStringOrUnknown(PyObject* obj) {
  if (obj != nullptr && PyUnicode_Check(obj)) {
    Py_ssize_t size = 0;
    const char* str = PyUnicode_AsUTF8AndSize(obj, &size);
    if (str != nullptr) {
      return absl::string_view(str, static_cast<size_t>(size));
    }
    PyErr_Clear();
  }
  return "<unknown>";
}

// Returns the trace event name of `event`. `to_string` returns the text of one
// of the event's Python string objects.
template <typename ToString>
std::string GetEventName(const PythonTraceEntry& event, ToString to_string) {
  if (event.co_filename != nullptr) {
    absl::string_view filename = to_string(event.co_filename);
    absl::string_view function = to_string(event.co_name);
    return absl::StrCat("$", tsl::io::Basename(filename), ":",
                        event.co_firstlineno, " ", function);
  }
  // Python stack does not have a filename/line_no for native calls.
  // Use module name and function/method name instead.
  absl::string_view filename = to_string(event.m_module);
  if (!event.method_name.empty()) {
    return absl::StrCat("$", filename, " ", event.method_name);
  }
  return "$<unknown>";
}

// Ends the calls still open in `active` at `end_time_ns` and moves them to
// `events`, innermost first.
void CloseActiveEvents(uint64_t end_time_ns,
                       std::stack<PythonTraceEntry>& active,
                       std::deque<PythonTraceEntry>& events) {
  while (!active.empty()) {
    PythonTraceEntry& entry = active.top();
    entry.end_time_ns = end_time_ns;
    events.push_back(std::move(entry));
    active.pop();
  }
}

// Moves the Python references held by `event` into `ref_counts`. Reads only the
// pointer values, so it does not need the GIL.
void TakeReferences(PythonTraceEntry& event,
                    absl::flat_hash_map<PyObject*, size_t>& ref_counts) {
  for (PyObject* obj : {event.co_filename, event.co_name, event.m_module}) {
    if (obj != nullptr) {
      ++ref_counts[obj];
    }
  }
  event.owns_python_refs = false;
}

// Drops `count` references to `obj`. Requires the GIL.
void ReleaseReferences(PyObject* obj, size_t count) {
#if !defined(Py_GIL_DISABLED) && !defined(Py_LIMITED_API) && \
    !defined(Py_REF_DEBUG)
  // Drop all but the last reference in one step, then let Py_DECREF free the
  // object if that was the last one. Neither call changes immortal objects.
  Py_SET_REFCNT(obj, Py_REFCNT(obj) - static_cast<Py_ssize_t>(count - 1));
  Py_DECREF(obj);
#else
  // Free-threaded, limited-API, and ref-debug builds: one Py_DECREF each.
  for (size_t i = 0; i < count; ++i) {
    Py_DECREF(obj);
  }
#endif
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
  if constexpr (tsl::kIsDebugBuild) {
    // If debug version of python runtime is used, (e.g. monolithic binaries in
    // g3). PyGILState_Check will fail because current thread's PyThreadState
    // is not the one that holding GIL (after PyThreadState_Swap). This extra
    // check in PyEval_SetProfile is not useful, but will sporadic crash if user
    // use debug version of python runtime. In this case, we fallback to only
    // set up profile hooks in current threads.
    // In OSS, the python runtime and tensorflow profiler are built separately.
    // So this workaround doesn't apply.
    callback(curr_thread);
  } else {
    for (PyThreadState* p = curr_thread; p != nullptr; p = p->next) {
      PyThreadState_Swap(p);
      callback(p);
    }
    for (PyThreadState* p = curr_thread->prev; p != nullptr; p = p->prev) {
      PyThreadState_Swap(p);
      callback(p);
    }
  }
}

#endif  // PY_VERSION_HEX

}  // namespace

/*static*/ PythonHookContext* PythonHooks::e2e_context_ = nullptr;

/*static*/ PyObject* PythonHookContext::PyProfileCallback(PyObject* /*self*/,
                                                          PyObject* const* args,
                                                          Py_ssize_t nargs) {
  if (nargs != 3) {
    PyErr_SetString(PyExc_TypeError, "profile_callback expects 3 arguments");
    return nullptr;
  }
  const char* event = PyUnicode_AsUTF8(args[1]);
  if (event == nullptr) {
    return nullptr;
  }

  PythonHooks::GetSingleton()->ProfileSlow(
      reinterpret_cast<PyFrameObject*>(args[0]), event, args[2]);
  if (!SysSetProfileNone()) {
    return nullptr;
  }
  PyEval_SetProfile(&PythonHooks::ProfileFunction, nullptr);
  Py_RETURN_NONE;
}

/*static*/ PyObject* PythonHookContext::PyAtexitCallback(PyObject* /*self*/,
                                                         PyObject* /*args*/) {
  PythonHooks* singleton = PythonHooks::GetSingleton();
  auto e2e_context = singleton->Stop();
  // Serialize into internal storage before the tracked PyCodeObjects
  // went out of scope.
  if (e2e_context) {
    e2e_context->CollectData(nullptr);
    PythonHooks::set_e2e_context(e2e_context.release());
  }
  Py_RETURN_NONE;
}

std::string PythonTraceEntry::Name() const {
  return GetEventName(*this, PyUnicodeToStringOrUnknown);
}

PythonHooks* PythonHooks::GetSingleton() {
  static PythonHooks* const singleton = new PythonHooks;
  return singleton;
}

PythonHookContext::~PythonHookContext() {
  if (Py_IsInitialized()) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    for (auto& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
      absl::MutexLock lock(&shard.mu);
#endif  // Py_GIL_DISABLED
      shard.entries.clear();
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHookContext::Start(const PythonHooksOptions& options) {
  if (!Py_IsInitialized()) {
    return;
  }

  options_ = options;
  start_timestamp_ns_ = tsl::profiler::GetCurrentTimeNanos();
  if (options_.enable_python_traceme || options_.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_python_traceme) {
      traceme_enabled = true;
    }
    if (options_.enable_trace_python_function) {
      if (!SetProfilerInAllThreads()) {
        LOG(ERROR) << "Can't set profiler in all threads.";
        PyErr_Print();
      }
    }
    if (options_.end_to_end_mode) {
      if (!RegisterAtexitCallback()) {
        LOG(ERROR) << "Can't install atexit handler for e2e mode.";
        PyErr_Print();
      }
    }
    PyGILState_Release(gil_state);
  }
}

void PythonHookContext::Stop() {
  stopped_ = true;
  if (!Py_IsInitialized()) {
    return;
  }
  if (options_.enable_python_traceme || options_.enable_trace_python_function) {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    if (options_.enable_trace_python_function) {
      if (!ClearProfilerInAllThreads()) {
        LOG(ERROR) << "Can't clear profiler in all threads.";
        PyErr_Print();
      }
    }
    if (options_.enable_python_traceme) {
      traceme_enabled = false;
    }
    PyGILState_Release(gil_state);
  }
}

std::vector<PerThreadConsumeData> PythonHookContext::Consume() {
  if (!Py_IsInitialized()) {
    return {};
  }

  struct HarvestedEvents {
    HarvestedEvents() = default;
    HarvestedEvents(const HarvestedEvents&) = delete;
    HarvestedEvents& operator=(const HarvestedEvents&) = delete;
    HarvestedEvents(HarvestedEvents&&) = default;
    HarvestedEvents& operator=(HarvestedEvents&&) = default;

    int64_t thread_id = 0;
    std::deque<PythonTraceEntry> events;
  };
  std::vector<HarvestedEvents> harvested;
  const bool stopped = stopped_.load(std::memory_order_acquire);

  // 1. With the GIL: move the per-thread event queues out of `entry_shards_`.
  // After Stop(), calls still on the stack are ended now and moved out too.
  {
    PyGILState_STATE gil_state = PyGILState_Ensure();
    const uint64_t now = tsl::profiler::GetCurrentTimeNanos();
    for (EntryShard& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
      absl::MutexLock lock(shard.mu);
#else
      DCHECK(PyGILState_Check());
#endif  // Py_GIL_DISABLED
      // NOLINTNEXTLINE
      for (auto& [thread_id, thread_events] : shard.entries) {
        HarvestedEvents& h = harvested.emplace_back();
        h.thread_id = thread_id;
        h.events.swap(thread_events.completed);
        if (stopped && options_.include_incomplete_events) {
          CloseActiveEvents(now, thread_events.active, h.events);
          CloseActiveEvents(now, thread_events.active_c, h.events);
        }
      }
    }
    PyGILState_Release(gil_state);
  }

  // 2. Without the GIL: take over the events' Python references, counted per
  // object. This reads pointer values only, never the objects.
  absl::flat_hash_map<PyObject*, size_t> ref_counts;
  for (HarvestedEvents& h : harvested) {
    for (PythonTraceEntry& event : h.events) {
      TakeReferences(event, ref_counts);
    }
  }

  // 3. With the GIL: copy the text of each distinct object, then drop all of
  // its references at once. If the interpreter is shutting down, skip this and
  // leak the references; the names then read "<unknown>".
  absl::flat_hash_map<PyObject*, std::string> names;
  if (!ref_counts.empty() && Py_IsInitialized()) {
    names.reserve(ref_counts.size());
    PyGILState_STATE gil_state = PyGILState_Ensure();
    // NOLINTNEXTLINE
    for (const auto& [obj, count] : ref_counts) {
      names.try_emplace(obj, PyUnicodeToStringOrUnknown(obj));
      ReleaseReferences(obj, count);
    }
    PyGILState_Release(gil_state);
  }

  // 4. Without the GIL: build the event names from `names`, then free the
  // queues. No entry owns a reference any more.
  auto copied_text = [&names](PyObject* obj) -> absl::string_view {
    auto it = names.find(obj);
    if (it == names.end()) {
      return "<unknown>";
    }
    return it->second;
  };
  std::vector<PerThreadConsumeData> consumed_data;
  consumed_data.reserve(harvested.size());
  for (HarvestedEvents& h : harvested) {
    // NOLINTNEXTLINE(clang-diagnostic-pre-c++20-compat)
    VLOG(1) << "Consuming " << h.events.size() << " events on thread "
            << h.thread_id;
    PerThreadConsumeData& thread_data = consumed_data.emplace_back();
    thread_data.thread_id = h.thread_id;
    thread_data.events.reserve(h.events.size());
    for (const PythonTraceEntry& event : h.events) {
      thread_data.events.push_back({GetEventName(event, copied_text),
                                    event.start_time_ns, event.end_time_ns});
    }
    h.events.clear();
  }

  return consumed_data;
}

void PythonHookContext::CollectData(tensorflow::profiler::XPlane* raw_plane) {
  if (raw_plane == nullptr) {
    end_to_end_xplane_.emplace();
    raw_plane = &*end_to_end_xplane_;
  }
  tsl::profiler::XPlaneBuilder plane(raw_plane);
  for (auto& shard : entry_shards_) {
#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(shard.mu);
#else
    DCHECK(PyGILState_Check());
#endif  // Py_GIL_DISABLED
    for (auto& it : shard.entries) {
      int64_t thread_id = it.first;
      auto& thread_events = it.second;
      VLOG(1) << "Collecting " << thread_events.completed.size() << ":"
              << thread_events.active.size() << " events on thread "
              << thread_id;
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
    shard.entries.clear();
  }
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

void PythonHooks::ProfileSlow(PyFrameObject* frame, absl::string_view event,
                              PyObject* arg) {
  int what;

  if (absl::ConsumePrefix(&event, "c_")) {
    if (event == "call") {
      what = PyTrace_C_CALL;
    } else if (event == "return") {
      what = PyTrace_C_RETURN;
    } else if (event == "exception") {
      what = PyTrace_C_EXCEPTION;
    } else {
      return;
    }
  } else {
    if (event == "call") {
      what = PyTrace_CALL;
    } else if (event == "return") {
      what = PyTrace_RETURN;
    } else if (event == "exception") {
      what = PyTrace_EXCEPTION;
    } else {
      return;
    }
  }

  ProfileFast(frame, what, arg);
}

void PythonHookContext::ProfileFast(PyFrameObject* frame, int what,
                                    PyObject* arg) {
  const int64_t thread_id = tsl::Env::Default()->GetCurrentThreadId();
  uint64_t now = tsl::profiler::GetCurrentTimeNanos();
  int shard_id = thread_id % kNumEntryShards;
  EntryShard& shard = entry_shards_[shard_id];
#ifdef Py_GIL_DISABLED
  absl::MutexLock lock(shard.mu);
#else
  DCHECK(PyGILState_Check());
#endif  // Py_GIL_DISABLED
  auto& thread_traces = shard.entries[thread_id];

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
        thread_traces.active_c.emplace(now, 0, func);
      }
      break;
    }
    case PyTrace_C_RETURN:
    case PyTrace_C_EXCEPTION: {
      if (PyCFunction_Check(arg)) {
        if (!thread_traces.active_c.empty()) {
          auto& entry = thread_traces.active_c.top();
          entry.end_time_ns = now;
          thread_traces.completed.emplace_back(std::move(entry));
          thread_traces.active_c.pop();
        } else if (options_.include_incomplete_events) {
          // Only the end of the events is recorded, use profiler start as
          // start timestamp of the new event.
          auto* func = reinterpret_cast<PyCFunctionObject*>(arg);
          thread_traces.completed.emplace_back(start_timestamp_ns_, now, func);
        }
      }
      break;
    }
    default:
      break;
  }
}

/*static*/ bool PythonHookContext::SetProfilerInAllThreads() {
  // We also want any new threads started to use our profiler.
  // NOTE: threading does not provide a C API equivalent to
  // `threading.setprofile` so we are forced to go via Python to setup the
  // profile when a new thread is created. After the first callback in that
  // thread we unregister the Python profile function and use
  // `PyEval_SetProfile` to register a C profiler which has significantly less
  // overhead (>2x faster).
  static PyMethodDef kProfileMethodDef = {
      "profile_callback",
      reinterpret_cast<PyCFunction>(PythonHookContext::PyProfileCallback),
      METH_FASTCALL, nullptr};
  PyObject* callback = PyCFunction_New(&kProfileMethodDef, nullptr);
  if (callback == nullptr) {
    return false;
  }
  bool ok = ThreadingSetProfile(callback);
  Py_DECREF(callback);
  if (!ok) {
    return false;
  }

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
  return true;
}

/*static*/ bool PythonHookContext::ClearProfilerInAllThreads() {
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
  return ThreadingSetProfile(nullptr);
}

/*static*/ bool PythonHookContext::RegisterAtexitCallback() {
  // When end to end mode is used, Stop() and Finalize() i.e. symbolization
  // and data collection happens during C's atexit(), when Py_FinalizeEx()
  // already called.
  static PyMethodDef kAtexitMethodDef = {"e2e_atexit_callback",
                                         PythonHookContext::PyAtexitCallback,
                                         METH_NOARGS, nullptr};
  PyObject* atexit_mod = PyImport_ImportModule("atexit");
  if (atexit_mod == nullptr) {
    return false;
  }
  PyObject* py_fn = PyCFunction_New(&kAtexitMethodDef, nullptr);
  if (py_fn == nullptr) {
    Py_DECREF(atexit_mod);
    return false;
  }
  PyObject* res = PyObject_CallMethod(atexit_mod, "register", "O", py_fn);
  Py_DECREF(py_fn);
  Py_DECREF(atexit_mod);
  Py_XDECREF(res);
  return res != nullptr;
}

}  // namespace profiler
}  // namespace xla
