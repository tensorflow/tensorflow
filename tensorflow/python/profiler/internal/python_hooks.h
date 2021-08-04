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
#ifndef TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
#define TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_

#include <memory>
#include <stack>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"

namespace tensorflow {
namespace profiler {

namespace py = ::pybind11;

struct PythonHooksOptions {
  bool enable_trace_python_function = false;
  bool enable_python_traceme = true;
  bool end_to_end_mode = false;
  // Incomplete events are defined as those python calls which we only see
  // either start or end, but not both. If we want to include them in the final
  // result, profiler start, end time are used respectively to the absent
  // timestamps.
  bool include_incomplete_events = true;
};

struct PythonTraceEntry {
  PythonTraceEntry(uint64 start, uint64 end, PyObject* filename, PyObject* name,
                   int firstlineno)
      : start_time_ns(start),
        end_time_ns(end),
        co_filename(filename),
        co_name(name),
        co_firstlineno(firstlineno) {
    Py_XINCREF(co_filename);
    Py_XINCREF(co_name);
  }
  PythonTraceEntry(uint64 start, uint64 end, PyCFunctionObject* func)
      : start_time_ns(start), end_time_ns(end), function_object(func) {
    Py_XINCREF(function_object);
  }

  ~PythonTraceEntry() {
    Py_XDECREF(co_filename);
    Py_XDECREF(co_name);
    Py_XDECREF(function_object);
  }

  PythonTraceEntry(PythonTraceEntry&& other) {
    start_time_ns = other.start_time_ns;
    end_time_ns = other.end_time_ns;
    co_firstlineno = other.co_firstlineno;
    co_filename = other.co_filename;
    co_name = other.co_name;
    function_object = other.function_object;
    other.co_filename = nullptr;
    other.co_name = nullptr;
    other.function_object = nullptr;
  }

  std::string Name() const;

  uint64 start_time_ns;
  uint64 end_time_ns;
  PyObject* co_filename = nullptr;
  PyObject* co_name = nullptr;
  int co_firstlineno = 0;
  PyCFunctionObject* function_object = nullptr;

  PythonTraceEntry(const PythonTraceEntry& other) = delete;
  void operator=(const PythonTraceEntry&) = delete;
  void operator=(PythonTraceEntry&&) = delete;
};

struct PerThreadEvents {
  std::deque<PythonTraceEntry> completed;
  std::stack<PythonTraceEntry> active;
};

class PythonHooks;

class PythonHookContext {
 public:
  void Finalize(XSpace* space);

  friend class ::tensorflow::profiler::PythonHooks;

 private:
  void Start(const PythonHooksOptions& option);
  void Stop();
  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg);
  void CollectData(XPlane* raw_plane);
  static void EnableTraceMe(bool enable);

  static void SetProfilerInAllThreads();
  static void ClearProfilerInAllThreads();

  void operator=(const PythonHookContext&) = delete;
  void operator=(PythonHookContext&&) = delete;

  absl::flat_hash_map<int64, PerThreadEvents> entries_;
  uint64 start_timestamp_ns_;
  PythonHooksOptions options_;
  // In end to end mode, Python get uninitialized before Stop()/Finalize(), we
  // need to buffer the result.
  absl::optional<XPlane> end_to_end_xplane_;
};

// Singleton for tracing python function calls.
class PythonHooks {
 public:
  static PythonHooks* GetSingleton();

  void Start(const PythonHooksOptions& option) {
    if (active_context_) return;
    active_context_ = std::make_unique<PythonHookContext>();
    active_context_->Start(option);
  }

  std::unique_ptr<PythonHookContext> Stop() {
    if (e2e_context_) {
      auto* e2e_context = e2e_context_;
      e2e_context_ = nullptr;
      return absl::WrapUnique(e2e_context);
    }

    if (!active_context_) return nullptr;
    active_context_->Stop();
    std::unique_ptr<PythonHookContext> output = std::move(active_context_);
    active_context_.reset();
    return output;
  }

  friend class ::tensorflow::profiler::PythonHookContext;

 private:
  void ProfileSlow(const py::object& frame, const string& event,
                   const py::object& arg);

  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg) {
    if (TF_PREDICT_TRUE(active_context_)) {
      active_context_->ProfileFast(frame, what, arg);
    }
  }

  static void set_e2e_context(PythonHookContext* e2e_context) {
    e2e_context_ = e2e_context;
  }

  static PythonHookContext* e2e_context() { return e2e_context_; }

  static int ProfileFunction(PyObject* obj, PyFrameObject* frame, int what,
                             PyObject* arg);

  // active_context_ are accessed when GIL is held, therefore no race
  // conditions.
  std::unique_ptr<PythonHookContext> active_context_;
  static PythonHookContext* e2e_context_;
};

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
