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
#ifndef XLA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
#define XLA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "pybind11/cast.h"
#include "pybind11/pybind11.h"
#include "pybind11/pytypes.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/types.h"
#include "tsl/profiler/protobuf/xplane.pb.h"

namespace xla {
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
  // Capture the source/line information for a PyCodeObject object.
  // In eager mode, keeping a reference to PyCodeObject leaks device memory.
  PythonTraceEntry(uint64_t start, uint64_t end, PyCodeObject* py_code_object)
      : start_time_ns(start),
        end_time_ns(end),
        co_filename(py_code_object->co_filename),
        co_name(py_code_object->co_name),
        co_firstlineno(py_code_object->co_firstlineno) {
    Py_XINCREF(co_filename);
    Py_XINCREF(co_name);
  }
  // Capture the source/line information for a PyCFunctionObject object.
  // In eager mode, keeping a reference to PyCFunctionObject leaks device
  // memory.
  PythonTraceEntry(uint64_t start, uint64_t end,
                   PyCFunctionObject* py_c_function)
      : start_time_ns(start),
        end_time_ns(end),
        m_module(py_c_function->m_module) {
    Py_XINCREF(m_module);
    if (auto* method_def = py_c_function->m_ml;
        method_def != nullptr && method_def->ml_name != nullptr) {
      method_name = method_def->ml_name;
    }
  }

  ~PythonTraceEntry() {
    Py_XDECREF(co_filename);
    Py_XDECREF(co_name);
    Py_XDECREF(m_module);
  }

  PythonTraceEntry(PythonTraceEntry&& other) noexcept {
    start_time_ns = other.start_time_ns;
    end_time_ns = other.end_time_ns;
    co_firstlineno = other.co_firstlineno;
    co_filename = other.co_filename;
    co_name = other.co_name;
    method_name = std::move(other.method_name);
    m_module = other.m_module;
    other.co_filename = nullptr;
    other.co_name = nullptr;
    other.method_name = "";
    other.m_module = nullptr;
  }

  std::string Name() const;

  uint64_t start_time_ns;
  uint64_t end_time_ns;
  PyObject* co_filename = nullptr;
  PyObject* co_name = nullptr;
  int co_firstlineno = 0;
  std::string method_name;
  PyObject* m_module = nullptr;

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
  void Finalize(tensorflow::profiler::XSpace* space);

  friend class ::xla::profiler::PythonHooks;

 private:
  void Start(const PythonHooksOptions& option);
  void Stop();
  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg);
  void CollectData(tensorflow::profiler::XPlane* raw_plane);
  static void EnableTraceMe(bool enable);

  static void SetProfilerInAllThreads();
  static void ClearProfilerInAllThreads();

  void operator=(const PythonHookContext&) = delete;
  void operator=(PythonHookContext&&) = delete;

  // The thread id to entries map, Note: by convention the thread id is
  // int64_t to be consistent with cpu tracer when serialize to Xspace.
  absl::flat_hash_map<int64_t, PerThreadEvents> entries_;
  uint64_t start_timestamp_ns_;
  PythonHooksOptions options_;
  // In end to end mode, Python get uninitialized before Stop()/Finalize(), we
  // need to buffer the result.
  std::optional<tensorflow::profiler::XPlane> end_to_end_xplane_;
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

  friend class ::xla::profiler::PythonHookContext;

 private:
  void ProfileSlow(const py::object& frame, const std::string& event,
                   const py::object& arg);

  void ProfileFast(PyFrameObject* frame, int what, PyObject* arg) {
    if (TF_PREDICT_TRUE(active_context_)) {
      active_context_->ProfileFast(frame, what, arg);
    }
  }

  static void set_e2e_context(PythonHookContext* e2e_context) {
    e2e_context_ = e2e_context;
  }

  static int ProfileFunction(PyObject* obj, PyFrameObject* frame, int what,
                             PyObject* arg);

  // active_context_ are accessed when GIL is held, therefore no race
  // conditions.
  std::unique_ptr<PythonHookContext> active_context_;
  static PythonHookContext* e2e_context_;
};

}  // namespace profiler
}  // namespace xla

#endif  // XLA_PYTHON_PROFILER_INTERNAL_PYTHON_HOOKS_H_
