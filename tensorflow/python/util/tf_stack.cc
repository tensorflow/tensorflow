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
*/

// We extract stack traces in Python using the logic in tf_stack.cc, which
// stores a list of PyCodeObject*. Such stack trace extraction is really fast.
//
// We store the retrieved stack trace within the Node object directly. Then
// whenever the graph is instantiated/copies, we copy the stack trace with it.
// Since the graph instantiation goes through the protobuf roundtrip, we store
// the original stack traces mapping attached in FunctionLibraryDefinition.

// clang-format off
// These headers must be at the top, before including Python.h header
// Otherwise, we get C2039 on MSVC due to 'copysign'
#include "absl/strings/str_cat.h"
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11/complex.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
// clang-format on
#include <frameobject.h>

#include <cstddef>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/synchronization/mutex.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stack_frame.h"
#include "tensorflow/core/util/managed_stack_trace.h"
#include "tensorflow/python/util/stack_trace.h"
#include "tsl/platform/mutex.h"

struct StackFrame;  // Forward declaration.
struct StackTrace;

PYBIND11_MAKE_OPAQUE(std::vector<StackFrame>);
PYBIND11_MAKE_OPAQUE(StackTrace);

namespace tensorflow {

namespace {

namespace py = pybind11;

using StringSet = absl::flat_hash_set<std::string>;

// Python wrapper for a SourceMap.
class PyBindSourceMap {
 public:
  struct State {
    SourceMap source_map;
#ifdef Py_GIL_DISABLED
    mutable absl::Mutex mu;
#endif  // Py_GIL_DISABLED
  };

  PyBindSourceMap() : state_(std::make_shared<State>()) {}

  void UpdateTo(const py::tuple& source_map) {
    // Convert Python-owned data before acquiring the native mutex.
    SourceMap updated;
    for (const auto& item : source_map) {
      const auto& tuple_item = py::cast<py::tuple>(item);

      const auto& key = py::cast<py::tuple>(tuple_item[0]);
      std::string&& k_filename = py::cast<std::string>(key[0]);
      int k_lineno = py::cast<int>(key[1]);

      const auto& value = py::cast<py::tuple>(tuple_item[1]);
      std::string&& v_filename = py::cast<std::string>(value[0]);
      int v_lineno = py::cast<int>(value[1]);
      const auto& function_name_val = value[2];
      std::string&& v_function_name =
          function_name_val.is_none()
              ? ""
              : py::cast<std::string>(function_name_val);

      updated.emplace(
          SourceLoc{k_filename, k_lineno},
          StackFrame({v_filename, v_lineno, v_function_name}));
    }

#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(&state_->mu);
#endif  // Py_GIL_DISABLED
    state_->source_map = std::move(updated);
  }

  std::shared_ptr<State> GetState() const { return state_; }

 private:
  // Stack traces share this state so updates remain visible until the trace is
  // materialized, matching the existing lazy stack-trace behavior.
  std::shared_ptr<State> state_;
};

// Python wrapper for a FileSet.
class PyBindFileSet {
 public:
  struct State {
    StringSet file_set;
#ifdef Py_GIL_DISABLED
    mutable absl::Mutex mu;
#endif  // Py_GIL_DISABLED
  };

  PyBindFileSet() : state_(std::make_shared<State>()) {}

  void UpdateTo(const py::set& file_set) {
    // Convert Python-owned data before acquiring the native mutex.
    StringSet updated;
    for (const auto& item : file_set) {
      updated.insert(py::cast<std::string>(item));
    }

#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(&state_->mu);
#endif  // Py_GIL_DISABLED
    state_->file_set = std::move(updated);
  }

  std::shared_ptr<State> GetState() const { return state_; }

 private:
  // Stack traces share this state so updates remain visible until the trace is
  // materialized, matching the existing lazy stack-trace behavior.
  std::shared_ptr<State> state_;
};

class PyBindGraphDebugInfoBuilder {
 public:
  absl::Status AppendGraphDebugInfo(std::string fn_name,
                                    py::bytes debug_info) {
    // Convert Python-owned data before acquiring the native mutex.
    std::string debug_info_str = debug_info;
#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(&mu_);
#endif  // Py_GIL_DISABLED
    return builder_.AppendGraphDebugInfoStr(fn_name, debug_info_str);
  }

  void AccumulateStackTrace(std::string function, std::string op,
                            const AbstractStackTrace& trace) {
    std::string key = absl::StrCat(op, "@", function);
    auto frozen = std::make_shared<FrozenStackTrace>(trace.ToFrames());

#ifdef Py_GIL_DISABLED
    absl::MutexLock lock(&mu_);
#endif  // Py_GIL_DISABLED
    builder_.AccumulateStackTrace(frozen, key);
  }

  py::bytes Build() const {
    std::string serialized;
#ifdef Py_GIL_DISABLED
    {
      absl::MutexLock lock(&mu_);
      serialized = builder_.ToGraphDebugInfoStr();
    }
#else
    serialized = builder_.ToGraphDebugInfoStr();
#endif  // Py_GIL_DISABLED

    // Construct the Python object after releasing the native mutex.
    return py::bytes(serialized);
  }

 private:
#ifdef Py_GIL_DISABLED
  mutable absl::Mutex mu_;
#endif  // Py_GIL_DISABLED

  GraphDebugInfoBuilder builder_;
};

// Simple caching wrapper around a captured stack trace.
//
// When required, stacks are computed and cached as a `FrozenStackTrace`.
class StackTraceWrapper : public AbstractStackTrace {
 public:
  StackTraceWrapper(
      const std::shared_ptr<StackTrace>& captured,
      const std::shared_ptr<PyBindSourceMap::State>& source_map,
      const std::shared_ptr<PyBindFileSet::State>& filter, int stacklevel)
      : captured_(captured),
        source_map_(source_map),
        filter_(filter),
        stacklevel_(stacklevel) {}

  ~StackTraceWrapper() override {
    PyGILState_STATE state = PyGILState_Ensure();
    captured_.reset();
    source_map_.reset();
    filter_.reset();
    PyGILState_Release(state);
  }

  StackTraceWrapper(StackTraceWrapper&& rhs) = default;
  StackTraceWrapper& operator=(StackTraceWrapper&& rhs) = default;

  static std::unique_ptr<StackTraceWrapper> ExtractStack(
      const std::shared_ptr<PyBindSourceMap::State>& source_map,
      const std::shared_ptr<PyBindFileSet::State>& filter, int stacklevel) {
    return std::make_unique<StackTraceWrapper>(StackTrace::Capture(-1),
                                               source_map, filter, stacklevel);
  }

  absl::Span<const StackFrame> ToFrames() const override {
    ComputeFrozen();
    return cache_->ToFrames();
  }

  std::vector<StackFrame> ToUncachedFrames() const override {
    SourceMap source_map;
    StringSet filter;
    SnapshotTransforms(&source_map, &filter);
    return ToUncachedFrames(source_map, filter);
  }

  std::vector<StackFrame> GetUserFrames(int limit) const override {
    ComputeFrozen();
    return cache_->GetUserFrames(limit);
  }

  StackFrame LastUserFrame() const override {
    ComputeFrozen();
    return cache_->LastUserFrame();
  }

  std::string ToString(const TracePrintingOptions& opts) const override {
    ComputeFrozen();
    return cache_->ToString(opts);
  }

 private:
  void SnapshotTransforms(SourceMap* source_map, StringSet* filter) const {
#ifdef Py_GIL_DISABLED
    {
      absl::MutexLock lock(&source_map_->mu);
      *source_map = source_map_->source_map;
    }
    {
      absl::MutexLock lock(&filter_->mu);
      *filter = filter_->file_set;
    }
#else
    *source_map = source_map_->source_map;
    *filter = filter_->file_set;
#endif  // Py_GIL_DISABLED
  }

  std::vector<StackFrame> ToUncachedFrames(
      const SourceMap& source_map, const StringSet& filter) const {
    std::vector<StackFrame> frames = captured_->ToStackFrames(
        source_map, [&](const char* f) { return filter.contains(f); },
        /*reverse_traversal=*/false, /*limit=*/-1);

    // Drop last stack frames.
    int newsize = frames.size() - stacklevel_;
    if (newsize < 0) {
      newsize = 0;
    }
    frames.resize(newsize);

    return frames;
  }

  void ComputeFrozen() const {
    tsl::mutex_lock lock(mu_);
    if (cache_ != nullptr) {
      return;
    }

    // Copy the transform state while holding its native mutexes, then release
    // them before stack processing, which may access Python objects.
    SourceMap source_map;
    StringSet filter;
    SnapshotTransforms(&source_map, &filter);

    std::vector<StackFrame> frames = ToUncachedFrames(source_map, filter);

    std::vector<StackFrame> user_frames = captured_->ToStackFrames(
        source_map,
        [&](const char* file_name) {
          return filter.contains(file_name) ||
                 IsInternalFrameForFilename(file_name);
        },
        /*reverse_traversal=*/true,
        /*limit=*/-1);
    // ensure we use the original (outermost first) ordering.
    absl::c_reverse(user_frames);

    cache_ = std::make_unique<FrozenStackTrace>(frames, user_frames);
  }

  mutable mutex mu_;
  mutable std::unique_ptr<FrozenStackTrace> cache_;
  std::shared_ptr<const StackTrace> captured_;
  std::shared_ptr<PyBindSourceMap::State> source_map_;
  std::shared_ptr<PyBindFileSet::State> filter_;
  int stacklevel_;
};

}  // namespace

PYBIND11_MODULE(
    _tf_stack, m, pybind11::mod_gil_not_used()) {
  pybind11::google::ImportStatusModule();

  py::class_<PyBindSourceMap>(m, "PyBindSourceMap")
      .def(py::init())
      .def("update_to", &PyBindSourceMap::UpdateTo);

  py::class_<PyBindFileSet>(m, "PyBindFileSet")
      .def(py::init())
      .def("update_to", &PyBindFileSet::UpdateTo);

  py::class_<PyBindGraphDebugInfoBuilder>(m, "GraphDebugInfoBuilder")
      .def(py::init())
      .def("AppendGraphDebugInfo",
           &PyBindGraphDebugInfoBuilder::AppendGraphDebugInfo,
           py::arg("prefix"), py::arg("debug_info"))
      .def("AccumulateStackTrace",
           &PyBindGraphDebugInfoBuilder::AccumulateStackTrace,
           py::arg("function"), py::arg("op"), py::arg("trace"))
      .def("Build", &PyBindGraphDebugInfoBuilder::Build);

  py::class_<StackFrame>(m, "StackFrame")
      .def_property_readonly(
          "filename",
          [](const StackFrame& self) { return py::str(self.file_name); })
      .def_property_readonly(
          "lineno",
          [](const StackFrame& self) { return py::int_(self.line_number); })
      .def_property_readonly(
          "name",
          [](const StackFrame& self) { return py::str(self.function_name); })
      .def_property_readonly("line", [](const StackFrame& self) { return ""; })
      .def("__eq__", &StackFrame::operator==)
      .def("__ne__", &StackFrame::operator!=)
      .def("__hash__",
           [](const StackFrame& self) {
             return absl::Hash<std::tuple<std::string, int, std::string>>()(
                 std::make_tuple(self.file_name, self.line_number,
                                 self.function_name));
           })
      .def("__getitem__",
           [](const StackFrame& self, const py::object& index) -> py::object {
             return py::make_tuple(
                 py::str(self.file_name), py::int_(self.line_number),
                 py::str(self.function_name), py::str(""))[index];
           })
      .def("__iter__",
           [](const StackFrame& self) -> py::iterator {
             return py::iter(py::make_tuple(
                 py::str(self.file_name), py::int_(self.line_number),
                 py::str(self.function_name), py::str("")));
           })
      .def("__repr__",
           [](const StackFrame& self) -> py::str {
             return absl::StrFormat("File \"%s\", line %d, in %s",
                                    self.file_name, self.line_number,
                                    py::str(self.function_name));
           })
      .def("__len__", [](const StackFrame&) { return 4; });

  py::class_<AbstractStackTrace, std::shared_ptr<AbstractStackTrace>>(
      m, "StackTrace")
      .def(
          "__getitem__",
          [](const AbstractStackTrace& self, py::ssize_t index) -> StackFrame {
            absl::Span<const StackFrame> frames = self.ToFrames();
            const size_t eff_index =
                index < 0 ? frames.size() + index : static_cast<size_t>(index);
            if (eff_index >= frames.size()) {
              throw py::index_error();
            }
            return frames[eff_index];
          },
          py::return_value_policy::take_ownership)
      .def(
          "__getitem__",
          [](const AbstractStackTrace& self,
             py::slice slice) -> std::shared_ptr<AbstractStackTrace> {
            absl::Span<const StackFrame> frames = self.ToFrames();
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(frames.size(), &start, &stop, &step,
                               &slicelength)) {
              throw py::error_already_set();
            }
            if (step == 1) {
              return std::make_shared<FrozenStackTrace>(
                  frames.subspan(start, slicelength));
            }
            std::vector<StackFrame> out;
            if (slicelength > 0) {
              out.reserve(slicelength);
            }
            // Python slices allow negative indexing.
            for (py::ssize_t i = start, count = 0; count < slicelength;
                 i += step, ++count) {
              out.push_back(frames[static_cast<size_t>(i)]);
            }
            return std::make_shared<FrozenStackTrace>(out);
          },
          py::return_value_policy::take_ownership)
      .def(
          "__len__",
          [](const AbstractStackTrace& self) { return self.ToFrames().size(); })
      .def("__eq__",
           [](const AbstractStackTrace& self, const AbstractStackTrace& other) {
             return self.ToFrames() == other.ToFrames();
           })
      .def("__hash__",
           [](const AbstractStackTrace& self) {
             return py::hash(py::str(self.ToString({})));
           })
      .def(
          "get_user_frames",
          [](const AbstractStackTrace& self)
              -> std::shared_ptr<AbstractStackTrace> {
            return std::make_shared<FrozenStackTrace>(self.GetUserFrames(-1));
          },
          "Returns the non-framework frames as a new trace object.")
      .def(
          "last_user_frame",
          [](const AbstractStackTrace& self) { return self.LastUserFrame(); },
          "Returns the last non-framework frame.")
      .def("__repr__",
           [](const AbstractStackTrace& self) { return self.ToString({}); });

  m.def(
      "extract_stack",
      [](const PyBindSourceMap& source_map, const PyBindFileSet& file_set,
         int stacklevel) -> std::shared_ptr<AbstractStackTrace> {
        return StackTraceWrapper::ExtractStack(
            source_map.GetState(), file_set.GetState(), stacklevel);
      },
      py::arg("source_map"), py::arg("file_set"), py::arg("stacklevel") = 1,
      py::return_value_policy::take_ownership);

  m.def(
      "LoadTracesFromDebugInfo",
      [](py::bytes data) { return LoadTracesFromDebugInfoStr(data); },
      py::arg("debug_info_proto"));
}

}  // namespace tensorflow
