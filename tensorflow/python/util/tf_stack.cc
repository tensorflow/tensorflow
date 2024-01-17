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
#include "pybind11_abseil/absl_casters.h"  // from @pybind11_abseil
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11/complex.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11/stl_bind.h"  // from @pybind11
// clang-format on
#include <frameobject.h>

#include <algorithm>
#include <vector>

#include "Python.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_debug_info_builder.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/status.h"
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
  PyBindSourceMap() : source_map_(std::make_shared<SourceMap>()) {}

  // Shares ownership with whoever captures traces in the scope of this map.
  std::shared_ptr<SourceMap> source_map_;
};

// Python wrapper for a FileSet.
class PyBindFileSet {
 public:
  PyBindFileSet() : file_set_(std::make_shared<StringSet>()) {}

  // Shares ownership with whoever captures traces in the scope of this set.
  std::shared_ptr<StringSet> file_set_;
};

// Simple caching wrapper around a captured stack trace.
//
// When required, stacks are computed and cached as a `FrozenStackTrace`.
class StackTraceWrapper : public AbstractStackTrace {
 public:
  StackTraceWrapper(const std::shared_ptr<StackTrace>& captured,
                    const std::shared_ptr<SourceMap>& source_map,
                    const std::shared_ptr<StringSet>& filter, int stacklevel)
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
      const std::shared_ptr<SourceMap>& source_map,
      const std::shared_ptr<StringSet>& filter, int stacklevel) {
    return std::make_unique<StackTraceWrapper>(StackTrace::Capture(-1),
                                               source_map, filter, stacklevel);
  }

  absl::Span<const StackFrame> ToFrames() const override {
    ComputeFrozen();
    return cache_->ToFrames();
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
  void ComputeFrozen() const {
    tsl::mutex_lock lock(mu_);
    if (cache_ != nullptr) {
      return;
    }

    std::vector<StackFrame> frames = captured_->ToStackFrames(
        *source_map_, [&](const char* f) { return StackTraceFiltering(f); },
        /*reverse_traversal=*/false, /*limit=*/-1);

    // Drop last stack frames.
    int newsize = frames.size() - stacklevel_;
    if (newsize < 0) {
      newsize = 0;
    }
    frames.resize(newsize);

    std::vector<StackFrame> user_frames = captured_->ToStackFrames(
        *source_map_,
        [&](const char* file_name) {
          return StackTraceFiltering(file_name) ||
                 IsInternalFrameForFilename(file_name);
        },
        /*reverse_traversal=*/true,
        /*limit=*/-1);
    // ensure we use the original (outermost first) ordering.
    absl::c_reverse(user_frames);

    cache_ = std::make_unique<FrozenStackTrace>(frames, user_frames);
  }

  bool StackTraceFiltering(const char* file_name) const {
    return filter_->contains(file_name);
  }

  mutable mutex mu_;
  mutable std::unique_ptr<FrozenStackTrace> cache_;
  std::shared_ptr<const StackTrace> captured_;
  std::shared_ptr<SourceMap> source_map_;
  std::shared_ptr<StringSet> filter_;
  int stacklevel_;
};

}  // namespace

PYBIND11_MODULE(_tf_stack, m) {
  pybind11::google::ImportStatusModule();

  py::class_<PyBindSourceMap>(m, "PyBindSourceMap")
      .def(py::init())
      .def("update_to",
           [](const PyBindSourceMap& self, const py::tuple& source_map) {
             self.source_map_->clear();
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

               self.source_map_->emplace(
                   SourceLoc{k_filename, k_lineno},
                   StackFrame({v_filename, v_lineno, v_function_name}));
             }
           });

  py::class_<PyBindFileSet>(m, "PyBindFileSet")
      .def(py::init())
      .def("update_to", [](const PyBindFileSet& self, const py::set& file_set) {
        self.file_set_->clear();
        for (const auto& item : file_set) {
          self.file_set_->insert(py::cast<std::string>(item));
        }
      });

  py::class_<GraphDebugInfoBuilder>(m, "GraphDebugInfoBuilder")
      .def(py::init())
      .def(
          "AppendGraphDebugInfo",
          [](GraphDebugInfoBuilder& self, std::string fn_name,
             py::bytes debug_info_str) {
            return self.AppendGraphDebugInfoStr(fn_name, debug_info_str);
          },
          py::arg("prefix"), py::arg("debug_info"))
      .def(
          "AccumulateStackTrace",
          [](GraphDebugInfoBuilder& self, std::string function, std::string op,
             const AbstractStackTrace& trace) {
            std::string key = absl::StrCat(op, "@", function);
            self.AccumulateStackTrace(
                std::make_shared<FrozenStackTrace>(trace.ToFrames()), key);
          },
          py::arg("function"), py::arg("op"), py::arg("trace"))
      .def("Build", [](GraphDebugInfoBuilder& self) -> py::bytes {
        return py::bytes(self.ToGraphDebugInfoStr());
      });

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
            out.reserve(slicelength);
            // Python slices allow negative indexing.
            for (int i = start; i != stop; i += step) {
              out.push_back(frames[i]);
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
        return StackTraceWrapper::ExtractStack(source_map.source_map_,
                                               file_set.file_set_, stacklevel);
      },
      py::arg("source_map"), py::arg("file_set"), py::arg("stacklevel") = 1,
      py::return_value_policy::take_ownership);

  m.def(
      "LoadTracesFromDebugInfo",
      [](py::bytes data) { return LoadTracesFromDebugInfoStr(data); },
      py::arg("debug_info_proto"));
}

}  // namespace tensorflow
