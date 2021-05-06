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
==============================================================================*/

// We extract stack traces in Python using the logic in tf_stack.cc, which
// stores a list of PyCodeObject*. Such stack trace extraction is really fast.
//
// We store the retrieved stack trace within the Node object directly. Then
// whenever the graph is instantiated/copies, we copy the stack trace with it.
// Since the graph instantiation goes through the protobuf roundtrip, we store
// the original stack traces mapping attached in FunctionLibraryDefinition.

#include <Python.h>
#include <frameobject.h>

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/python/util/stack_trace.h"

struct StackFrame;  // Forward declaration.
struct StackTrace;

PYBIND11_MAKE_OPAQUE(std::vector<StackFrame>);
PYBIND11_MAKE_OPAQUE(StackTrace);

namespace tensorflow {

namespace {

namespace py = pybind11;

using SourceLoc = std::tuple<std::string, int>;

using SourceMap = absl::flat_hash_map<SourceLoc, StackFrame>;

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

// Returns contents of the line corresponding to the given frame.
//
// Precondition: must be holding Python GIL.
py::str LineContents(const StackFrame& frame) {
  DCheckPyGilStateForStackTrace();
  static const auto* linecache =
      new py::module(py::module::import("linecache"));
  const auto& checkcache = linecache->attr("checkcache");
  const auto& getline = linecache->attr("getline");
  checkcache(py::str(frame.file_name));
  return py::cast<py::str>(
      getline(py::str(frame.file_name), py::int_(frame.line_number))
          .attr("strip")());
}

// Ignores the frames containing this substring for common prefix calculation.
static const char* kFilenameToIgnorePrefix = "<embedded";

// Converts the given stack frame to string, according to options defined in
// `opts`.
std::string StackFrameToString(
    const StackFrame& frame,
    const AbstractStackTrace::TracePrintingOptions& opts,
    int shared_prefix_size = 0) {
  std::string out = absl::StrFormat(
      "File \"%s\", line %d, in %s",
      absl::StrContains(frame.file_name, kFilenameToIgnorePrefix)
          ? frame.file_name
          : frame.file_name.substr(shared_prefix_size),
      frame.line_number, frame.function_name);

  if (opts.show_line_contents) {
    PyGILState_STATE state = PyGILState_Ensure();
    std::string line_contents = std::string(LineContents(frame));
    PyGILState_Release(state);
    if (!line_contents.empty()) {
      absl::StrAppend(&out, "\n  ", line_contents);
    }
  }
  return out;
}

class StackTraceWrapper : public AbstractStackTrace {
 public:
  StackTraceWrapper(StackTrace&& captured,
                    const std::shared_ptr<SourceMap>& source_map,
                    const std::shared_ptr<StringSet>& filter)
      : captured_(std::move(captured)),
        source_map_(source_map),
        filter_(filter) {}

  explicit StackTraceWrapper(absl::Span<StackFrame const> stack_frames)
      : stack_frames_cache_(std::vector<StackFrame>(stack_frames.begin(),
                                                    stack_frames.end())) {}

  static StackTraceWrapper ExtractStack(
      const std::shared_ptr<SourceMap>& source_map,
      const std::shared_ptr<StringSet>& filter) {
    return StackTraceWrapper{StackTrace::Capture(-1), source_map, filter};
  }

  absl::Span<StackFrame const> ToFrames() const override {
    if (stack_frames_cache_) {
      return *stack_frames_cache_;
    }

    // Grabbing the GIL solves two purposes: 1) makes the class thread-safe,
    // and 2) ToStackFrames and LineContents actually need it.
    PyGILState_STATE state = PyGILState_Ensure();

    stack_frames_cache_ = captured_.ToStackFrames(
        [&](std::pair<const char*, int> p) { return StackTraceMapping(p); },
        [&](const char* f) { return StackTraceFiltering(f); });
    stack_frames_cache_->pop_back();  // Drop last stack frame.
    PyGILState_Release(state);
    return *stack_frames_cache_;
  }

  StackFrame LastUserFrame() const override {
    if (last_stack_frame_cache_) {
      return *last_stack_frame_cache_;
    }

    PyGILState_STATE state = PyGILState_Ensure();
    std::vector<StackFrame> last_frame = captured_.ToStackFrames(
        [&](std::pair<const char*, int> p) { return StackTraceMapping(p); },
        [&](const char* file_name) {
          return StackTraceFiltering(file_name) ||
                 IsInternalFrameForFilename(file_name);
        },
        /*reverse_traversal=*/true,
        /*limit=*/1);

    if (last_frame.empty()) {
      last_stack_frame_cache_ = StackFrame{"", -1, ""};
    } else {
      DCHECK_EQ(last_frame.size(), 1);
      last_stack_frame_cache_ = last_frame[0];
    }
    PyGILState_Release(state);
    return *last_stack_frame_cache_;
  }

  std::string ToString(const TracePrintingOptions& opts) const override {
    std::vector<std::string> files_to_find_prefix;
    for (const StackFrame& frame : ToFrames()) {
      if (!absl::StrContains(frame.file_name, kFilenameToIgnorePrefix)) {
        files_to_find_prefix.push_back(frame.file_name);
      }
    }
    int shared_prefix_size =
        opts.filter_common_prefix
            ? io::CommonPathPrefix(files_to_find_prefix).size()
            : 0;

    if (!opts.drop_internal_frames) {
      return ToStringHelper(*stack_frames_cache_, opts, shared_prefix_size);
    }

    std::vector<StackFrame> filtered_frames;
    for (const StackFrame& frame : *stack_frames_cache_) {
      if (!IsInternalFrameForFilename(frame.file_name)) {
        filtered_frames.push_back(frame);
      }
    }
    return ToStringHelper(filtered_frames, opts, shared_prefix_size);
  }

  StackTraceWrapper(StackTraceWrapper&&) = default;
  ~StackTraceWrapper() override {
    PyGILState_STATE state = PyGILState_Ensure();
    captured_.Clear();
    source_map_.reset();
    filter_.reset();
    PyGILState_Release(state);
  }

 private:
  static std::string ToStringHelper(absl::Span<StackFrame const> stack_frames,
                                    const TracePrintingOptions& opts,
                                    int shared_prefix_size) {
    return absl::StrJoin(
        stack_frames, "\n", [&](std::string* out, const StackFrame& frame) {
          absl::StrAppend(out,
                          StackFrameToString(frame, opts, shared_prefix_size));
        });
  }

  absl::optional<StackFrame> StackTraceMapping(SourceLoc loc) const {
    if (source_map_->contains(loc)) {
      return source_map_->at(loc);
    }

    return absl::nullopt;
  }

  bool StackTraceFiltering(const char* file_name) const {
    return filter_->contains(file_name);
  }

  StackTrace captured_;
  std::shared_ptr<SourceMap> source_map_;
  std::shared_ptr<StringSet> filter_;

  // Using optional to force destruction while we hold a GIL.
  mutable absl::optional<std::vector<StackFrame>> stack_frames_cache_;
  mutable absl::optional<StackFrame> last_stack_frame_cache_;
};

}  // namespace

PYBIND11_MODULE(_tf_stack, m) {
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
                   SourceLoc(k_filename, k_lineno),
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
      .def_property_readonly(
          "line", [](const StackFrame& self) { return LineContents(self); })

      // For compatibility with the traceback module.
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
                 py::str(self.function_name), LineContents(self))[index];
           })
      .def("__iter__",
           [](const StackFrame& self) {
             return py::iter(py::make_tuple(
                 py::str(self.file_name), py::int_(self.line_number),
                 py::str(self.function_name), LineContents(self))

             );
           })
      .def("__repr__",
           [](const StackFrame& self) { return StackFrameToString(self, {}); })
      .def("__len__", [](const StackFrame&) { return 4; });

  py::class_<StackTraceWrapper>(m, "StackTraceWrapper", py::module_local(true))
      // TODO(slebedev): upstream negative indexing support into pybind11.
      .def(
          "__getitem__",
          [](const StackTraceWrapper& self, ssize_t index) {
            absl::Span<StackFrame const> frames = self.ToFrames();
            const size_t eff_index =
                index < 0 ? frames.size() + index : static_cast<size_t>(index);
            if (eff_index >= frames.size()) {
              throw py::index_error();
            }
            return frames[eff_index];
          },
          py::return_value_policy::reference_internal)
      .def(
          "__getitem__",
          [](const StackTraceWrapper& self, py::slice slice) {
            absl::Span<StackFrame const> frames = self.ToFrames();
            py::ssize_t start, stop, step, slicelength;
            if (!slice.compute(frames.size(), &start, &stop, &step,
                               &slicelength)) {
              throw py::error_already_set();
            }
            if (step == 1) {
              return StackTraceWrapper{frames.subspan(start, slicelength)};
            }
            // TODO(cheshire): Cleanup, use Python slicing logic directly
            // instead.
            std::vector<StackFrame> out;
            out.reserve(slicelength);
            // Python slices allow negative indexing.
            for (int i = start; i != stop; i += step) {
              out.push_back(frames[i]);
            }
            return StackTraceWrapper{out};
          },
          py::return_value_policy::reference_internal)
      .def("__len__",
           [](const StackTraceWrapper& self) { return self.ToFrames().size(); })
      .def("__eq__",
           [](const StackTraceWrapper& self, const StackTraceWrapper& other) {
             return self.ToFrames() == other.ToFrames();
           })
      .def("__hash__",
           [](const StackTraceWrapper& self) {
             return py::hash(py::str(self.ToString({})));
           })
      .def("__repr__",
           [](const StackTraceWrapper& self) {
             return py::str(self.ToString({}));
           })
      .def("last_user_frame",
           [](const StackTraceWrapper& self) { return self.LastUserFrame(); });

  m.def(
      "extract_stack_for_node",
      [](const PyBindSourceMap& source_map, const PyBindFileSet& file_set,
         TF_Operation* op) -> const AbstractStackTrace& {
        Node* node = reinterpret_cast<Node*>(op);
        DCHECK(!node->GetStackTrace()) << "Should not reset the stack trace";
        node->SetStackTrace(
            std::make_shared<StackTraceWrapper>(StackTraceWrapper::ExtractStack(
                source_map.source_map_, file_set.file_set_)));
        return *node->GetStackTrace();
      },
      py::return_value_policy::reference);

  m.def(
      "extract_stack",
      [](const PyBindSourceMap& source_map, const PyBindFileSet& file_set) {
        return StackTraceWrapper::ExtractStack(source_map.source_map_,
                                               file_set.file_set_);
      },
      py::return_value_policy::move);
}

}  // namespace tensorflow
