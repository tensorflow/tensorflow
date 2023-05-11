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
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/python/util/stack_trace.h"

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

// Returns contents of the line corresponding to the given frame.
//
// Precondition: must be holding Python GIL.
py::str LineContents(const StackFrame& frame) {
  DCheckPyGilStateForStackTrace();
  // Pointers are to avoid static destruction of pybind::object, which
  // occurs in uncontrollable states.
  static const auto* inspect = new py::module(py::module::import("inspect"));
  static const auto* getmodule = new py::function(inspect->attr("getmodule"));
  static const auto* linecache =
      new py::module(py::module::import("linecache"));
  static const auto* checkcache =
      new py::function(linecache->attr("checkcache"));
  static const auto* getline = new py::function(linecache->attr("getline"));
  (*checkcache)(py::str(frame.file_name));

  // Here we use the undocumented second argument of inspect.getmodule to look
  // up a module from a filename. It has been unchanged since 2015.
  const auto& module = (*getmodule)(py::none(), py::str(frame.file_name));
  py::object dict = py::none();
  if (!module.is_none()) {
    // module dict is used by getline to resolve import hooks; see the
    // stdlib's inspect module.
    dict = module.attr("__dict__");
  }
  return py::cast<py::str>(
      (*getline)(py::str(frame.file_name), py::int_(frame.line_number), dict)
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
  explicit StackTraceWrapper(absl::Span<const StackFrame> stack_frames)
      : stack_frames_cache_(std::vector<StackFrame>(stack_frames.begin(),
                                                    stack_frames.end())) {}

  StackTraceWrapper(StackTraceWrapper&& rhs) {
    captured_ = std::move(rhs.captured_);
    source_map_ = std::move(rhs.source_map_);
    filter_ = std::move(rhs.filter_);
    stacklevel_ = rhs.stacklevel_;
    tensorflow::mutex_lock lock(rhs.mu_);
    stack_frames_cache_ = std::move(rhs.stack_frames_cache_);
    last_stack_frame_cache_ = std::move(rhs.last_stack_frame_cache_);
  }

  StackTraceWrapper& operator=(StackTraceWrapper&& rhs) {
    if (&rhs == this) return *this;

    captured_ = std::move(rhs.captured_);
    source_map_ = std::move(rhs.source_map_);
    filter_ = std::move(rhs.filter_);
    stacklevel_ = rhs.stacklevel_;

    tensorflow::mutex_lock self_lock(mu_);
    tensorflow::mutex_lock rhs_lock(rhs.mu_);

    stack_frames_cache_ = std::move(rhs.stack_frames_cache_);
    last_stack_frame_cache_ = std::move(rhs.last_stack_frame_cache_);
    return *this;
  }

  static StackTraceWrapper ExtractStack(
      const std::shared_ptr<SourceMap>& source_map,
      const std::shared_ptr<StringSet>& filter, int stacklevel) {
    return StackTraceWrapper{StackTrace::Capture(-1), source_map, filter,
                             stacklevel};
  }

  absl::Span<const StackFrame> ToFrames() const override {
    tensorflow::mutex_lock lock(mu_);
    if (stack_frames_cache_) {
      return *stack_frames_cache_;
    }
    stack_frames_cache_ = ToFramesInternal(*source_map_);
    return *stack_frames_cache_;
  }

  std::vector<StackFrame> ToUncachedFrames() const override {
    SourceMap source_map;
    return ToFramesInternal(source_map);
  }

  int get_stacklevel() const { return stacklevel_; }

  void set_stacklevel(int stacklevel) { stacklevel_ = stacklevel; }

  std::vector<StackFrame> GetUserFrames(int limit = -1) const {
    PyGILState_STATE state = PyGILState_Ensure();
    std::vector<StackFrame> user_frames = captured_.ToStackFrames(
        *source_map_,
        [&](const char* file_name) {
          return StackTraceFiltering(file_name) ||
                 IsInternalFrameForFilename(file_name);
        },
        /*reverse_traversal=*/true,
        /*limit=*/limit);
    PyGILState_Release(state);
    // ensure we use the original (outermost first) ordering.
    absl::c_reverse(user_frames);
    return user_frames;
  }

  StackFrame LastUserFrame() const override {
    tensorflow::mutex_lock lock(mu_);
    if (last_stack_frame_cache_) {
      return *last_stack_frame_cache_;
    }

    PyGILState_STATE state = PyGILState_Ensure();
    std::vector<StackFrame> last_frame = GetUserFrames(1);

    if (last_frame.empty()) {
      last_stack_frame_cache_ = StackFrame{"", -1, ""};
    } else {
      DCHECK_EQ(last_frame.size(), 1);
      last_stack_frame_cache_ = last_frame[0];
    }
    PyGILState_Release(state);
    return *last_stack_frame_cache_;
  }

  // Erases a section of the stack trace.
  void Erase(int first, int last) {
    tensorflow::mutex_lock lock(mu_);
    if (!stack_frames_cache_) {
      ToFrames();
    }
    DCHECK_GE(first, 0);
    DCHECK_LT(first, stack_frames_cache_->size());
    DCHECK_GE(last, 0);
    DCHECK_LE(last, stack_frames_cache_->size());
    auto it = stack_frames_cache_->begin();
    stack_frames_cache_->erase(it + first, it + last);
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

    tensorflow::mutex_lock lock(mu_);
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

  ~StackTraceWrapper() override {
    PyGILState_STATE state = PyGILState_Ensure();
    captured_.Clear();
    source_map_.reset();
    filter_.reset();
    PyGILState_Release(state);
  }

 private:
  StackTraceWrapper(StackTrace&& captured,
                    const std::shared_ptr<SourceMap>& source_map,
                    const std::shared_ptr<StringSet>& filter, int stacklevel)
      : captured_(std::move(captured)),
        source_map_(source_map),
        filter_(filter),
        stacklevel_(stacklevel) {}

  static std::string ToStringHelper(absl::Span<const StackFrame> stack_frames,
                                    const TracePrintingOptions& opts,
                                    int shared_prefix_size) {
    return absl::StrJoin(
        stack_frames, "\n", [&](std::string* out, const StackFrame& frame) {
          absl::StrAppend(out,
                          StackFrameToString(frame, opts, shared_prefix_size));
        });
  }

  std::vector<StackFrame> ToFramesInternal(SourceMap& source_map) const {
    // Grabbing the GIL solves two purposes: 1) makes the class thread-safe,
    // and 2) ToStackFrames and LineContents actually need it.
    PyGILState_STATE state = PyGILState_Ensure();

    std::vector<StackFrame> frames = captured_.ToStackFrames(
        source_map, [&](const char* f) { return StackTraceFiltering(f); });

    // Drop last stack frames.
    int newsize = frames.size() - stacklevel_;
    if (newsize < 0) {
      newsize = 0;
    }
    frames.resize(newsize);

    PyGILState_Release(state);
    return frames;
  }

  bool StackTraceFiltering(const char* file_name) const {
    return filter_->contains(file_name);
  }

  // Note: Make sure to update move constructor while adding new member
  // variables.
  StackTrace captured_;
  std::shared_ptr<SourceMap> source_map_;
  std::shared_ptr<StringSet> filter_;
  int stacklevel_;

  // Using optional to force destruction while we hold a GIL.
  mutable absl::optional<std::vector<StackFrame>> stack_frames_cache_
      TF_GUARDED_BY(mu_);
  mutable absl::optional<StackFrame> last_stack_frame_cache_ TF_GUARDED_BY(mu_);
  mutable mutex mu_;
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

  py::class_<StackTraceWrapper>(m, "StackTraceWrapper")
      // TODO(slebedev): upstream negative indexing support into pybind11.
      .def(
          "__getitem__",
          [](const StackTraceWrapper& self, py::ssize_t index) {
            absl::Span<const StackFrame> frames = self.ToFrames();
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
            absl::Span<const StackFrame> frames = self.ToFrames();
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
      .def("__delitem__",
           [](StackTraceWrapper& self, py::ssize_t index) {
             absl::Span<const StackFrame> frames = self.ToFrames();
             const size_t eff_index =
                 index < 0 ? frames.size() + index : static_cast<size_t>(index);
             if (eff_index >= frames.size()) {
               throw py::index_error();
             }
             self.Erase(eff_index, eff_index + 1);
           })
      .def("__delitem__",
           [](StackTraceWrapper& self, py::slice slice) {
             absl::Span<const StackFrame> frames = self.ToFrames();
             py::ssize_t start, stop, step, slicelength;
             if (!slice.compute(frames.size(), &start, &stop, &step,
                                &slicelength)) {
               throw py::error_already_set();
             }
             if (step != 1) {
               throw py::index_error();
             }
             if (stop > start) {
               self.Erase(start, stop);
             }
           })
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
      // NOTE(feyu): consider remove this and use traceback.format_list(tb)
      // to format the trace.
      .def("__repr__",
           [](const StackTraceWrapper& self) {
             return py::str(self.ToString({}));
           })
      .def_property(
          "_stacklevel", &StackTraceWrapper::get_stacklevel,
          &StackTraceWrapper::set_stacklevel,
          "Adjusts stacklevel; no effects after ToFrames() is called.")
      .def(
          "uncached",
          [](const StackTraceWrapper& self) {
            return StackTraceWrapper{self.ToUncachedFrames()};
          },
          "Gets stack frames without using (or filling) caches.")
      .def(
          "get_user_frames",
          [](const StackTraceWrapper& self) {
            return StackTraceWrapper{self.GetUserFrames()};
          },
          "Returns the non-framework frames as a new trace object.")
      .def(
          "last_user_frame",
          [](const StackTraceWrapper& self) { return self.LastUserFrame(); },
          "Returns the last non-framework frame.");

  m.def("extract_stack_for_op", [](const PyBindSourceMap& source_map,
                                   const PyBindFileSet& file_set,
                                   TF_Operation* op, int stacklevel) {
    DCHECK(!op->node.GetStackTrace()) << "Should not reset the stack trace";
    op->node.SetStackTrace(
        std::make_shared<StackTraceWrapper>(StackTraceWrapper::ExtractStack(
            source_map.source_map_, file_set.file_set_, stacklevel)));
  });

  m.def(
      "extract_stack",
      [](const PyBindSourceMap& source_map, const PyBindFileSet& file_set) {
        return StackTraceWrapper::ExtractStack(source_map.source_map_,
                                               file_set.file_set_, 1);
      },
      py::return_value_policy::move);
}

}  // namespace tensorflow
