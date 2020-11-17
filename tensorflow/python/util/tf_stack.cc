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

#include <Python.h>
#include <frameobject.h>

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/hash/hash.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"
#include "tensorflow/python/util/stack_trace.h"

struct StackFrame;  // Forward declaration.
struct StackTrace;

PYBIND11_MAKE_OPAQUE(std::vector<StackFrame>);
PYBIND11_MAKE_OPAQUE(StackTrace);

namespace tensorflow {

namespace {

namespace py = pybind11;

py::object LineContents(const StackFrame& frame) {
  static const auto* linecache =
      new py::module(py::module::import("linecache"));
  const auto& checkcache = linecache->attr("checkcache");
  const auto& getline = linecache->attr("getline");
  checkcache(py::str(frame.file_name));
  const auto& code = py::cast<py::str>(
      getline(py::str(frame.file_name), py::int_(frame.line_number))
          .attr("strip")());
  ssize_t size = 0;
#if PY_MAJOR_VERSION == 3
    if (PyUnicode_AsUTF8AndSize(code.ptr(), &size) == nullptr) {
      throw py::error_already_set();
    }
#else
    size = PyString_Size(code.ptr());
#endif
    return size > 0 ? static_cast<py::object>(code) : py::none();
}

std::string StackFrameToString(const StackFrame& frame) {
  return py::str("<FrameSummary file {}, line {} in {}>")
      .format(py::str(frame.file_name), py::int_(frame.line_number),
              py::str(frame.function_name));
}

class StackTraceWrapper {
 public:
  StackTraceWrapper(StackTrace&& captured,
                    const StackTraceMapper& stack_trace_mapper,
                    const StackTraceFilter& stack_trace_filter)
      : captured_(std::move(captured)),
        stack_trace_mapper_(stack_trace_mapper),
        stack_trace_filter_(stack_trace_filter) {}

  explicit StackTraceWrapper(absl::Span<StackFrame const> stack_frames)
      : stack_frames_cache_(std::vector<StackFrame>(stack_frames.begin(),
                                                    stack_frames.end())) {}

  absl::Span<StackFrame const> ToFrames() const {
    GenerateCache();
    return *stack_frames_cache_;
  }

  std::string ToString() const {
    GenerateCache();
    return absl::StrJoin(*stack_frames_cache_, "\n",
                         [&](std::string* out, const StackFrame& frame) {
                           absl::StrAppend(out, StackFrameToString(frame));
                         });
  }

  bool IsCacheGenerated() const { return stack_frames_cache_.has_value(); }

  void GenerateCache() const {
    if (stack_frames_cache_) {
      return;
    }
    stack_frames_cache_ =
        captured_.ToStackFrames(stack_trace_mapper_, stack_trace_filter_);
    stack_frames_cache_->pop_back();  // Drop last stack frame.
  }

 private:
  mutable absl::optional<std::vector<StackFrame>> stack_frames_cache_;
  StackTrace captured_;

  // TODO(cheshire): store those as C++ datastructures instead.
  StackTraceMapper stack_trace_mapper_;
  StackTraceFilter stack_trace_filter_;
};

}  // namespace

PYBIND11_MODULE(_tf_stack, m) {
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
          "line",
          [](const StackFrame& self) { return py::str(LineContents(self)); })

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
           [](const StackFrame& self) { return StackFrameToString(self); })
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
            std::vector<StackFrame> out;
            out.reserve(slicelength);
            for (int i = start; i < stop; i += step) {
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
             return py::hash(py::str(self.ToString()));
           })
      .def("__repr__", [](const StackTraceWrapper& self) {
        if (self.IsCacheGenerated()) {
          return py::str("<Opaque Stack Trace, access to initialize>");
        }
        return py::str(self.ToString());
      });

  m.def(
      "extract_stack",
      [](const py::object& limit, const py::list& mappers,
         const py::list& filters) {
        // In Python 3.X ``traceback.extract_stack`` allows ``limit`` to
        // either be None or -1.
        int casted_limit = limit.is_none() ? -1 : py::cast<ssize_t>(limit);

        // Raise limit by one since we are dropping the last frame.
        if (casted_limit != -1) casted_limit++;

        const py::dict& source_map = mappers.empty()
                                         ? py::dict()
                                         : mappers[mappers.size() - 1].attr(
                                               "get_effective_source_map")();
        const py::set& filtered_filenames =
            filters.empty()
                ? py::set()
                : filters[filters.size() - 1].attr("get_filtered_filenames")();

        auto mapper = [=](std::string filename,
                          int line_no) -> absl::optional<StackFrame> {
          if (source_map.empty()) {
            return absl::nullopt;
          }
          const auto& key =
              py::make_tuple(py::str(filename), py::int_(line_no));
          if (source_map.contains(key)) {
            const py::tuple& mapped = source_map[key];
            return StackFrame{std::string(py::cast<py::str>(mapped[0])),
                              py::cast<py::int_>(mapped[1]),
                              std::string(py::cast<py::str>(mapped[2]))};
          }

          return absl::nullopt;
        };

        auto filter = [=](std::string filename) -> bool {
          return !filters.empty() &&
                 filtered_filenames.contains(py::str(filename));
        };
        return StackTraceWrapper{StackTrace::Capture(casted_limit), mapper,
                                 filter};
      },
      py::return_value_policy::move);
}

}  // namespace tensorflow
