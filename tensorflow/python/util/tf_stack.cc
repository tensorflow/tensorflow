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

#include "include/pybind11/pybind11.h"
#include "include/pybind11/stl_bind.h"

struct StackFrame;  // Forward declaration.

PYBIND11_MAKE_OPAQUE(std::vector<StackFrame>);

namespace tensorflow {

namespace {

namespace py = pybind11;

struct StackFrame {
  py::str filename;
  int lineno;
  py::str name;
  py::object globals;
  int func_start_lineno;
};

std::vector<StackFrame> ExtractStack(ssize_t limit, const py::list& mappers,
                                     const py::list& filters) {
  const py::dict& source_map =
      mappers.size() == 0
          ? py::dict()
          : mappers[mappers.size() - 1].attr("get_effective_source_map")();
  const py::set& filtered_filenames =
      filters.size() == 0
          ? py::set()
          : filters[filters.size() - 1].attr("get_filtered_filenames")();

  const auto* tstate = PyThreadState_GET();
  // Drop extract_stack() wrapper-function frame from the result.
  const PyFrameObject* f = tstate->frame->f_back;  // TODO(slebedev): INCREF?

  std::vector<StackFrame> ret;
  // 16 is somewhat arbitrary, but TensorFlow stack traces tend to be deep.
  ret.reserve(limit < 0 ? 16 : static_cast<size_t>(limit));
  for (; f != nullptr && (limit < 0 || ret.size() < limit); f = f->f_back) {
    PyCodeObject* co = f->f_code;
    int lineno = PyFrame_GetLineNumber(const_cast<PyFrameObject*>(f));
    auto filename = py::reinterpret_borrow<py::str>(co->co_filename);
    auto name = py::reinterpret_borrow<py::str>(co->co_name);

    // TODO(slebedev): consider moving the mappers/filters to C++ as well.
    if (source_map.size() > 0) {
      const auto& key = py::make_tuple(filename, lineno);
      if (source_map.contains(key)) {
        const py::tuple& mapped = source_map[key];
        filename = mapped[0];
        lineno = py::cast<py::int_>(mapped[1]);
        name = mapped[2];
      }
    }

    // Never filter the innermost frame.
    // TODO(slebedev): upstream py::set::contains to pybind11.
    if (!ret.empty() &&
        PySet_Contains(filtered_filenames.ptr(), filename.ptr()))
      continue;

    const auto& globals = py::reinterpret_borrow<py::object>(f->f_globals);
    const int func_start_lineno = co->co_firstlineno;
    ret.push_back({std::move(filename), lineno, std::move(name), globals,
                   func_start_lineno});
  }

  std::reverse(ret.begin(), ret.end());
  return ret;
}

}  // namespace

PYBIND11_MODULE(_tf_stack, m) {
  // TODO(slebedev): consider dropping convert_stack in favor of
  // a lazily initialized StackFrame.code property (using linecache).
  py::class_<StackFrame>(m, "StackFrame")
      .def(py::init<const py::str&, int, const py::str&, const py::object&,
                    int>())
      .def_readonly("filename", &StackFrame::filename)
      .def_readonly("lineno", &StackFrame::lineno)
      .def_readonly("name", &StackFrame::name)
      .def_readonly("globals", &StackFrame::globals)
      .def_readonly("func_start_lineno", &StackFrame::func_start_lineno)
      .def("__repr__", [](const StackFrame& self) {
        return py::str(
                   "StackFrame(filename={}, lineno={}, name={}, globals={}, "
                   "func_start_lineno={})")
            .format(self.filename, self.lineno, self.name, self.globals,
                    self.func_start_lineno);
      });

  py::bind_vector<std::vector<StackFrame>>(m, "Stack", py::module_local(true));

  m.def("extract_stack", [](const py::object& limit, const py::list& mappers,
                            const py::list& filters) {
    // In Python 3.X ``traceback.extract_stack`` allows ``limit`` to
    // either be None or -1.
    return ExtractStack(limit.is_none() ? -1 : py::cast<ssize_t>(limit),
                        mappers, filters);
  });
}

}  // namespace tensorflow
