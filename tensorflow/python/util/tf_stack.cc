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

  py::object line() const {
    static const auto* linecache =
        new py::module(py::module::import("linecache"));
    const auto& checkcache = linecache->attr("checkcache");
    const auto& getline = linecache->attr("getline");
    checkcache(filename);
    const auto& code =
        py::cast<py::str>(getline(filename, lineno, globals).attr("strip")());
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
    const PyCodeObject* co = f->f_code;
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

    if (!ret.empty() &&  // Never filter the innermost frame.
        filtered_filenames.size() > 0 &&
        PySet_Contains(filtered_filenames.ptr(), filename.ptr())) {
      continue;
    }

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
  // TODO(slebedev): rename to FrameSummary to match Python 3.5+.
  py::class_<StackFrame>(m, "StackFrame")
      .def(py::init<const py::str&, int, const py::str&, const py::object&,
                    int>())
      .def_readonly("filename", &StackFrame::filename)
      .def_readonly("lineno", &StackFrame::lineno)
      .def_readonly("name", &StackFrame::name)
      // TODO(slebedev): remove globals and make the constructor private.
      .def_readonly("globals", &StackFrame::globals)
      .def_readonly("func_start_lineno", &StackFrame::func_start_lineno)
      .def_property_readonly("line", &StackFrame::line)
      .def("__repr__",
           [](const StackFrame& self) {
             return py::str("<StackFrame file {}, line {} in {}>")
                 .format(self.filename, self.lineno, self.name);
           })

      // For compatibility with the traceback module.
      .def("__getitem__",
           [](const StackFrame& self, ssize_t index) -> py::object {
             switch (index >= 0 ? index : 4 + index) {
               case 0:
                 return self.filename;
               case 1:
                 return py::cast(self.lineno);
               case 2:
                 return self.name;
               case 3:
                 return self.line();
               default:
                 throw py::index_error();
             }
           })
      .def("__len__", [](const StackFrame&) {
        return 4;  // For compatibility with the traceback module.
      });

  // TODO(slebedev): rename to StackSummary to match Python 3.5+.
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
