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

#include <Python.h>

#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/types/span.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/pytypes.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/python/lib/core/safe_pyobject_ptr.h"
#include "tensorflow/python/util/function_parameter_canonicalizer.h"

namespace py = pybind11;

class FunctionParameterCanonicalizerWrapper {
 public:
  FunctionParameterCanonicalizerWrapper(absl::Span<const char*> arg_names,
                                        absl::Span<PyObject*> defaults)
      : function_parameter_canonicalizer_(arg_names, defaults) {}

  tensorflow::FunctionParameterCanonicalizer function_parameter_canonicalizer_;
};

PYBIND11_MODULE(_function_parameter_canonicalizer_binding_for_test, m) {
  py::class_<FunctionParameterCanonicalizerWrapper>(
      m, "FunctionParameterCanonicalizer")
      .def(py::init([](std::vector<std::string> arg_names, py::tuple defaults) {
        std::vector<const char*> arg_names_c_str;
        for (const std::string& name : arg_names)
          arg_names_c_str.emplace_back(name.c_str());

        tensorflow::Safe_PyObjectPtr defaults_fast(
            PySequence_Fast(defaults.ptr(), "Expected tuple"));
        if (!defaults) throw py::error_already_set();
        PyObject** default_items = PySequence_Fast_ITEMS(defaults_fast.get());
        return new FunctionParameterCanonicalizerWrapper(
            absl::MakeSpan(arg_names_c_str),
            absl::MakeSpan(default_items,
                           PySequence_Fast_GET_SIZE(defaults_fast.get())));
      }))
      .def("canonicalize", [](FunctionParameterCanonicalizerWrapper& self,
                              py::args args, py::kwargs kwargs) {
        std::vector<PyObject*> result_raw(
            self.function_parameter_canonicalizer_.GetArgSize());

        bool is_suceeded = self.function_parameter_canonicalizer_.Canonicalize(
            args.ptr(), kwargs.ptr(), absl::MakeSpan(result_raw));

        if (!is_suceeded) {
          CHECK(PyErr_Occurred());
          throw py::error_already_set();
        }

        py::list result;
        for (PyObject* obj : result_raw) result.append(obj);
        return result;
      });
}
