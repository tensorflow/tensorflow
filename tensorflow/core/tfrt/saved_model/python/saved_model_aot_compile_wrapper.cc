/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/runtime/runtime.h"
#include "tensorflow/core/tfrt/saved_model/saved_model_aot_compile.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;

PYBIND11_MODULE(_pywrap_saved_model_aot_compile, m) {
  py::google::ImportStatusModule();

  py::class_<tensorflow::tfrt_stub::AotOptions>(m, "AotOptions")
      .def(py::init<>());
  m.doc() = "pybind11 AotOptions Python - C++ Wrapper";

  m.def("AotCompileSavedModel", &tensorflow::tfrt_stub::AotCompileSavedModel,
        py::arg("input_model_dir") = absl::string_view(),
        py::arg("aot_options") = tensorflow::tfrt_stub::AotOptions(),
        py::arg("output_model_dir") = absl::string_view());
  m.doc() = "pybind11 AotCompileSavedModel Python - C++ Wrapper";
}
