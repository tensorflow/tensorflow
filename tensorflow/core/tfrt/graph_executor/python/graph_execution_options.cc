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
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"

#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow/core/tfrt/runtime/runtime.h"

namespace py = pybind11;

PYBIND11_MODULE(GraphExecutionRunOptions, m) {
  py::class_<tensorflow::tfrt_stub::GraphExecutionRunOptions>(
      m, "GraphExecutionRunOptions")
      .def(py::init<>());
  m.doc() =
      "pybind11 GraphExecutionRunOptions wrapper";  // optional module docstring
}

PYBIND11_MODULE(_pywrap_graph_execution_options, m) {
  py::class_<tensorflow::tfrt_stub::GraphExecutionOptions>(
      m, "GraphExecutionOptions")
      .def(py::init<const tensorflow::tfrt_stub::Runtime *>());
}
