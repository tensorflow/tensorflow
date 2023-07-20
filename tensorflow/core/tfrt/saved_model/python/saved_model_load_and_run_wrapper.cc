/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "pybind11/detail/common.h"  // from @pybind11
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11/stl.h"  // from @pybind11
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "tensorflow/core/tfrt/graph_executor/graph_execution_options.h"
#include "tensorflow/core/tfrt/saved_model/python/saved_model_load_and_run.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace py = pybind11;
namespace tensorflow::tfrt_stub {

PYBIND11_MODULE(_pywrap_saved_model, m) {
  py::google::ImportStatusModule();

  m.def("LoadSavedModel", &tensorflow::tfrt_stub::LoadSavedModel,
        py::arg("saved_model_dir") = absl::string_view(),
        py::arg("tags") = std::unordered_set<std::string>());

  m.def("Run", &tensorflow::tfrt_stub::Run,
        py::arg("saved_model") =
            *(tensorflow::tfrt_stub::LoadSavedModel("", {}).value()),
        py::arg("run_options") =
            tensorflow::tfrt_stub::GraphExecutionRunOptions(),
        py::arg("name") = absl::string_view(),
        py::arg("inputs") = absl::Span<const tensorflow::Tensor>(),
        py::arg("outputs") = std::vector<tensorflow::Tensor>());
}
}  // namespace tensorflow::tfrt_stub
