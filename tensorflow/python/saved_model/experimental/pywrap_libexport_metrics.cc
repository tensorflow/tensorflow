/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");;
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "pybind11/pybind11.h"
#include "tensorflow/cc/experimental/libexport/metrics.h"

namespace tensorflow {
namespace py = pybind11;

void DefineMetricsModule(py::module main_module) {
  auto m = main_module.def_submodule("metrics");

  m.doc() = "Python bindings for TensorFlow SavedModel Metrics";

  m.def("IncrementWrite",
        []() { tensorflow::libexport::metrics::Write().IncrementBy(1); });

  m.def("GetWrite",
        []() { return tensorflow::libexport::metrics::Write().value(); });

  m.def("IncrementWriteApi", [](const char* api_label) {
    tensorflow::libexport::metrics::WriteApi(api_label).IncrementBy(1);
  });

  m.def("GetWriteApi", [](const char* api_label) {
    return tensorflow::libexport::metrics::WriteApi(api_label).value();
  });

  m.def("IncrementRead",
        []() { tensorflow::libexport::metrics::Read().IncrementBy(1); });

  m.def("GetRead",
        []() { return tensorflow::libexport::metrics::Read().value(); });

  m.def("IncrementReadApi", [](const char* api_label) {
    tensorflow::libexport::metrics::ReadApi(api_label).IncrementBy(1);
  });

  m.def("GetReadApi", [](const char* api_label) {
    return tensorflow::libexport::metrics::ReadApi(api_label).value();
  });
}

}  // namespace tensorflow
