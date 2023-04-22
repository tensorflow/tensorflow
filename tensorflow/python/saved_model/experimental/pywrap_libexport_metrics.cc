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
namespace metrics = libexport::metrics;

void DefineMetricsModule(py::module main_module) {
  auto m = main_module.def_submodule("metrics");

  m.doc() = "Python bindings for TensorFlow SavedModel Metrics";

  m.def(
      "IncrementWrite", []() { metrics::Write().IncrementBy(1); },
      py::doc("Increment the '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "GetWrite", []() { return metrics::Write().value(); },
      py::doc("Get value of '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "IncrementWriteApi",
      [](const char* api_label, const char* write_version) {
        metrics::WriteApi(api_label, write_version).IncrementBy(1);
      },
      py::arg("api_label"), py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/write/api' "
              "counter for API with `api_label` that writes a SavedModel "
              "with the specifed version."));

  m.def(
      "GetWriteApi",
      [](const char* api_label, const char* write_version) {
        return metrics::WriteApi(api_label, write_version).value();
      },
      py::arg("api_label"), py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/write/api' "
              "counter for (`api_label`, `write_version`) cell."));

  m.def(
      "IncrementRead", []() { metrics::Read().IncrementBy(1); },
      py::doc("Increment the '/tensorflow/core/saved_model/read/count' "
              "counter."));

  m.def(
      "GetRead", []() { return metrics::Read().value(); },
      py::doc("Get value of '/tensorflow/core/saved_model/read/count' "
              "counter."));

  m.def(
      "IncrementReadApi",
      [](const char* api_label, const char* write_version) {
        metrics::ReadApi(api_label, write_version).IncrementBy(1);
      },
      py::arg("api_label"), py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/read/api' "
              "counter for API with `api_label` that reads a SavedModel "
              "with the specifed `write_version`."));

  m.def(
      "GetReadApi",
      [](const char* api_label, const char* write_version) {
        return metrics::ReadApi(api_label, write_version).value();
      },
      py::arg("api_label"), py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/read/api' "
              "counter for (`api_label`, `write_version`) cell."));
}

}  // namespace tensorflow
