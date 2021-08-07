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

#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "tensorflow/cc/saved_model/metrics.h"

namespace tensorflow {
namespace saved_model {
namespace python {

namespace py = pybind11;

void DefineMetricsModule(py::module main_module) {
  auto m = main_module.def_submodule("metrics");

  m.doc() = "Python bindings for TensorFlow SavedModel and Checkpoint Metrics.";

  m.def(
      "IncrementWrite",
      [](const char* write_version) {
        metrics::SavedModelWrite(write_version).IncrementBy(1);
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "GetWrite",
      [](const char* write_version) {
        return metrics::SavedModelWrite(write_version).value();
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/write/count' "
              "counter."));

  m.def(
      "IncrementWriteApi",
      [](const char* api_label) {
        metrics::SavedModelWriteApi(api_label).IncrementBy(1);
      },
      py::doc("Increment the '/tensorflow/core/saved_model/write/api' "
              "counter for API with `api_label`"));

  m.def(
      "GetWriteApi",
      [](const char* api_label) {
        return metrics::SavedModelWriteApi(api_label).value();
      },
      py::doc("Get value of '/tensorflow/core/saved_model/write/api' "
              "counter for `api_label` cell."));

  m.def(
      "IncrementRead",
      [](const char* write_version) {
        metrics::SavedModelRead(write_version).IncrementBy(1);
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Increment the '/tensorflow/core/saved_model/read/count' "
              "counter after reading a SavedModel with the specifed "
              "`write_version`."));

  m.def(
      "GetRead",
      [](const char* write_version) {
        return metrics::SavedModelRead(write_version).value();
      },
      py::kw_only(), py::arg("write_version"),
      py::doc("Get value of '/tensorflow/core/saved_model/read/count' "
              "counter for SavedModels with the specified `write_version`."));

  m.def(
      "IncrementReadApi",
      [](const char* api_label) {
        metrics::SavedModelReadApi(api_label).IncrementBy(1);
      },
      py::doc("Increment the '/tensorflow/core/saved_model/read/api' "
              "counter for API with `api_label`."));

  m.def(
      "GetReadApi",
      [](const char* api_label) {
        return metrics::SavedModelReadApi(api_label).value();
      },
      py::doc("Get value of '/tensorflow/core/saved_model/read/api' "
              "counter for `api_label` cell."));

  m.def(
      "AddCheckpointReadDuration",
      [](const char* api_label, double microseconds) {
        metrics::CheckpointReadDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label`for "
              "'/tensorflow/core/checkpoint/read/read_durations'."));

  m.def(
      "GetCheckpointReadDurations",
      [](const char* api_label) {
        // This function is called sparingly in unit tests, so protobuf
        // (de)-serialization round trip is not an issue.
        return py::bytes(metrics::CheckpointReadDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/read/read_durations'."));

  m.def(
      "AddCheckpointWriteDuration",
      [](const char* api_label, double microseconds) {
        metrics::CheckpointWriteDuration(api_label).Add(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/write_durations'."));

  m.def(
      "GetCheckpointWriteDurations",
      [](const char* api_label) {
        // This function is called sparingly, so protobuf (de)-serialization
        // round trip is not an issue.
        return py::bytes(metrics::CheckpointWriteDuration(api_label)
                             .value()
                             .SerializeAsString());
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get serialized HistogramProto of `api_label` cell for "
              "'/tensorflow/core/checkpoint/write/write_durations'."));

  m.def(
      "AddTrainingTimeSaved",
      [](const char* api_label, double microseconds) {
        metrics::TrainingTimeSaved(api_label).IncrementBy(microseconds);
      },
      py::kw_only(), py::arg("api_label"), py::arg("microseconds"),
      py::doc("Add `microseconds` to the cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/training_time_saved'."));

  m.def(
      "GetTrainingTimeSaved",
      [](const char* api_label) {
        return metrics::TrainingTimeSaved(api_label).value();
      },
      py::kw_only(), py::arg("api_label"),
      py::doc("Get cell `api_label` for "
              "'/tensorflow/core/checkpoint/write/training_time_saved'."));
}

}  // namespace python
}  // namespace saved_model
}  // namespace tensorflow
