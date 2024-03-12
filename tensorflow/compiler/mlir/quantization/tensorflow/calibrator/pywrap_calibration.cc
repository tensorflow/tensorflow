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
#include <optional>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"  // from @pybind11
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"

namespace py = ::pybind11;

namespace {

using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibratorSingleton;

// Retrieves collected statistics of a `CustomAggregator` node from the
// singleton. `id` is the identifier of the `CustomAggregator`.
CalibrationStatistics GetStatisticsFromCalibrator(const absl::string_view id) {
  std::optional<CalibrationStatistics> statistics =
      CalibratorSingleton::GetStatistics(id);

  if (!statistics.has_value()) {
    throw py::value_error(absl::StrFormat(
        "Calibrated data does not exist. Cannot find statistics."
        "value for id: '%s'",
        id));
  }

  return *statistics;
}

}  // namespace

PYBIND11_MODULE(pywrap_calibration, m) {
  // Allows type casting protobuf objects.
  pybind11_protobuf::ImportNativeProtoCasters();

  m.doc() = "Defines functions for interacting with CalibratorSingleton.";

  m.def(
      // If the function signature changes, likely its corresponding .pyi type
      // hinting should also change.
      // LINT.IfChange
      "clear_calibrator",
      []() -> void
      // LINT.ThenChange(pywrap_calibration.pyi:clear_calibrator)
      { CalibratorSingleton::ClearCollectedInformation(); },
      R"pbdoc(
      Clears the collected metrics from the calibrator.
    )pbdoc");
  m.def(
      // If the function signature changes, likely its corresponding .pyi type
      // hinting should also change.
      // LINT.IfChange
      "clear_data_from_calibrator",
      [](const absl::string_view id) -> void
      // LINT.ThenChange(pywrap_calibration.pyi:clear_data_from_calibrator)
      { CalibratorSingleton::ClearData(id); },
      R"pbdoc(
      Clears the collected data of the given id from calibrator.
      )pbdoc",
      py::arg("id"));
  m.def(
      // If the function signature changes, likely its corresponding .pyi type
      // hinting should also change.
      // LINT.IfChange
      "get_statistics_from_calibrator",
      [](const absl::string_view id) -> CalibrationStatistics {
        // LINT.ThenChange(pywrap_calibration.pyi:get_statistics_from_calibrator)
        return GetStatisticsFromCalibrator(id);
      },
      R"pbdoc(
      Returns the proto CalibrationStatistics given id from calibrator.
      )pbdoc",
      py::arg("id"));
}
