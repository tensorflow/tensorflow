/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model_wrapper.h"

#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "pybind11/pybind11.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/exported_model.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/quantize_model.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/python/lib/core/pybind11_lib.h"

namespace tensorflow {
namespace quantization {

void ClearCollectedInformationFromCalibrator() {
  calibrator::CalibratorSingleton::ClearCollectedInformation();
}

void ClearDataFromCalibrator(absl::string_view id) {
  calibrator::CalibratorSingleton::ClearData(id);
}

float GetMinFromCalibrator(absl::string_view id) {
  std::optional<std::pair<float, float>> min_max =
      calibrator::CalibratorSingleton::GetMinMax(id);
  if (!min_max.has_value()) {
    throw py::value_error(absl::StrFormat(
        "No calibrated data; cannot find min value for '%s'", id));
  }

  return min_max->first;
}

float GetMaxFromCalibrator(absl::string_view id) {
  std::optional<std::pair<float, float>> min_max =
      calibrator::CalibratorSingleton::GetMinMax(id);
  if (!min_max.has_value()) {
    throw py::value_error(absl::StrFormat(
        "No calibrated data; cannot find max value for '%s'", id));
  }

  return min_max->second;
}

}  // namespace quantization
}  // namespace tensorflow
