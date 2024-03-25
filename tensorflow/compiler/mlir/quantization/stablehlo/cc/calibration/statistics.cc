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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/statistics.h"

#include <optional>
#include <string>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/graph_def.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"

namespace stablehlo::quantization {
namespace {

using ::stablehlo::quantization::CalibrationOptions;
using ::tensorflow::GraphDef;
using ::tensorflow::NodeDef;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibratorSingleton;
using ::tensorflow::quantization::PyFunctionLibrary;

}  // namespace

absl::Status AddCalibrationStatistics(
    GraphDef& graph_def, const CalibrationOptions& calibration_options,
    const PyFunctionLibrary& py_function_library) {
  absl::Status status = absl::OkStatus();
  MutateNodeDefs(graph_def, [&py_function_library, &calibration_options,
                             &status](NodeDef& node_def) {
    if (node_def.op() != "CustomAggregator") return;
    const std::string& id = node_def.attr().at("id").s();
    std::optional<CalibrationStatistics> statistics =
        CalibratorSingleton::GetStatistics(id);
    if (statistics == std::nullopt) {
      status = absl::InternalError(
          absl::StrFormat("Calibrated data does not exist. Cannot find "
                          "statistics. value for id: %s",
                          id));
      return;
    }

    const auto [min_value, max_value] =
        py_function_library.GetCalibrationMinMaxValue(*statistics,
                                                      calibration_options);
    CalibratorSingleton::ClearData(id);

    (*node_def.mutable_attr())["min"].set_f(min_value);
    (*node_def.mutable_attr())["max"].set_f(max_value);
  });
  return status;
}

}  // namespace stablehlo::quantization
