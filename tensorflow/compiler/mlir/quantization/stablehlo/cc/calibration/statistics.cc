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

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/calibration/min_max_value.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibrator_singleton.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_quant_ops.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"

namespace stablehlo::quantization {
namespace {

using ::stablehlo::quantization::CalibrationOptions;
using ::tensorflow::calibrator::CalibrationStatistics;
using ::tensorflow::calibrator::CalibratorSingleton;
using ::tensorflow::quantization::PyFunctionLibrary;

}  // namespace

absl::Status AddCalibrationStatistics(
    mlir::ModuleOp module_op, const CalibrationOptions& calibration_options,
    const PyFunctionLibrary& py_function_library) {
  absl::Status status = absl::OkStatus();
  module_op.walk([&py_function_library, &calibration_options,
                  &status](mlir::TF::CustomAggregatorOp aggregator_op) {
    mlir::StringRef id = aggregator_op.getId();
    std::optional<CalibrationStatistics> statistics =
        CalibratorSingleton::GetStatistics(id);
    if (statistics == std::nullopt) {
      status = absl::InternalError(
          absl::StrFormat("Calibrated data does not exist. Cannot find "
                          "statistics. value for id: %s",
                          id));
      return;
    }

    const std::optional<MinMaxValue> min_max_values =
        py_function_library.GetCalibrationMinMaxValue(*statistics,
                                                      calibration_options);
    CalibratorSingleton::ClearData(id);

    if (min_max_values == std::nullopt) {
      status = absl::InternalError(
          "Cannot find min/max values for calibration statistics.");
      return;
    }

    const auto [min_value, max_value] = *min_max_values;
    mlir::OpBuilder builder(aggregator_op);
    aggregator_op->setAttr("min", builder.getF32FloatAttr(min_value));
    aggregator_op->setAttr("max", builder.getF32FloatAttr(max_value));
  });
  return status;
}

}  // namespace stablehlo::quantization
