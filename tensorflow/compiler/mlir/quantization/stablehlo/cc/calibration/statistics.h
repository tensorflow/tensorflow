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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_STATISTICS_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_STATISTICS_H_

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/calibrator/calibration_statistics.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/python/py_function_lib.h"

namespace stablehlo::quantization {

// Reads the calibration statistics from the given directory.
absl::StatusOr<absl::flat_hash_map<
    std::string, tensorflow::calibrator::CalibrationStatistics>>
ReadStatistics(absl::string_view calibration_data_dir);

// Adds calibrated min / max values to CustomAggregator nodes in `graph_def`.
// The min and max values will be added to the "min" and "max" attributes,
// respectively. `calibration_options` provides the strategy to retrieve min and
// max values.
absl::Status AddCalibrationStatistics(
    mlir::ModuleOp module_op, absl::string_view calibration_data_dir,
    const stablehlo::quantization::CalibrationOptions& calibration_options,
    const tensorflow::quantization::PyFunctionLibrary& py_function_library);

// Checks if the model required calibration.
bool IsCalibrationRequired(mlir::ModuleOp module_op);

}  // namespace stablehlo::quantization

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_CALIBRATION_STATISTICS_H_
