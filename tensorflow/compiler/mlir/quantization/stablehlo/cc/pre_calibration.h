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
#ifndef TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PRE_CALIBRATION_H_
#define TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PRE_CALIBRATION_H_

#include <utility>

#include "absl/log/die_if_null.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/component.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/quantization_options.pb.h"

namespace mlir::quant::stablehlo {

// Performs pre-calibration graph transformation as part of post-training
// static-range quantization.

// The resulting `ModuleOp` contains `TF::CustomAggregatorOp`s for collecting
// quantization statistics, along with `TF::XlaCallModuleOp`s that correspond to
// lifted quantizable functions.
class PreCalibrationComponent : public Component {
 public:
  PreCalibrationComponent(
      MLIRContext* ctx,
      tensorflow::quantization::CalibrationOptions calibration_options)
      : ctx_(*ABSL_DIE_IF_NULL(ctx)),  // Crash OK
        calibration_options_(std::move(calibration_options)) {}

  absl::StatusOr<ModuleOp> Run(
      ModuleOp,
      const ::stablehlo::quantization::QuantizationConfig& config) override;

 private:
  MLIRContext& ctx_;
  // TODO: b/315747711 - Allow `QuantizationConfig` to express calibration
  // options and remove this field.
  tensorflow::quantization::CalibrationOptions calibration_options_;
};

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PRE_CALIBRATION_H_
