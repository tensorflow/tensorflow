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

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
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
  // Name of the post-training quantization pre-calibration step. Used for
  // debugging purposes.
  static constexpr absl::string_view kName = "quant_ptq_pre_calibration";

  explicit PreCalibrationComponent(MLIRContext* /*absl_nonnull*/ ctx);

  absl::StatusOr<ModuleOp> Run(
      ModuleOp,
      const ::stablehlo::quantization::QuantizationConfig& config) override;

 private:
  MLIRContext* /*absl_nonnull*/ ctx_;
};

}  // namespace mlir::quant::stablehlo

#endif  // TENSORFLOW_COMPILER_MLIR_QUANTIZATION_STABLEHLO_CC_PRE_CALIBRATION_H_
