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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_post_calibration.h"

#include <memory>

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/status/statusor.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/config.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/tf_pass_pipeline.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/instrumentations/tf_save_report.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "xla/tsl/platform/errors.h"

namespace mlir::tf_quant::stablehlo {

using ::stablehlo::quantization::GetReportFilePath;
using ::stablehlo::quantization::PipelineConfig;
using ::stablehlo::quantization::QuantizationConfig;
using ::stablehlo::quantization::QuantizationSpecs;
using ::tensorflow::quantization::RunPasses;

PostCalibrationComponent::PostCalibrationComponent(
    MLIRContext* absl_nonnull ctx)
    : ctx_(ABSL_DIE_IF_NULL(ctx)) {}  // Crash OK

absl::StatusOr<ModuleOp> PostCalibrationComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  TF_RETURN_IF_ERROR(RunPasses(
      kName, /*add_passes_func=*/
      [&config](PassManager& pm) {
        // Add instrumentation to save quantization report after quantization.
        pm.addInstrumentation(
            std::make_unique<SaveQuantizationReportInstrumentation>(
                GetReportFilePath(config)));

        tf_quant::stablehlo::AddPostCalibrationPasses(
            pm, config.pipeline_config(), config.specs());
      },
      *ctx_, module_op));
  return module_op;
}

void PostCalibrationComponent::AddPasses(
    OpPassManager& pm, const QuantizationSpecs& specs,
    const PipelineConfig& pipeline_config) const {
  tf_quant::stablehlo::AddPostCalibrationPasses(pm, pipeline_config, specs);
}

}  // namespace mlir::tf_quant::stablehlo
