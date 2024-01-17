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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/post_calibration.h"

#include "absl/base/nullability.h"
#include "absl/log/die_if_null.h"
#include "absl/status/statusor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project  // IWYU: keep
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/cc/run_passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"
#include "tsl/platform/errors.h"

namespace mlir::quant::stablehlo {

using ::stablehlo::quantization::QuantizationConfig;
using ::tensorflow::quantization::RunPasses;

PostCalibrationComponent::PostCalibrationComponent(
    absl::Nonnull<MLIRContext*> ctx)
    : ctx_(ABSL_DIE_IF_NULL(ctx)) {}  // Crash OK

absl::StatusOr<ModuleOp> PostCalibrationComponent::Run(
    ModuleOp module_op, const QuantizationConfig& config) {
  TF_RETURN_IF_ERROR(
      RunPasses(/*name=*/kName,
                /*add_passes_func=*/[this](PassManager& pm) { AddPasses(pm); },
                *ctx_, module_op));
  return module_op;
}

void PostCalibrationComponent::AddPasses(OpPassManager& pm) const {
  pm.addNestedPass<func::FuncOp>(
      CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(createQuantizeCompositeFunctionsPass());
  // Put InlinerPass before OptimizeGraphPass so that the optimize pass can
  // match the sub-optimal patterns.
  pm.addPass(createInlinerPass());
  pm.addPass(createOptimizeGraphPass());
  AddStablehloQuantToIntPasses(pm);
  AddCallModuleSerializationPasses(pm);
}

}  // namespace mlir::quant::stablehlo
