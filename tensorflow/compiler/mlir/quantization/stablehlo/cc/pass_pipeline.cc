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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/pass_pipeline.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

namespace mlir::quant::stablehlo {

using ::stablehlo::quantization::DebuggerConfig;
using ::stablehlo::quantization::PipelineConfig;
using ::stablehlo::quantization::QuantizationSpecs;
using ::stablehlo::quantization::StaticRangePtqPreset;
using ::tensorflow::quantization::CalibrationOptions;

void AddPreCalibrationPasses(OpPassManager& pm,
                             const CalibrationOptions& calibration_options,
                             const QuantizationSpecs& quantization_specs,
                             const DebuggerConfig& debugger_config) {
  // For models with NCHW convolution format. This pass is required because
  // downstream pipeline handles NHWC convolution better for most cases.
  pm.addNestedPass<func::FuncOp>(createNchwConvolutionToNhwcPass());
  pm.addPass(CreateLiftQuantizableSpotsAsFunctionsPass(quantization_specs));
  if (debugger_config.debugger_type() !=
      DebuggerConfig::DEBUGGER_TYPE_UNSPECIFIED) {
    pm.addPass(CreateAddDumpTensorOpPass(debugger_config.debugger_type(),
                                         debugger_config.log_dir_path()));
  }
  pm.addNestedPass<func::FuncOp>(
      CreateInsertCustomAggregationOpsPass(calibration_options));
  pm.addPass(CreateIssueIDsOfCustomAggregationOpsPass());
  // StableHLO Quantizer currently uses TF's calibration passes. Serialize
  // the StableHLO module as tf.XlaCallModule to run calibration.
  AddCallModuleSerializationPasses(pm);
}

void AddPostCalibrationPasses(
    OpPassManager& pm, const PipelineConfig& pipeline_config,
    const StaticRangePtqPreset& static_range_ptq_preset) {
  QuantizeCompositeFunctionsPassOptions options;
  options.enable_per_channel_quantized_weight_ =
      static_range_ptq_preset.enable_per_channel_quantized_weight();
  // For debugging purposes.
  options.mlir_dump_file_name_ = "quantize_composite_functions";
  pm.addNestedPass<func::FuncOp>(
      CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(createQuantizeCompositeFunctionsPass(options));
  if (pipeline_config.unpack_quantized_types()) {
    AddStablehloQuantToIntPasses(pm);
  }
  AddCallModuleSerializationPasses(pm);
}

void AddXlaCallModuleOpDeserializationPasses(OpPassManager& pm) {
  pm.addPass(TF::CreateXlaCallModuleDeserializationPass());
  pm.addPass(createRestoreFunctionNamePass());
  pm.addPass(createUnwrapXlaCallModuleOpPass());
  pm.addPass(createSymbolDCEPass());
}

void AddShapeLegalizationPasses(OpPassManager& pm) {
  pm.addPass(mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<func::FuncOp>(
      mhlo::createShapeLegalizeToHloPass(/*legalizeConstraints=*/true));
  // The following 2 passes are used to clean up the spurious UnrealizedCast ops
  // and shape.assuming regions leftover from the ShapeLegalizeToHlo pass. See
  // pass definition for details.
  pm.addPass(createReconcileUnrealizedCastsPass());
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
}

void AddStablehloQuantToIntPasses(OpPassManager& pm) {
  pm.addPass(createInlinerPass());
  // StableHLO -> MHLO legalization.
  pm.addPass(mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<func::FuncOp>(mhlo::createMhloQuantLegalizeToIntPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  // Integer graph optimization relies on chlo broadcast ops for easier handling
  // of dynamic shapes. Therefore we lower chlo ops after optimization.
  pm.addNestedPass<func::FuncOp>(CreateOptimizeIntGraphPass());
  pm.addNestedPass<func::FuncOp>(mhlo::createChloLegalizeToHloPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
  // MHLO -> StableHLO legalization.
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
}

// NOMUTANTS -- Add tests for individual passes with migration below.
void AddCallModuleSerializationPasses(OpPassManager& pm) {
  AddShapeLegalizationPasses(pm);
  // Add an inliner pass to inline quantized StableHLO functions (and others) so
  // that StableHLO ops are properly grouped and converted into XlaCallModule
  // ops by the ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass.
  pm.addPass(createInlinerPass());
  pm.addPass(createReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass());
  // ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass may create
  // duplicate constants. Add canonicalizer to deduplicate.
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(TF::CreateXlaCallModuleSerializationPass());
}

}  // namespace mlir::quant::stablehlo
