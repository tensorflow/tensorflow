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
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/quantization_config.pb.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

namespace mlir::quant::stablehlo {

using ::stablehlo::quantization::CalibrationOptions;
using ::stablehlo::quantization::DebuggerConfig;
using ::stablehlo::quantization::PipelineConfig;
using ::stablehlo::quantization::QuantizationSpecs;
using ::stablehlo::quantization::StaticRangePtqPreset;

void AddPreCalibrationPasses(OpPassManager& pm,
                             const CalibrationOptions& calibration_options,
                             const QuantizationSpecs& quantization_specs,
                             const DebuggerConfig& debugger_config) {
  // Convert NCHW tensors to NHWC at along with extra optimizations as
  // downstream passes perform better optimizations when dealing with NHWC
  // formatted tensors.
  AddProcessNchwTensorPasses(pm);

  pm.addPass(CreateLiftQuantizableSpotsAsFunctionsPass(quantization_specs));
  if (debugger_config.debugger_type() !=
      DebuggerConfig::DEBUGGER_TYPE_UNSPECIFIED) {
    pm.addPass(CreateAddDumpTensorOpPass(debugger_config.debugger_type(),
                                         debugger_config.log_dir_path()));
  }
  pm.addNestedPass<func::FuncOp>(
      CreateInsertCustomAggregationOpsPass(calibration_options));
  pm.addPass(CreateIssueIDsOfCustomAggregationOpsPass());
}

void AddPostCalibrationPasses(OpPassManager& pm,
                              const PipelineConfig& pipeline_config,
                              const QuantizationSpecs& specs) {
  QuantizeCompositeFunctionsPassOptions options;
  // TODO: b/331120943 - Temporarily set below to true, signaling per-channel
  // quantization will be applied for all where applicable. This will be
  // replaced by individual `Method` in `QuantizationSpecs`.
  options.enable_per_channel_quantized_weight_ = true;
  // For debugging purposes.
  options.mlir_dump_file_name_ = "quantize_composite_functions";
  options.enable_weight_only_ = false;
  options.merge_fusion_with_dequantize_ =
      pipeline_config.merge_fusion_with_dequantize();

  AddShapeLegalizationPasses(pm);
  pm.addNestedPass<func::FuncOp>(
      CreateConvertCustomAggregationOpToQuantStatsPass());
  pm.addPass(createQuantizeCompositeFunctionsPass(options));
  // Add an inliner pass to inline quantized StableHLO functions.
  pm.addPass(createInlinerPass());
  if (pipeline_config.unpack_quantized_types()) {
    AddStablehloQuantToIntPasses(pm);
  }
}

void AddWeightOnlyQuantizationPasses(
    OpPassManager& pm, const QuantizationSpecs& quantization_specs,
    const PipelineConfig& pipeline_config,
    const DebuggerConfig& debugger_config) {
  // For models with NCHW convolution format. This pass is required because
  // downstream pipeline handles NHWC convolution better for most cases.
  pm.addNestedPass<func::FuncOp>(createNchwConvolutionToNhwcPass());

  // Folds `stablehlo.constant`->`stablehlo.transpose` patterns, which is often
  // generated as by-products after optimizing dimension numbers (e.g.
  // NCHW->NHWC convolution conversion).
  pm.addNestedPass<func::FuncOp>(createFoldConstantTransposePass());
  pm.addPass(CreateLiftQuantizableSpotsAsFunctionsPass(quantization_specs));
  if (debugger_config.debugger_type() !=
      DebuggerConfig::DEBUGGER_TYPE_UNSPECIFIED) {
    pm.addPass(CreateAddDumpTensorOpPass(debugger_config.debugger_type(),
                                         debugger_config.log_dir_path()));
  }
  AddShapeLegalizationPasses(pm);
  QuantizeCompositeFunctionsPassOptions options;
  // For debugging purposes.
  options.mlir_dump_file_name_ = "quantize_composite_functions";
  options.enable_weight_only_ = true;
  pm.addPass(createQuantizeCompositeFunctionsPass(options));

  // Add an inliner pass to inline quantized StableHLO functions.
  pm.addPass(createInlinerPass());
  if (pipeline_config.unpack_quantized_types()) {
    AddStablehloQuantToIntPasses(pm);
  }
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
  pm.addPass(createReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass());
  // ReplaceStablehloOpsInMainFunctionWithXlaCallModuleOpsPass may create
  // duplicate constants. Add canonicalizer to deduplicate.
  pm.addNestedPass<func::FuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(TF::CreateXlaCallModuleSerializationPass());
}

void AddProcessNchwTensorPasses(OpPassManager& pm) {
  // For models with NCHW convolution format. This pass is required because
  // downstream pipeline handles NHWC convolution better for most cases.
  pm.addNestedPass<func::FuncOp>(createNchwConvolutionToNhwcPass());

  // Recursively push down the `stablehlo.transpose` ops for activations
  // generated by the `NchwConvolutionToNhwc` pass.
  pm.addNestedPass<func::FuncOp>(createDeferActivationTransposePass());

  // Folds `stablehlo.constant`->`stablehlo.transpose` patterns, which is often
  // generated as by-products after optimizing dimension numbers (e.g.
  // NCHW->NHWC convolution conversion).
  pm.addNestedPass<func::FuncOp>(createFoldConstantTransposePass());
}

void RegisterPassPipelines() {
  static PassPipelineRegistration<> nchw_tensor_format_processing_pipeline(
      /*arg=*/"stablehlo-process-nchw-tensor",
      /*description=*/"Optimizes tensors with NCHW format.",
      AddProcessNchwTensorPasses);
}

}  // namespace mlir::quant::stablehlo
