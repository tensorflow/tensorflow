/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/transforms.h"

#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_tf_xla_call_module_to_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/smuggle_disallowed_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/stablehlo_passes.h"
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/fold_broadcast_pass.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/rename_entrypoint_to_main.h"
#include "tensorflow/compiler/mlir/stablehlo/transforms/tf_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

namespace mlir {
namespace odml {

void AddTFToStablehloPasses(OpPassManager& pm, bool skip_resize,
                            bool smuggle_disallowed_ops) {
  pm.addPass(CreateRenameEntrypointToMainPass());

  // if the input is a call_xla_module, then unwrap the content
  pm.addPass(mlir::odml::CreateLegalizeTFXlaCallModuleToStablehloPass());
  // TODO: b/230572023 - Consider improving shape inference for While op instead
  // of dropping the attribute. This need not be correct for models not trained
  // on TPU.

  // Optimizes TF graph via cleanups, merges, rewrites, constant folding,
  // and edge case handling where possible.
  pm.addNestedPass<func::FuncOp>(TF::CreateDropWhileShapeInvariantPass());
  pm.addNestedPass<func::FuncOp>(
      tf_executor::CreateTFExecutorGraphPruningPass());
  pm.addNestedPass<func::FuncOp>(
      tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(TF::CreateTFFunctionalControlFlowToRegions());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTensorListOpsDecompositionPass());
  pm.addNestedPass<func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());

  // FreezeVariables only freezes variables for TF v1 types. Separately handle
  // freezing of TF v2 GlobalTensor ops. (Ref: b/206855389)
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
      /*allow_mutable_tensors=*/true));

  // Generic MLIR optimization passes.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  // Legalizes TF UniformQuantized types into MHLO.
  pm.addNestedPass<func::FuncOp>(
      mlir::quant::stablehlo::CreateConvertTFQuantOpsToMHLOPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // TF -> StableHLO legalization.
  AddLegalizeTFToStablehloPasses(pm, /*skip_quantization_ops=*/false,
                                 skip_resize,
                                 /*skip_partitioned_calls=*/false);

  // Wrap disallowed ops in stablehlo.custom_call ops.
  if (smuggle_disallowed_ops) {
    pm.addNestedPass<func::FuncOp>(CreateSmuggleDisallowedOpsPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }
}

void AddMhloOptimizationPasses(OpPassManager& pm,
                               const bool add_fold_broadcast_pass) {
  pm.addNestedPass<func::FuncOp>(createStablehloUnfuseBatchNormPass());
  pm.addNestedPass<func::FuncOp>(createStablehloFuseConvolutionPass());
  // StableHLO -> MHLO legalization.
  pm.addPass(mhlo::createStablehloLegalizeToHloPass());
  // Rewrites some patterns for better performance.
  pm.addNestedPass<func::FuncOp>(createOptimizePass());
  // Conditionally enable below pass because this causes unfused convolutions
  // described in b/293149194. This problem is not replicated in
  // StableHLO Quantizer.
  if (add_fold_broadcast_pass) {
    pm.addNestedPass<func::FuncOp>(createFoldBroadcastPass());
  }

  // Rewrites legacy StableHLO ops.
  pm.addNestedPass<func::FuncOp>(mhlo::createLegalizeEinsumToDotGeneralPass());
  pm.addNestedPass<func::FuncOp>(
      mhlo::createLegalizeTorchIndexSelectToGatherPass());

  pm.addPass(mlir::createCanonicalizerPass());
}

void AddStablehloOptimizationPasses(OpPassManager& pm) {
  // The current plan of record is to avoid doing optimization passes
  // on StableHLO, treating StableHLO purely as an input format, and do all
  // optimizations via MHLO passes that can be shared with the OpenXLA compiler.
  // Therefore, this function inserts a StableHLO <=> MHLO roundtrip to make
  // this happen.

  AddMhloOptimizationPasses(pm, /*enable_stablehlo_quantizer=*/false);
  // TODO: b/293149194 - Add `createFoldBroadcastPass` back to
  // `AddMhloOptimizationPasses`
  pm.addNestedPass<func::FuncOp>(createFoldBroadcastPass());

  // MHLO -> StableHLO legalization.
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
}

}  // namespace odml
}  // namespace mlir
