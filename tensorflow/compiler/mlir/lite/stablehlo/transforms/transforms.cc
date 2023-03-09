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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/drop_savedmodel_semantics.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/rename_entrypoint_to_main.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/smuggle_disallowed_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/tf_stablehlo_pass.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"

namespace mlir {
namespace odml {

void AddTFToStablehloPasses(OpPassManager& pm, bool skip_resize,
                            bool smuggle_disallowed_ops) {
  pm.addPass(CreateRenameEntrypointToMainPass());
  // TODO(b/230572023): Consider improving shape inference for While op instead
  // of dropping the attribute. This need not be correct for models not trained
  // on TPU.
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
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());
  pm.addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass(
      /*allow_mutable_tensors=*/true));
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::TF::CreateTFShapeInferencePass());
  pm.addNestedPass<func::FuncOp>(
      mlir::quant::CreateConvertTFQuantOpsToMHLOPass());
  pm.addPass(mlir::createCanonicalizerPass());
  AddLegalizeTFToStablehloPasses(pm, /*skip_quantization_ops=*/false,
                                 skip_resize);
  if (smuggle_disallowed_ops) {
    pm.addNestedPass<func::FuncOp>(CreateSmuggleDisallowedOpsPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }
  pm.addPass(CreateDropSavedModelSemanticsPass());
}

void AddStablehloOptimizationPasses(OpPassManager& pm) {
  // The current plan of record is to avoid doing optimization passes
  // on StableHLO, treating StableHLO purely as an input format, and do all
  // optimizations via MHLO passes that can be shared with the OpenXLA compiler.
  // Therefore, this function inserts a StableHLO <=> MHLO roundtrip to make
  // this happen.
  pm.addPass(mhlo::createStablehloLegalizeToHloPass());
  pm.addNestedPass<func::FuncOp>(createUnfuseBatchNormPass());
  pm.addNestedPass<func::FuncOp>(createFuseConvolutionPass());
  pm.addNestedPass<func::FuncOp>(createFoldBroadcastPass());
  pm.addNestedPass<func::FuncOp>(createOptimizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
}

}  // namespace odml
}  // namespace mlir
