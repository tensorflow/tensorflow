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
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/transforms/passes.h"

namespace mlir::quant::stablehlo {

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
  pm.addNestedPass<func::FuncOp>(createConvertMHLOQuantToIntPass(
      /*legalize_chlo=*/true));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(CreateOptimizeIntGraphPass());
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

}  // namespace mlir::quant::stablehlo
