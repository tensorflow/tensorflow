/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/transforms/mlprogram.h"

#include <string>
#include <utility>

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/Twine.h"
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/bridge.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_saved_model_passes.h"
#include "tensorflow/compiler/mlir/tf2xla/api/v0/compile_mlir_util.h"
#include "tensorflow/compiler/mlir/tf2xla/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/transforms/passes.h"

namespace tensorflow {

void PopulateLowerToMlProgramAndHloPipeline(mlir::OpPassManager& pm) {
  mlir::TF::CreateTFXLABridgePipeline(pm);

  // Remove unused global tensors, or make then immutable if possible.
  pm.addPass(mlir::tf_saved_model::CreateOptimizeGlobalTensorsPass());

  pm.addPass(
      mlir::tf_saved_model::CreateConvertSessionInitializerToFunctionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TFDevice::CreateDecomposeResourceOpsPass());
  pm.addPass(mlir::TF::CreateNameAnonymousIteratorsPass());

  // This will add regions to IfOp/WhileOp (turning them into IfRegionOp
  // and WhileRegionOp), but be aware that those regions will still contain
  // calls.
  pm.addPass(mlir::TF::CreateTFFunctionalControlFlowToRegions());

  pm.addPass(mlir::tf_saved_model::CreateLowerVariableOpsToMlProgramPass());
  pm.addPass(mlir::tf_saved_model::CreateLowerGlobalsToMlProgramPass());
  pm.addPass(mlir::TF::CreateLocalizeVarHandlesPass());
  pm.addPass(mlir::tf_saved_model::CreateAddFunctionsForExportedNamesPass());
  pm.addPass(mlir::tf_saved_model::CreateStripSavedModuleMetadataPass());

  pm.addPass(mlir::TF::CreateRemoveUnusedArgumentsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::TF::CreateRemoveUnusedWhileResultsPass());

  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSymbolDCEPass());

  pm.addPass(mlir::TF::CreateTFShapeInferencePass());

  llvm::StringRef tf2xla_fallback_device_type = "XLA_CPU_JIT";
  pm.addPass(mlir::mhlo::createLegalizeTFPass(
      /*allow_partial_conversion=*/true, /*legalize_chlo=*/true,
      tf2xla_fallback_device_type, /*prefer_tf2xla=*/false));

  pm.addPass(mlir::TF::CreateStripTfAttributesPass());

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(mlir::createSymbolDCEPass());
  pm.addPass(mlir::createCanonicalizerPass());

  pm.addPass(mlir::TF::CreateOrderByDialectPass());

  pm.addPass(mlir::mhlo::createHloLegalizeToStablehloPass());
}

}  // namespace tensorflow
