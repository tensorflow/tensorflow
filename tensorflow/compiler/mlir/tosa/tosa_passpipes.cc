/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tosa/tosa_passpipes.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {

namespace tosa {

void addPreOptMlirPasses(mlir::OpPassManager& pm) {
  // Inline all functions into main and then delete the functions themselves.
  pm.addPass(mlir::createInlinerPass());

  // Now that there is only one function, run some MLIR passes on it.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::createLoopFusionPass());
  pm.addPass(mlir::createMemRefDataFlowOptPass());
}

void addPostOptMlirPasses(mlir::OpPassManager& pm) {
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  // Inline the call/return basic blocks within TOSA control flow ops.
  pm.addPass(mlir::createInlinerPass());
  // Clean up with DCE.
  pm.addPass(mlir::createSymbolDCEPass());
}

void createTFtoTOSALegalizationPipeline(
    OpPassManager& pm, const TOSALegalizationPipelineOptions& opts) {
  addPreOptMlirPasses(pm);

  pm.addPass(mlir::tosa::createFuseBiasTFPass());
  pm.addPass(mlir::tosa::createLegalizeTFPass());

  addPostOptMlirPasses(pm);
}

void createTFLtoTOSALegalizationPipeline(
    OpPassManager& pm, const TOSALegalizationPipelineOptions& opts) {
  addPreOptMlirPasses(pm);

  pm.addPass(mlir::tosa::createConvertTFLUint8Pass());
  pm.addPass(mlir::tosa::createLegalizeTFLPass());

  addPostOptMlirPasses(pm);
}

}  // namespace tosa

}  // namespace mlir
