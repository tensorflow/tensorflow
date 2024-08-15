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

#include "tensorflow/compiler/mlir/tosa/tfl_passes.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"  // from @llvm-project
#include "mlir/Dialect/Affine/Passes.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/Transforms/Passes.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

namespace mlir {
namespace tosa {

void createTFLtoTOSALegalizationPipeline(
    OpPassManager& pm, const TOSATFLLegalizationPipelineOptions& opts) {
  //----------------------------------------------------------------------------
  // Prepare TFL module for conversion
  //----------------------------------------------------------------------------
  pm.addPass(createRetainCallOnceFuncsPass());

  // Inline all functions into main and then delete the functions themselves.
  pm.addPass(mlir::createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
  if (opts.target_compilation_backend) {
    pm.nest<func::FuncOp>().addPass(createConvertFunctionMetadataPass());
  }

  // Add pass to decompose TFLite mixed quantization to non-quantized variants.
  pm.addPass(TFL::CreateDecomposeHybridQuantizationPass());

  // Now that there is only one function, run some MLIR passes on it.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  pm.addPass(mlir::affine::createLoopFusionPass());
  pm.addPass(mlir::affine::createAffineScalarReplacementPass());

  //----------------------------------------------------------------------------
  // Perform main conversion.
  //----------------------------------------------------------------------------
  pm.addPass(mlir::tosa::createConvertTFLUint8Pass());
  if (opts.dequantize_tfl_softmax) {
    pm.addPass(mlir::tosa::createDequantizeTFLSoftmaxPass());
  }
  pm.addPass(mlir::tosa::createLegalizeTFLStatefulPass());
  pm.addPass(mlir::tosa::createLegalizeTFLPass(opts.disabled_patterns,
                                               opts.enabled_patterns));

  //----------------------------------------------------------------------------
  // Post conversion cleanup.
  //----------------------------------------------------------------------------
  pm.addPass(mlir::tosa::createLowerComplexTypesPass());
  pm.addPass(mlir::tosa::createTosaInferShapesPass());
  pm.addPass(mlir::tosa::createTosaMakeBroadcastablePass());
  // Inline the call/return basic blocks within TOSA control flow ops.
  pm.addPass(mlir::createInlinerPass());
  // Clean up with DCE.
  pm.addPass(mlir::createSymbolDCEPass());

  if (opts.target_compilation_backend) {
    pm.nest<func::FuncOp>().addPass(mlir::tosa::createStripQuantTypesPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.nest<func::FuncOp>().addPass(createStripFunctionMetadataPass());
    pm.addPass(createStripModuleMetadataPass());
    pm.addPass(createVerifyFullyConvertedPass());
  }
}

void registerTFLtoTOSALegalizationPipeline() {
  mlir::PassPipelineRegistration<TOSATFLLegalizationPipelineOptions>(
      "tfl-to-tosa-pipeline", "TensorFlow Lite to TOSA legalization pipeline",
      createTFLtoTOSALegalizationPipeline);
}

}  // namespace tosa
}  // namespace mlir
