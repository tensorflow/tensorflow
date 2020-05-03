/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfjs/tf_tfjs_passes.h"

#include <memory>

#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/decode_constant.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tfjs/transforms/passes.h"

namespace mlir {
/// Create a pass to convert from the TFExecutor to the TF control dialect.
std::unique_ptr<OperationPass<FuncOp>>
CreateTFExecutorToControlDialectConversion();
}  // namespace mlir

namespace tensorflow {

void AddTFToTFJSConversionPasses(mlir::OpPassManager* pm) {
  // Then we pass the MLIR module through the TF standard pipeline, which for
  mlir::TF::StandardPipelineOptions tf_options;
  tf_options.enable_inliner = true;
  mlir::TF::CreateTFStandardPipeline(*pm, tf_options);

  // freeze global tensors.
  pm->addPass(mlir::tf_saved_model::CreateFreezeGlobalTensorsPass());

  // TFJS dialect passes.
  pm->addPass(mlir::tfjs::CreateOptimizePass());

  // Canonicalize, CSE etc.
  pm->addNestedPass<mlir::FuncOp>(mlir::createCanonicalizerPass());
  pm->addNestedPass<mlir::FuncOp>(mlir::createCSEPass());
}

}  // namespace tensorflow
