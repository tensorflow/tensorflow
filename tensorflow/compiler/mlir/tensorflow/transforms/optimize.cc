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
#include <iostream>

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_optimize.inc"

// Canonicalize operations in functions.
struct TFOptimizePass : public PassWrapper<TFOptimizePass, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsAndFoldGreedily(func, patterns);
  }
};

}  // namespace

// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager &pm,
                              const StandardPipelineOptions &options) {
  OpPassManager &func_pm = pm.nest<FuncOp>();

  // First operates on the executor dialect:
  // - eliminate trivial switch/merge.
  // - remove dead islands.
  // - fuse islands as much as possible.
  // - materialize the eventual "pass-through" ops by inlining their content.
  func_pm.addPass(tf_executor::CreateSwitchFoldPass());
  func_pm.addPass(tf_executor::CreateTFExecutorGraphPruningPass());
  func_pm.addPass(tf_executor::CreateTFExecutorIslandCoarseningPass());
  func_pm.addPass(CreateMaterializePassthroughOpPass());

  // Hopefully there is a single island left, or there wasn't any to begin with.
  // We now run the optimizer which operates mostly inside islands.
  func_pm.addPass(createCanonicalizerPass());
  if (options.enable_inliner) {
    pm.addPass(createInlinerPass());
  }
  pm.addPass(createSymbolDCEPass());
  pm.addPass(CreateTFShapeInferencePass());
  pm.addNestedPass<FuncOp>(CreateTFOptimizePass());
  pm.addNestedPass<FuncOp>(createCSEPass());
}

std::unique_ptr<OperationPass<FuncOp>> CreateTFOptimizePass() {
  return std::make_unique<TFOptimizePass>();
}

static PassRegistration<TFOptimizePass> pass("tf-optimize", "Optimizes TF.");

// Registers a pipeline builder function for the default canonicalize/optimizer.
static mlir::PassPipelineRegistration<StandardPipelineOptions> pipeline(
    "tf-standard-pipeline",
    "Run all the passes involved in transforming/optimizing the graph after "
    "importing into MLIR, without any target specialization.",
    CreateTFStandardPipeline);

}  // namespace TF
}  // namespace mlir
