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

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassManager.h"  // TF:local_config_mlir
#include "mlir/Transforms/Passes.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TF {
namespace {

#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_optimize.inc"

// Canonicalize operations in functions.
struct TFOptimizePass : public FunctionPass<TFOptimizePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto func = getFunction();
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsGreedily(func, patterns);
  }
};

// NOLINTNEXTLINE - MLIR contract is pass by mutable reference.
void CreateTFStandardPipeline(OpPassManager &pm) {
  // First operates on the executor dialect:
  // - eliminate trivial switch/merge
  // - fuse islands as much as possible.
  // - materialize the eventual "pass-through" ops by inlining their content.
  pm.addPass(tf_executor::CreateSwitchFoldPass());
  pm.addPass(tf_executor::CreateTFExecutorIslandCoarseningPass());
  pm.addPass(CreateMaterializePassthroughOpPass());

  // Hopefully there is a single island left, or there wasn't any to begin with.
  // We now run the optimizer which operates mostly inside islands.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(CreateTFOptimizePass());
  pm.addPass(createCSEPass());
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateTFOptimizePass() {
  return std::make_unique<TFOptimizePass>();
}

static PassRegistration<TFOptimizePass> pass("tf-optimize", "Optimizes TF.");

// Registers a pipeline builder function for the default canonicalize/optimizer.
static mlir::PassPipelineRegistration<> pipeline(
    "tf-standard-pipeline",
    "Run all the passes involved in transforming/optimizing the graph after "
    "importing into MLIR, without any target specialization.",
    CreateTFStandardPipeline);

}  // namespace TF
}  // namespace mlir
