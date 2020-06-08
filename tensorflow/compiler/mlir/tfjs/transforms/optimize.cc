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

// This transformation pass takes operations in TensorFlow dialect and
// optimizes them to resulting operations in TensorFlow.js dialect.

#include <memory>

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tfjs/ir/tfjs_ops.h"

namespace mlir {
namespace tfjs {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {

// Optimize TFJS operations in functions.
struct Optimize : public PassWrapper<Optimize, FunctionPass> {
  void runOnFunction() override;
};

#include "tensorflow/compiler/mlir/tfjs/transforms/generated_optimize.inc"

void Optimize::runOnFunction() {
  OwningRewritePatternList patterns;
  auto *ctx = &getContext();
  auto func = getFunction();

  populateWithGenerated(ctx, &patterns);
  applyPatternsAndFoldGreedily(func, patterns);
}
}  // namespace

// Creates an instance of the TensorFlow.js dialect Optimize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizePass() {
  return std::make_unique<Optimize>();
}

static PassRegistration<Optimize> pass(
    "tfjs-optimize", "Optimize within the TensorFlow.js dialect");

}  // namespace tfjs
}  // namespace mlir
