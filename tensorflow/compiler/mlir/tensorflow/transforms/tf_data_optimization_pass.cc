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

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_data_optimization.h"

namespace mlir {
namespace TF {
namespace {

// Perform tf.data optimizations.
struct TFDataOptimization
    : public PassWrapper<TFDataOptimization, FunctionPass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    mlir::TF::PopulateTFDataOptimizationPatterns(&getContext(), &patterns);

    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

}  // namespace
}  // namespace TF
}  // namespace mlir

static mlir::PassRegistration<mlir::TF::TFDataOptimization> pass(
    "tf-data-optimization", "Performs tf.data optimizations");
