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

#include "mlir/IR/Attributes.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/PatternMatch.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/StandardOps/Ops.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
namespace mlir {
namespace {
#include "tensorflow/compiler/mlir/tensorflow/transforms/generated_optimize.inc"
}  // namespace

/// Canonicalize operations in functions.
struct TFOptimizePass : public FunctionPass<TFOptimizePass> {
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    auto& func = getFunction();
    populateWithGenerated(&getContext(), &patterns);
    applyPatternsGreedily(func, std::move(patterns));
  }
};

FunctionPassBase* createTFOptimizePass() { return new TFOptimizePass(); }

static PassRegistration<TFOptimizePass> pass("tf-optimize", "Optimizes TF.");

}  // namespace mlir
