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

#include <memory>
#include <utility>

#include "mlir/Dialect/Vector/VectorRewritePatterns.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

struct RewriteVectorMultiReductionPass
    : public RewriteVectorMultiReductionPassBase<
          RewriteVectorMultiReductionPass> {
  void runOnFunction() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns, mlir::vector::VectorMultiReductionLowering::InnerReduction);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> createRewriteVectorMultiReductionPass() {
  return std::make_unique<RewriteVectorMultiReductionPass>();
}

}  // namespace tensorflow
