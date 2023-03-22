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

#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // from @llvm-project
#include "mlir/Dialect/X86Vector/X86VectorDialect.h"  // from @llvm-project
#include "tensorflow/compiler/xla/mlir/math/transforms/passes.h"

namespace xla {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_MATHOPTIMIZATIONPASS
#include "tensorflow/compiler/xla/mlir/math/transforms/passes.h.inc"

struct MathOptimizationPass
    : public impl::MathOptimizationPassBase<MathOptimizationPass> {
  explicit MathOptimizationPass(bool enable_avx2) {
    enable_avx2_ = enable_avx2;
  }
  void runOnOperation() override;
};

void MathOptimizationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateMathAlgebraicSimplificationPatterns(patterns);

  MathPolynomialApproximationOptions approx_options;
  approx_options.enableAvx2 = enable_avx2_;
  populateMathPolynomialApproximationPatterns(patterns, approx_options);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<OperationPass<func::FuncOp>> CreateMathOptimizationPass(
    bool enable_avx2) {
  return std::make_unique<MathOptimizationPass>(enable_avx2);
}

}  // namespace xla
