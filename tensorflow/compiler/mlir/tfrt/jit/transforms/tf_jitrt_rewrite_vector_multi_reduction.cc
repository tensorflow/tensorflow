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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::MLIRContext;
using mlir::Operation;
using mlir::vector::MultiDimReductionOp;
using mlir::vector::VectorMultiReductionLowering;

struct RewriteVectorMultiReductionPass
    : public RewriteVectorMultiReductionPassBase<
          RewriteVectorMultiReductionPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    Operation* op = getOperation();
    if (failed(RewriteTwoAndMoreDimReductions(ctx, op))) signalPassFailure();
    if (failed(RewriteOneDimReductions(ctx, op))) signalPassFailure();
  }

  // Rewrite N-D reductions as the sequence of vector operations without
  // horizontal reduction, i.e. `vector.reduction`.
  mlir::LogicalResult RewriteTwoAndMoreDimReductions(MLIRContext* ctx,
                                                     Operation* op) const {
    mlir::ConversionTarget target(*ctx);
    target.addLegalDialect<mlir::arith::ArithmeticDialect,
                           mlir::vector::VectorDialect>();
    target.addDynamicallyLegalOp<MultiDimReductionOp>(
        [&](MultiDimReductionOp op) {
          return op.getSourceVectorType().getRank() == 1;
        });

    mlir::RewritePatternSet patterns(ctx);
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns, VectorMultiReductionLowering::InnerParallel);
    return applyPartialConversion(op, target, std::move(patterns));
  }

  // Rewrite 1D reductions as a `vector.reduction`.
  mlir::LogicalResult RewriteOneDimReductions(MLIRContext* ctx,
                                              Operation* op) const {
    mlir::RewritePatternSet patterns(ctx);
    mlir::vector::populateVectorMultiReductionLoweringPatterns(
        patterns, VectorMultiReductionLowering::InnerReduction);
    return applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRewriteVectorMultiReductionPass() {
  return std::make_unique<RewriteVectorMultiReductionPass>();
}

}  // namespace tensorflow
