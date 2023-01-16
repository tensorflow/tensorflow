/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_REWRITEVECTORMULTIREDUCTIONPASS
#include "gml_st/transforms/passes.h.inc"

struct RewriteVectorMultiReductionPass
    : public impl::RewriteVectorMultiReductionPassBase<
          RewriteVectorMultiReductionPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    Operation* op = getOperation();
    if (failed(rewriteTwoAndMoreDimReductions(ctx, op))) signalPassFailure();
    if (failed(rewriteOneDimReductions(ctx, op))) signalPassFailure();
  }

  // Rewrite N-D reductions as the sequence of vector operations without
  // horizontal reduction, i.e. `vector.reduction`.
  LogicalResult rewriteTwoAndMoreDimReductions(MLIRContext* ctx,
                                               Operation* op) const {
    ConversionTarget target(*ctx);
    target.addLegalDialect<arith::ArithDialect, vector::VectorDialect>();
    target.addDynamicallyLegalOp<vector::MultiDimReductionOp>(
        [&](vector::MultiDimReductionOp op) {
          return op.getSourceVectorType().getRank() == 1;
        });

    RewritePatternSet patterns(ctx);
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vector::VectorMultiReductionLowering::InnerParallel);
    return applyPartialConversion(op, target, std::move(patterns));
  }

  // Rewrite 1D reductions as a `vector.reduction`.
  LogicalResult rewriteOneDimReductions(MLIRContext* ctx, Operation* op) const {
    RewritePatternSet patterns(ctx);
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vector::VectorMultiReductionLowering::InnerReduction);
    return applyPatternsAndFoldGreedily(op, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteVectorMultiReductionPass() {
  return std::make_unique<RewriteVectorMultiReductionPass>();
}

}  // namespace gml_st
}  // namespace mlir
