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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_REWRITEFROMELEMENTSOPPASS
#include "gml_st/transforms/passes.h.inc"

// Rewrite `tensor.from_elements(x)` into `tensor.insert(x, tensor.empty)`.
// In combination with `empty-tensor-elimination` it removes the alloc that can
// result from `tensor.from_elements`.
struct RewriteFromElementsOpInDestinationPassingStyle
    : public OpRewritePattern<tensor::FromElementsOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::FromElementsOp op,
                                PatternRewriter &rewriter) const override {
    return linalg::rewriteInDestinationPassingStyle(rewriter, op);
  }
};

class RewriteFromElementsOpPass
    : public impl::RewriteFromElementsOpPassBase<RewriteFromElementsOpPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<RewriteFromElementsOpInDestinationPassingStyle>(context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createRewriteFromElementsOpPass() {
  return std::make_unique<RewriteFromElementsOpPass>();
}

}  // namespace gml_st
}  // namespace mlir
