/* Copyright 2021 The OpenXLA Authors.

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

// This file contains the patterns to convert arith.index_cast on tensors to
// tensor ops and index_cast on scalars.

#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_LOWERINDEXCASTPASS
#include "transforms/passes.h.inc"

namespace {

// index_cast is not defined on tensors, so lower it to a tensor.generate.
template <typename T>
struct IndexCastConverter : public OpRewritePattern<T> {
 public:
  using OpRewritePattern<T>::OpRewritePattern;
  LogicalResult matchAndRewrite(T op, PatternRewriter &rewriter) const final {
    auto resultTy = mlir::dyn_cast<RankedTensorType>(op.getType());
    if (!resultTy) return failure();

    SmallVector<Value> dynamicExtents =
        tensor::createDynamicDimValues(rewriter, op.getLoc(), op.getIn());
    rewriter.replaceOpWithNewOp<tensor::GenerateOp>(
        op, resultTy, dynamicExtents,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value extent = b.create<tensor::ExtractOp>(loc, op.getIn(), args);
          Value cast = b.create<T>(loc, resultTy.getElementType(), extent);
          b.create<tensor::YieldOp>(loc, cast);
        });
    return success();
  }
};

struct LowerIndexCastPass
    : public impl::LowerIndexCastPassBase<LowerIndexCastPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<IndexCastConverter<arith::IndexCastOp>,
                 IndexCastConverter<arith::IndexCastUIOp>>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLowerIndexCastPass() {
  return std::make_unique<LowerIndexCastPass>();
}

}  // namespace mlir
