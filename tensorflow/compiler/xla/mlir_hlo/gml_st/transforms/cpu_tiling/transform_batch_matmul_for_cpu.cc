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

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMBATCHMATMULFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

using tensor::CollapseShapeOp;
using tensor::ExpandShapeOp;

void rewriteBatchMatmulAsMatmul(linalg::BatchMatmulOp bmOp,
                                PatternRewriter &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(bmOp);
  Value lhs = bmOp.getInputs()[0];
  Value rhs = bmOp.getInputs()[1];
  Value init = bmOp.getOutputs()[0];

  Location loc = bmOp.getLoc();
  SmallVector<ReassociationIndices> map{{0, 1}, {2}};
  Value newLhs = rewriter.create<CollapseShapeOp>(loc, lhs, map);
  Value newRhs = rewriter.create<CollapseShapeOp>(loc, rhs, map);
  Value newInit;
  if (auto fillOp = init.getDefiningOp<linalg::FillOp>()) {
    Value collapsedInit =
        rewriter.create<CollapseShapeOp>(loc, fillOp.getOutputs().front(), map);
    newInit = rewriter
                  .create<linalg::FillOp>(loc, fillOp.getInputs(),
                                          ValueRange{collapsedInit})
                  .getResult(0);
  } else {
    newInit = rewriter.create<CollapseShapeOp>(loc, init, map);
  }

  auto matmul = rewriter.create<linalg::MatmulOp>(
      loc, newInit.getType(), ValueRange{newLhs, newRhs}, ValueRange{newInit});

  rewriter.replaceOpWithNewOp<ExpandShapeOp>(bmOp, bmOp.getType(0),
                                             matmul.getResult(0), map);
}

// Tile linalg.batch_matmul to 1 in the outermost dimension, then transform a
// unit linalg.batch_matmul into a matmul using reshape ops.
struct BatchMatmulOpTransformPattern
    : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp bmOp,
                                PatternRewriter &rewriter) const override {
    // Tile and fuse fillOp into the loop nest.
    auto tilingResult = tileUsingSCFForallOpAndFuseGreedily(
        rewriter, bmOp.getOperation(), getSCFTilingOptions({1, 0, 0, 0}));
    if (failed(tilingResult)) return failure();

    auto tiledBm = cast<linalg::BatchMatmulOp>(tilingResult->tiledOps.front());
    rewriteBatchMatmulAsMatmul(tiledBm, rewriter);
    return success();
  }
};

struct TransformBatchMatmulForCpuPass
    : public impl::TransformBatchMatmulForCpuPassBase<
          TransformBatchMatmulForCpuPass> {
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<BatchMatmulOpTransformPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformBatchMatmulForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformBatchMatmulForCpuPass>();
}

}  // namespace mlir::gml_st
