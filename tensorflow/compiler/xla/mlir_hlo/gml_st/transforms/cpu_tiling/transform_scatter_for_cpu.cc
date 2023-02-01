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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMSCATTERFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kScatterTransformedLabel =
    "__scatter_transformed_label__";

struct TileScatterPattern : public OpRewritePattern<thlo::ScatterOp> {
  using OpRewritePattern<thlo::ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kScatterTransformedLabel)) return failure();

    if (isa<scf::ForOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(
          op, "has already been tiled by another pass.");
    }

    // Tile everything to points.
    scf::SCFTilingOptions opts;
    opts.setTileSizeComputationFunction([](OpBuilder &b, Operation *op) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(
          &op->getParentOfType<func::FuncOp>().getBody().front());

      auto loops = cast<TilingInterface>(op).getLoopIteratorTypes();
      return SmallVector<Value>(
          loops.size(), b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
    });

    auto tilingResult = scf::tileUsingSCFForOp(
        rewriter, cast<TilingInterface>(op.getOperation()), opts);
    if (failed(tilingResult)) return failure();

    // If we did not tile, do not replace original op and just mark it as
    // transformed then return.
    if (!tilingResult->loops.empty()) {
      rewriter.replaceOp(op, tilingResult->replacements);
    }
    setLabel(tilingResult->tiledOps.front(), kScatterTransformedLabel);
    return success();
  }
};

struct TransformScatterForCpuPass
    : public impl::TransformScatterForCpuPassBase<TransformScatterForCpuPass> {
  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, gml_st::GmlStDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<TileScatterPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();

    // Ensure we drop the marker in the end.
    f.walk([](thlo::ScatterOp scatterOp) {
      removeLabel(scatterOp, kScatterTransformedLabel);
    });
  }
};

}  // namespace
}  // namespace mlir::gml_st

namespace mlir::gml_st {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformScatterForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformScatterForCpuPass>();
}

}  // namespace mlir::gml_st
