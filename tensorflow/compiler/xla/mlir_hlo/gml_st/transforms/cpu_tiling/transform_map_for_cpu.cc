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
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMAPFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kMapTransformedLabel =
    "__map_transformed_label__";

struct TileMapPattern : public OpRewritePattern<linalg::MapOp> {
  TileMapPattern(MLIRContext *context, int64_t innerDimTileSize,
                 PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MapOp>(context, benefit),
        innerDimTileSize(innerDimTileSize) {}

  LogicalResult matchAndRewrite(linalg::MapOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kMapTransformedLabel)) return failure();

    if (isa<scf::ForallOp, scf::ForOp>(op->getParentOp())) {
      return rewriter.notifyMatchFailure(
          op, "has already been tiled by another pass.");
    }

    auto fuseFilterFn = [](Operation *op) {
      return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp>(op);
    };

    // Find there another linalg.map where this op can be fused.
    op = findRootMap(op, fuseFilterFn);

    if (hasLabel(op, kMapTransformedLabel)) return failure();

    mlir::gml_st::TilingOptions opts;
    opts.tileSizeComputationFn = [&](OpBuilder &b, Operation *op) {
      auto numLoops = cast<linalg::MapOp>(op).getNumLoops();
      SmallVector<Value> tiles(
          numLoops, b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
      if (!tiles.empty())
        tiles.back() =
            b.create<arith::ConstantIndexOp>(op->getLoc(), innerDimTileSize);
      return tiles;
    };

    auto tiledLoop = tileUsingGmlStParallelAndFuseGreedily(
        rewriter, op, opts, kMapTransformedLabel, fuseFilterFn);
    if (failed(tiledLoop)) return failure();

    // Peel parallel loops.
    auto peelingResult = peelAllLoops(*tiledLoop, rewriter);
    setLabel(*tiledLoop, kPerfectlyTiledLoopLabel);

    // Tile ops in the peeled loop again, to size 1, so they can be
    // scalarized.
    if (failed(tilePeeledOpsToScalars(rewriter, peelingResult,
                                      kMapTransformedLabel, fuseFilterFn)))
      return failure();

    return success();
  }

 private:
  // Find the root of the fusion cluster.
  linalg::MapOp findRootMap(
      linalg::MapOp op,
      llvm::function_ref<bool(Operation *)> fuseFilterFn) const {
    linalg::MapOp rootMap = op;

    Operation *curOp = op;
    while (fuseFilterFn(curOp)) {
      auto users = llvm::to_vector(curOp->getUsers());
      // The op has more than 1 user. It will no be fused.
      if (users.size() != 1) break;
      curOp = users[0];

      if (auto curMap = dyn_cast<linalg::MapOp>(curOp)) rootMap = curMap;
    }
    return rootMap;
  }

  int64_t innerDimTileSize;
};

struct TransformMapForCpuPass
    : public impl::TransformMapForCpuPassBase<TransformMapForCpuPass> {
  explicit TransformMapForCpuPass(int64_t ts) { tileSize = ts; }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<TileMapPattern>(context, tileSize);
    populateCollapseForallOpDimensionsPattern(patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    f.walk([](linalg::MapOp op) { removeLabel(op, kMapTransformedLabel); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMapForCpuPass(int64_t tileSize) {
  return std::make_unique<mlir::gml_st::TransformMapForCpuPass>(tileSize);
}

}  // namespace mlir::gml_st
