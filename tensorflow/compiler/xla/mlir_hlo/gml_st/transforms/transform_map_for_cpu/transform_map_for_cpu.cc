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
#include "gml_st/interfaces/tiling_interface_impl.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
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

template <typename OpType>
struct TileMapPattern : public OpRewritePattern<OpType> {
  TileMapPattern(MLIRContext *context, TilingOptions options,
                 PatternBenefit benefit = 1)
      : OpRewritePattern<OpType>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(OpType op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kMapTransformedLabel)) return failure();

    if (isa<gml_st::ParallelOp, gml_st::ForOp>(op->getParentOp()))
      return rewriter.notifyMatchFailure(
          op, "has already been tiled by another pass.");

    auto fuseFilterFn = [](Operation *op) {
      return isa<linalg::BroadcastOp, OpType>(op);
    };

    // Find there another linalg.map where this op can be fused.
    op = findRootMap(op, fuseFilterFn);

    if (hasLabel(op, kMapTransformedLabel)) return failure();

    auto tilingResult =
        tile(options, rewriter, cast<TilingInterface>(op.getOperation()));
    if (failed(tilingResult)) return failure();

    // If we did not tile (e.g. when all tile sizes are 0), do not replace
    // original op and just mark it as transformed then return.
    if (tilingResult->loop != nullptr) {
      rewriter.replaceOp(op, tilingResult->loop->getResults());

      // Fuse ops into the loop.
      fuseGreedily(rewriter, *tilingResult->tiledOps.front()->getBlock(),
                   fuseFilterFn);
    }
    setLabel(tilingResult->tiledOps.front(), kMapTransformedLabel);

    // Peel parallel loops.
    if (auto loop = dyn_cast_or_null<ParallelOp>(tilingResult->loop)) {
      peelAllLoops(loop, rewriter);
    }

    return success();
  }

 private:
  // Find the root of the fusion cluster.
  OpType findRootMap(OpType op,
                     llvm::function_ref<bool(Operation *)> fuseFilterFn) const {
    OpType rootMap = op;

    Operation *curOp = op;
    while (fuseFilterFn(curOp)) {
      auto users = llvm::to_vector(curOp->getUsers());
      // The op has more than 1 user. It will no be fused.
      if (users.size() != 1) break;
      curOp = users[0];

      if (auto curMap = dyn_cast<OpType>(curOp)) rootMap = curMap;
    }
    return rootMap;
  }

  TilingOptions options;
};

struct TransformMapForCpuPass
    : public impl::TransformMapForCpuPassBase<TransformMapForCpuPass> {
  explicit TransformMapForCpuPass(int64_t ts) { tileSize = ts; }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
    mlir::gml_st::registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *context = &getContext();

    mlir::gml_st::TilingOptions opts;

    opts.tileSizeComputationFn = [&](OpBuilder &b, Operation *op) {
      assert(isa<linalg::MapOp>(op) ||
             isa<linalg::FillOp>(op) &&
                 " only linalg.map or linalg.fill expected");
      auto numLoops = isa<linalg::MapOp>(op)
                          ? cast<linalg::MapOp>(op).getNumLoops()
                          : cast<linalg::FillOp>(op).getNumLoops();
      SmallVector<Value> tiles(
          numLoops, b.create<arith::ConstantIndexOp>(op->getLoc(), 1));
      if (!tiles.empty())
        tiles.back() = b.create<arith::ConstantIndexOp>(op->getLoc(), tileSize);
      return tiles;
    };

    RewritePatternSet patterns(context);
    patterns.add<TileMapPattern<linalg::MapOp>, TileMapPattern<linalg::FillOp>>(
        context, opts);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    f.walk([](Operation *op) {
      if (isa<linalg::MapOp, linalg::FillOp>(op))
        removeLabel(op, kMapTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMapForCpuPass(int64_t tileSize) {
  return std::make_unique<mlir::gml_st::TransformMapForCpuPass>(tileSize);
}

}  // namespace mlir::gml_st
