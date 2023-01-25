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
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMTRANSPOSEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

using mlir::arith::ConstantIndexOp;

static constexpr llvm::StringRef kTransposeTransformedLabel =
    "__transpose_transformed_label__";

struct TileTransposePattern : public OpRewritePattern<linalg::TransposeOp> {
  TileTransposePattern(MLIRContext *context, TilingOptions options,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::TransposeOp>(context, benefit),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(linalg::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kTransposeTransformedLabel)) return failure();

    if (isa<LoopLikeOpInterface>(op->getParentOp()))
      return rewriter.notifyMatchFailure(
          op, "has already been tiled by another pass.");

    auto tilingResult = tileUsingGmlSt(
        options, rewriter, cast<TilingInterface>(op.getOperation()));
    if (failed(tilingResult)) return failure();

    // If we did not tile (e.g. when all tile sizes are 0), do not replace
    // original op and just mark it as transformed then return.
    if (tilingResult->loop != nullptr) {
      rewriter.replaceOp(op, tilingResult->loop->getResults());
    }
    setLabel(tilingResult->tiledOps.front(), kTransposeTransformedLabel);

    // Peel parallel loops, label the main loop as "perfectly tiled" one, to
    // enable vectorization after canonicalization.
    if (auto loop = dyn_cast_or_null<ParallelOp>(tilingResult->loop)) {
      auto peelingResult = peelAllLoops(loop, rewriter);
      setLabel(loop, kPerfectlyTiledLoopLabel);

      // Tile ops in the peeled loop again, to size 1, so they can be
      // scalarized.
      if (failed(tilePeeledOpsToScalars(rewriter, peelingResult,
                                        kTransposeTransformedLabel,
                                        /*fuseFilterFn=*/nullptr)))
        return failure();
    }

    return success();
  }

 private:
  TilingOptions options;
};

struct TransformTransposeForCpuPass
    : public impl::TransformTransposeForCpuPassBase<
          TransformTransposeForCpuPass> {
  TransformTransposeForCpuPass() = default;
  explicit TransformTransposeForCpuPass(
      llvm::ArrayRef<int64_t> transposeTileSizes) {
    tileSizes = transposeTileSizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect, arith::ArithDialect, linalg::LinalgDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    auto getTileSize = [&](mlir::OpBuilder b, Operation *op) {
      SmallVector<Value> tiles;
      for (int64_t tileSize : tileSizes) {
        tiles.push_back(b.create<ConstantIndexOp>(op->getLoc(), tileSize));
      }
      if (!tiles.empty()) return tiles;
      auto transposeOp = llvm::cast<linalg::TransposeOp>(op);
      unsigned numLoops = transposeOp.getNumLoops();
      assert(numLoops >= 2 && "Expect two or more dimension in transpose op");

      // Compute the tile sizes for the 2-D vectorization of the transpose. We
      // pick eight as default vectorization factor for both dimensions since
      // it's the most performant AVX2 pattern for now. We pick the contiguous
      // dimension of the input as first vector dimension and the contiguous
      // dimension of the output as second vector dimension. This will maximize
      // contiguous vector loads/stores and minimize insert/extract/gather/
      // scatter operations.
      tiles.resize(numLoops, b.create<ConstantIndexOp>(op->getLoc(), 1));
      auto indexingMaps = transposeOp.getIndexingMapsArray();
      unsigned lastDim = numLoops - 1;
      unsigned vecFactor0 = 8, vecFactor1 = 8;
      unsigned vecDim0 = indexingMaps[0].getDimPosition(lastDim);
      unsigned vecDim1 = indexingMaps[1].getDimPosition(lastDim);

      // If the contiguous dimensions of both input and output are not
      // transposed (i.e, they are the same), we vectorize only that dimension.
      // That transpose case doesn't require intra-register transposition but
      // just copying a set of contiguous sub-buffers from the input to the
      // output tensor. Vectorizing a second dimension would increase too much
      // the memory pressure for no reason.
      if (vecDim0 == vecDim1) {
        tiles[vecDim0] = b.create<ConstantIndexOp>(op->getLoc(), vecFactor0);
      } else {
        tiles[vecDim0] = b.create<ConstantIndexOp>(op->getLoc(), vecFactor0);
        tiles[vecDim1] = b.create<ConstantIndexOp>(op->getLoc(), vecFactor1);
      }

      return tiles;
    };

    TilingOptions tilingOptions;
    tilingOptions.tileSizeComputationFn = getTileSize;

    auto func = getOperation();
    RewritePatternSet patterns(func.getContext());
    patterns.add<TileTransposePattern>(patterns.getContext(), tilingOptions);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    func.walk([](linalg::TransposeOp op) {
      removeLabel(op, kTransposeTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformTransposeForCpuPass(llvm::ArrayRef<int64_t> transposeTileSizes) {
  return std::make_unique<TransformTransposeForCpuPass>(transposeTileSizes);
}

}  // namespace mlir::gml_st
