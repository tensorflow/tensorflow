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

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMATMULFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

FailureOr<TilingResult> tileMatmul(PatternRewriter &rewriter, Operation *op,
                                   ArrayRef<int64_t> tileSizes) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  return tileUsingGmlSt(opts, rewriter, cast<TilingInterface>(op));
}

/// Pattern to tile `linalg.matmul`, fuse `linalg.fill` into generated
/// `gml_st.parallel`, and peel the generated loops.
struct MatmulTransformPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  MatmulTransformPattern(MLIRContext *context,
                         MatmulTileSizeComputationFn tileSizeFn,
                         PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        tileSizeFn(std::move(tileSizeFn)) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(matmulOp, kTransformedLabel))
      return rewriter.notifyMatchFailure(matmulOp,
                                         "has already been transformed.");

    auto cluster = findMapFusionCluster(matmulOp);
    auto fusionCluster = cluster.operations;
    auto *tilingRoot = cluster.root;

    auto lhsTy = matmulOp.getOperandTypes()[0].cast<ShapedType>();
    auto resultTy = matmulOp.getResultTypes()[0].cast<ShapedType>();

    auto tileSize = tileSizeFn(
        {resultTy.getDimSize(0), resultTy.getDimSize(1), lhsTy.getDimSize(1)});

    // Tiling of linalg.map requires two dimensions, linalg.matmul requires
    // three.
    SmallVector<int64_t> parallelDimsTileSizes{tileSize.m, tileSize.n};
    if (isa<linalg::MatmulOp>(tilingRoot)) parallelDimsTileSizes.push_back(0);

    // First level tiling: parallel dimensions.
    auto tilingParallelDimsResult =
        tileMatmul(rewriter, tilingRoot, parallelDimsTileSizes);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingParallelDimsResult->loop != nullptr) {
      rewriter.replaceOp(tilingRoot,
                         tilingParallelDimsResult->loop->getResults());
      tilingRoot = tilingParallelDimsResult->tiledOps.front();

      // Fuse ops into the loop.
      fuseGreedily(rewriter, *tilingRoot->getBlock(),
                   [&](Operation *op) { return fusionCluster.contains(op); });
      (void)fuseFillOpsIntoForallOp(rewriter, tilingParallelDimsResult->loop);
    }

    // Second level tiling: reduction dimension for matmuls.
    SmallVector<scf::SCFTilingResult> tilingReductionDimsResults;
    for (auto op :
         llvm::to_vector(tilingRoot->getBlock()->getOps<linalg::MatmulOp>())) {
      auto result = tileMatmulReductionDims(rewriter, op, tileSize);
      if (failed(result)) return failure();
      tilingReductionDimsResults.push_back(*result);
    }

    // Peel parallel loops.
    //
    // We only want to peel (1) the parallel loop then (2) our kernel.
    auto peelingResult = peelAllLoops(tilingParallelDimsResult->loop, rewriter);

    // Peel reduction loop inside the main parallel loop, label the main loop as
    // "perfectly tiled" one, to enable vectorization after canonicalization.
    for (auto &res : tilingReductionDimsResults) {
      if (res.loops.size() == 1) {
        auto peelingResult = peelSCFForOp(rewriter, res.loops.front());
        setLabel(peelingResult.mainLoop, kPerfectlyTiledLoopLabel);
      }
    }
    return success();
  }

 private:
  FailureOr<scf::SCFTilingResult> tileMatmulReductionDims(
      PatternRewriter &rewriter, linalg::MatmulOp matmulOp,
      const MatmulSizes &tileSize) const {
    SmallVector<int64_t> reductionDimsTileSizes{0, 0, tileSize.k};
    scf::SCFTilingOptions opts;
    opts.setTileSizes(reductionDimsTileSizes);
    auto tilingReductionDimsResult =
        scf::tileUsingSCFForOp(rewriter, matmulOp.getOperation(), opts);
    if (failed(tilingReductionDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (!tilingReductionDimsResult->loops.empty()) {
      rewriter.replaceOp(matmulOp, tilingReductionDimsResult->replacements);
      matmulOp =
          cast<linalg::MatmulOp>(tilingReductionDimsResult->tiledOps.front());
    }

    setLabel(matmulOp, kTransformedLabel);
    return tilingReductionDimsResult;
  }

  MatmulTileSizeComputationFn tileSizeFn;
};

struct TransformMatmulForCpuPass
    : public impl::TransformMatmulForCpuPassBase<TransformMatmulForCpuPass> {
  TransformMatmulForCpuPass() = default;

  explicit TransformMatmulForCpuPass(MatmulTileSizeComputationFn tileSizeFn)
      : tileSizeFn(tileSizeFn ? std::move(tileSizeFn)
                              : [](MatmulSizes) -> MatmulSizes {
          return {4, 4, 4};
        }) {}

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulTransformPattern>(ctx, tileSizeFn);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();

    // Ensure we drop the marker in the end.
    f.walk([](linalg::MatmulOp op) { removeLabel(op, kTransformedLabel); });
  }

 private:
  MatmulTileSizeComputationFn tileSizeFn;
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass(MatmulTileSizeComputationFn tileSizeFn) {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>(
      std::move(tileSizeFn));
}

}  // namespace mlir::gml_st
