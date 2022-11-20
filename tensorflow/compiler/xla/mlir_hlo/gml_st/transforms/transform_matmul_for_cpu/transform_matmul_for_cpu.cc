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
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMATMULFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

FailureOr<TilingResult> tileMatmul(PatternRewriter &rewriter,
                                   linalg::MatmulOp matmulOp,
                                   ArrayRef<int64_t> tileSizes,
                                   bool distribute) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  opts.distribute = distribute;
  return tile(opts, rewriter, cast<TilingInterface>(matmulOp.getOperation()));
}

/// Pattern to tile `linalg.matmul` and fuse `linalg.fill` into generated
/// `gml_st.parallel`.
struct MatmulTransformPattern : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  explicit MatmulTransformPattern(MLIRContext *context,
                                  int64_t lhsParallelDimTileSize = 2,
                                  int64_t rhsParallelDimTileSize = 4,
                                  int64_t reductionDimTileSize = 8,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern<linalg::MatmulOp>(context, benefit),
        lhsParallelDimTileSize(lhsParallelDimTileSize),
        rhsParallelDimTileSize(rhsParallelDimTileSize),
        reductionDimTileSize(reductionDimTileSize) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    if (hasTransformationAttr(matmulOp))
      return rewriter.notifyMatchFailure(matmulOp,
                                         "has already been transformed.");

    // First level tiling: parallel dimensions.
    SmallVector<int64_t> parallelDimsTileSizes{lhsParallelDimTileSize,
                                               rhsParallelDimTileSize, 0};
    auto tilingParallelDimsResult = tileMatmul(
        rewriter, matmulOp, parallelDimsTileSizes, /*distribute=*/true);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling occurred.
    if (tilingParallelDimsResult->loop != nullptr) {
      rewriter.replaceOp(matmulOp,
                         tilingParallelDimsResult->loop->getResults());
      matmulOp = cast<linalg::MatmulOp>(tilingParallelDimsResult->tiledOp);
    }

    // Fusion into the output.
    OpOperand *matmulOutput =
        cast<linalg::MatmulOp>(matmulOp.getOperation()).getDpsInitOperand(0);
    auto materialize = matmulOutput->get().getDefiningOp<MaterializeOp>();
    if (!materialize) return failure();
    if (materialize.getSource().getDefiningOp<linalg::FillOp>()) {
      if (failed(fuse(rewriter, materialize))) return failure();
    }

    // Second level tiling: reduction dimension.
    SmallVector<int64_t> reductionDimsTileSizes{0, 0, reductionDimTileSize};
    auto tilingReductionDimsResult = tileMatmul(
        rewriter, matmulOp, reductionDimsTileSizes, /*distribute=*/false);
    if (failed(tilingReductionDimsResult)) return failure();

    // Update the results if tiling succeeded.
    if (tilingReductionDimsResult->loop != nullptr) {
      rewriter.replaceOp(matmulOp,
                         tilingReductionDimsResult->loop->getResults());
      matmulOp = cast<linalg::MatmulOp>(tilingReductionDimsResult->tiledOp);
    }

    setTransformationAttr(rewriter, matmulOp);

    // Peel parallel loops.
    if (auto loop =
            dyn_cast_or_null<ParallelOp>(tilingParallelDimsResult->loop)) {
      auto peelingResult = peelAllLoops(loop, rewriter);
      // Mark all for loops inside remainder parallel loops as peeled to prevent
      // downstream peeling pass from peeling them.
      for (auto *remParLoop : peelingResult) {
        remParLoop->walk([&](Operation *childOp) {
          if (isa<ForOp>(childOp)) {
            setTransformationAttr(rewriter, childOp, kPeeledMarker);
          }
        });
      }
    }

    // Peel reduction loop inside the main parallel loop.
    if (auto loop = dyn_cast_or_null<ForOp>(tilingReductionDimsResult->loop)) {
      peelAllLoops(loop, rewriter);
    }

    return success();
  }

 private:
  int64_t lhsParallelDimTileSize;
  int64_t rhsParallelDimTileSize;
  int64_t reductionDimTileSize;
};

struct TransformMatmulForCpuPass
    : public impl::TransformMatmulForCpuPassBase<TransformMatmulForCpuPass> {
  TransformMatmulForCpuPass() = default;

  explicit TransformMatmulForCpuPass(llvm::ArrayRef<int64_t> matmulTileSizes) {
    tileSizes = matmulTileSizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, tensor::TensorDialect>();
    mlir::gml_st::registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    if (tileSizes.empty()) {
      tileSizes = {2, 4, 8};
    }

    assert(tileSizes.size() == 3 &&
           "Tiling sizes for MatMul should have 3 elements");

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulTransformPattern>(ctx, tileSizes[0], tileSizes[1],
                                         tileSizes[2]);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::MatmulOp op) { gml_st::removeTransformationAttr(op); });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>();
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMatmulForCpuPass(llvm::ArrayRef<int64_t> matmulTileSizes) {
  return std::make_unique<mlir::gml_st::TransformMatmulForCpuPass>(
      matmulTileSizes);
}

}  // namespace mlir::gml_st
