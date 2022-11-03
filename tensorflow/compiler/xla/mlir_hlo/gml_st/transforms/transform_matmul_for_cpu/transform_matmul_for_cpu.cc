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
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMMATMULFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

FailureOr<TilingResult> tileMatmul(PatternRewriter &rewriter,
                                   TilingInterface op,
                                   ArrayRef<int64_t> tileSizes,
                                   bool distribute) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  opts.distribute = distribute;
  return tile(opts, rewriter, op);
}

/// Pattern to tile `linalg.matmul` and fuse `linalg.fill` into generated
/// `gml_st.parallel`.
struct MatmulTransformPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern<TilingInterface>::OpInterfaceRewritePattern;

  MatmulTransformPattern() = default;

  explicit MatmulTransformPattern(MLIRContext *context,
                                  int64_t lhsParallelDimTileSize = 2,
                                  int64_t rhsParallelDimTileSize = 4,
                                  int64_t reductionDimTileSize = 8,
                                  PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        lhsParallelDimTileSize(lhsParallelDimTileSize),
        rhsParallelDimTileSize(rhsParallelDimTileSize),
        reductionDimTileSize(reductionDimTileSize) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (hasTransformationAttr(op) || !isa<mlir::linalg::MatmulOp>(op))
      return failure();

    TilingInterface matmul = op;

    // First level tiling: parallel dimensions.
    SmallVector<int64_t> parallelDimsTileSizes{lhsParallelDimTileSize,
                                               rhsParallelDimTileSize, 0};
    auto tilingParallelDimsResult =
        tileMatmul(rewriter, matmul, parallelDimsTileSizes,
                   /*distribute=*/true);
    if (failed(tilingParallelDimsResult)) return failure();

    // Update the results if tiling succeeded.
    if (tilingParallelDimsResult->loop != nullptr) {
      rewriter.replaceOp(matmul, tilingParallelDimsResult->loop->getResults());
      matmul = cast<TilingInterface>(tilingParallelDimsResult->tiledOp);
    }

    // Fusion into the output.
    OpOperand *matmulOutput =
        cast<linalg::MatmulOp>(matmul.getOperation()).getDpsInitOperand(0);
    auto materialize = matmulOutput->get().getDefiningOp<MaterializeOp>();
    if (!materialize) return failure();
    if (materialize.getSource().getDefiningOp<linalg::FillOp>()) {
      if (failed(fuse(rewriter, materialize))) return failure();
    }

    // Second level tiling: reduction dimension.
    SmallVector<int64_t> reductionDimsTileSizes{0, 0, reductionDimTileSize};
    auto tilingReductionDimsResult = tileMatmul(
        rewriter, matmul, reductionDimsTileSizes, /*distribute=*/false);
    if (failed(tilingReductionDimsResult)) return failure();

    // Update the results if tiling succeeded.
    if (tilingReductionDimsResult->loop != nullptr) {
      rewriter.replaceOp(matmul, tilingReductionDimsResult->loop->getResults());
      matmul = cast<TilingInterface>(tilingReductionDimsResult->tiledOp);
    }

    setTransformationAttr(rewriter, matmul);
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

  explicit TransformMatmulForCpuPass(
      llvm::ArrayRef<int64_t> matmulTileSizes) {
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

    if ((*tileSizes).empty()) {
      tileSizes = {2, 4, 8};
    }

    assert(tileSizes.size() == 3 &&
           "Tiling sizes for MatMul should have 3 elements");

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulTransformPattern>(ctx, (*tileSizes)[0], (*tileSizes)[1],
                                         (*tileSizes)[2]);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::LinalgOp op) { gml_st::removeTransformationAttr(op); });
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
