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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMREVERSEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kReverseTransformedLabel =
    "__reverse_transformed_label__";

FailureOr<TilingResult> tileReverseAndUpdateResultIfTiled(
    PatternRewriter &rewriter, thlo::ReverseOp &reverseOp,
    ArrayRef<int64_t> tileSizes) {
  TilingOptions opts;
  opts.setTileSizeComputationFn(tileSizes);
  auto tilingResult = tileUsingGmlSt(
      opts, rewriter, cast<TilingInterface>(reverseOp.getOperation()));

  if (failed(tilingResult)) return failure();

  // Update the results if tiling occurred.
  if (tilingResult->loop != nullptr) {
    rewriter.replaceOp(reverseOp, tilingResult->loop->getResults());
    reverseOp = cast<thlo::ReverseOp>(tilingResult->tiledOps.front());
  }

  return tilingResult;
}

SmallVector<int64_t> getTileSizes(int64_t rank, int64_t vectorSize,
                                  bool tileToScalarize) {
  SmallVector<int64_t> sizes(rank, 1);
  if (!tileToScalarize) sizes[rank - 1] = vectorSize;
  return sizes;
}

/// Pattern to tile `thlo.reverse`.
struct ReverseTransformPattern : public OpRewritePattern<thlo::ReverseOp> {
  using OpRewritePattern<thlo::ReverseOp>::OpRewritePattern;

  explicit ReverseTransformPattern(MLIRContext *context, int64_t vectorSize,
                                   PatternBenefit benefit = 1)
      : OpRewritePattern<thlo::ReverseOp>(context, benefit),
        vectorSize(vectorSize) {}

  LogicalResult matchAndRewrite(thlo::ReverseOp reverseOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(reverseOp, kReverseTransformedLabel))
      return rewriter.notifyMatchFailure(reverseOp,
                                         "has already been transformed.");
    if (isa<scf::ForallOp, scf::ForOp>(reverseOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          reverseOp, "has already been tiled by another pass.");
    }
    // Parallel dimension tiling. Tiling will be of the form
    // 1x1x..x1xVectorSize.
    int64_t rank = reverseOp.getInput().getType().getRank();
    auto tilingResult = tileReverseAndUpdateResultIfTiled(
        rewriter, reverseOp, getTileSizes(rank, vectorSize, false));

    // Peel parallel loop.
    auto peelingResult = peelAllLoops(tilingResult->loop, rewriter);

    // Fold operations in the remainder loops before scalarization. thlo.reverse
    // will be folded if the last dim is not reversed.
    for (scf::ForallOp remParLoop : peelingResult.tailLoops) {
      remParLoop->walk([&](Operation *childOp) {
        SmallVector<Value> foldValue;
        if (succeeded(rewriter.tryFold(childOp, foldValue))) {
          childOp->replaceAllUsesWith(foldValue);
        }
      });
    }

    // Tile ops in the peeled loop again, to size 1, so they can be
    // scalarized.
    if (failed(tilePeeledOpsToScalars(rewriter, peelingResult,
                                      kReverseTransformedLabel,
                                      /*fuseFilterFn=*/nullptr)))
      return failure();

    setLabel(reverseOp, kReverseTransformedLabel);
    return success();
  }

 private:
  int64_t vectorSize;
};

struct TransformReverseForCpuPass
    : public impl::TransformReverseForCpuPassBase<TransformReverseForCpuPass> {
  TransformReverseForCpuPass() = default;

  explicit TransformReverseForCpuPass(int64_t reverseVectorSize = 8) {
    vectorSize = reverseVectorSize;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    tensor::TensorDialect, scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<ReverseTransformPattern>(ctx, vectorSize);
    populateCollapseForallOpDimensionsPattern(patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](thlo::ReverseOp reverseOp) {
      removeLabel(reverseOp, kReverseTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformReverseForCpuPass(int64_t vectorSize) {
  return std::make_unique<mlir::gml_st::TransformReverseForCpuPass>(vectorSize);
}

}  // namespace mlir::gml_st
