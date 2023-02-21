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

#include <algorithm>
#include <array>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/IR/TensorInferTypeOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/TensorTilingInterfaceImpl.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMDOTFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kDotTransformedLabel = "__dot_transformed_label__";

FailureOr<scf::SCFTilingResult> tileDot(PatternRewriter &rewriter,
                                        Operation *op,
                                        ArrayRef<int64_t> tileSizes) {
  scf::SCFTilingOptions opts;
  opts.setTileSizes(tileSizes);

  auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, opts);
  if (failed(tilingResult)) return failure();

  // Update the results if tiling occurred.
  if (!tilingResult->loops.empty()) {
    rewriter.replaceOp(op, tilingResult->replacements);
    op = tilingResult->tiledOps.front();
  }

  setLabel(op, kDotTransformedLabel);
  return tilingResult;
}

/// Pattern to tile dot operations (linalg.matvec, linalg.vecmat, linalg.dot)
/// and peel the generated loops.
template <typename DotTy>
struct DotTransformPattern : public OpRewritePattern<DotTy> {
  using OpRewritePattern<DotTy>::OpRewritePattern;

  explicit DotTransformPattern(MLIRContext *context,
                               llvm::ArrayRef<int64_t> parallelDimsTileSizes,
                               llvm::ArrayRef<int64_t> reductionDimTileSizes,
                               PatternBenefit benefit = 1)
      : OpRewritePattern<DotTy>(context, benefit),
        parallelDimsTileSizes(parallelDimsTileSizes),
        reductionDimTileSizes(reductionDimTileSizes) {}

  LogicalResult matchAndRewrite(DotTy dotOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(dotOp, kDotTransformedLabel)) {
      return rewriter.notifyMatchFailure(dotOp,
                                         "has already been transformed.");
    }
    if (isa<scf::ForOp>(dotOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          dotOp, "has already been tiled by another pass.");
    }

    auto tilingParallelDimsResult =
        tileDot(rewriter, dotOp.getOperation(), parallelDimsTileSizes);
    if (failed(tilingParallelDimsResult)) return failure();

    if (!tilingParallelDimsResult->loops.empty()) {
      dotOp = cast<DotTy>(tilingParallelDimsResult->tiledOps.back());
    }

    // Second level tiling: reduction dimension.
    auto tilingReductionDimResult =
        tileDot(rewriter, dotOp.getOperation(), reductionDimTileSizes);
    if (failed(tilingReductionDimResult)) return failure();

    if (!tilingReductionDimResult->loops.empty()) {
      dotOp = cast<DotTy>(tilingReductionDimResult->tiledOps.back());
    }

    // Peel parallel loops.
    for (auto &loop : tilingParallelDimsResult->loops) {
      (void)peelSCFForOp(rewriter, loop);
    }

    // Peel reduction loop inside the main parallel loop, label the main loop as
    // "perfectly tiled" one, to enable vectorization after canonicalization.
    auto peelingResult =
        peelSCFForOp(rewriter, tilingReductionDimResult->loops.front());
    setLabel(peelingResult.mainLoop, kPerfectlyTiledLoopLabel);

    return success();
  }

 private:
  llvm::SmallVector<int64_t> parallelDimsTileSizes;
  llvm::SmallVector<int64_t> reductionDimTileSizes;
};

struct TransformDotForCpuPass
    : public impl::TransformDotForCpuPassBase<TransformDotForCpuPass> {
  TransformDotForCpuPass() = default;

  explicit TransformDotForCpuPass(llvm::ArrayRef<int64_t> dotTileSizes) {
    tileSizes = dotTileSizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<mlir::gml_st::GmlStDialect, arith::ArithDialect,
                    linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
    tensor::registerTilingInterfaceExternalModels(registry);
    tensor::registerInferTypeOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    if (tileSizes.empty()) {
      tileSizes = {8, 8, 8};
    }
    // Dot operations can have at most 3 dimensions ((upto) 2 parallel + 1
    // reduction), so the first two tileSizes' elements are for parallel
    // dimensions tiling, and the last element is for reduction dimension
    // tiling.
    // - for dot ops that have 2 parallel dimensions (linalg.matmul): the whole
    // tileSizes vector will be used.
    // - for ops that only have only 1 parallel dimension
    // (linalg.matvec/linalg.vecmat), only the first and last elements of
    // tileSizes are used.
    // - for ops that have no parallel dimension (linalg.dot), only the last
    // element of tileSizes is used.
    assert(tileSizes.size() == 3 &&
           "Tiling sizes for Dot operations should have 3 elements");

    RewritePatternSet patterns(ctx);
    patterns.add<DotTransformPattern<linalg::MatvecOp>>(
        ctx, llvm::ArrayRef<int64_t>{tileSizes[0], 0},
        llvm::ArrayRef<int64_t>{0, tileSizes[2]});
    patterns.add<DotTransformPattern<linalg::VecmatOp>>(
        ctx, llvm::ArrayRef<int64_t>{tileSizes[0], 0},
        llvm::ArrayRef<int64_t>{0, tileSizes[2]});
    patterns.add<DotTransformPattern<linalg::DotOp>>(
        ctx, llvm::ArrayRef<int64_t>{}, llvm::ArrayRef<int64_t>{tileSizes[2]});
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](Operation *op) {
      if (isa<linalg::MatvecOp, linalg::VecmatOp, linalg::DotOp>(op))
        removeLabel(op, kDotTransformedLabel);
    });
  }
};
}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformDotForCpuPass(llvm::ArrayRef<int64_t> dotTileSizes) {
  return std::make_unique<mlir::gml_st::TransformDotForCpuPass>(dotTileSizes);
}

}  // namespace mlir::gml_st
