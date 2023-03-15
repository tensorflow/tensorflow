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
#include <optional>
#include <string>
#include <utility>

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/utils/linalg_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TILINGSOFTMAXPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kTileSoftmaxAppliedLabel =
    "__tile_softmax_applied_label__";

Operation *fuseIthOperandInPlace(PatternRewriter &rewriter, Operation *op,
                                 int64_t i) {
  auto matOp =
      llvm::cast<tensor::ExtractSliceOp>(op->getOperand(i).getDefiningOp());
  FailureOr<Operation *> fused = fuse(rewriter, matOp);
  assert(succeeded(fused) && "expect success after matching");
  return *fused;
}

LogicalResult tilePartialSoftmax(
    TilingInterface op, PatternRewriter &rewriter,
    llvm::function_ref<FailureOr<Operation *>(Operation *, int64_t)>
        tileOperationFn) {
  // Match cwise root op.
  // Match all operands to be derived from the same source value in one of two
  // ways:
  //   i)  by a reduction and subsequent bcast in one dimension, or
  //   ii) by using the source value as is.
  Value commonSource;
  std::optional<int64_t> commonReductionDim;
  SmallVector<std::optional<SimpleBcastReduction>> simpleBcastReductions;
  auto mapOp = llvm::dyn_cast_or_null<linalg::MapOp>(op.getOperation());
  if (!mapOp || mapOp.getNumDpsInits() != 1)
    return rewriter.notifyMatchFailure(op, "no mapOp");
  for (Value operand : mapOp.getInputs()) {
    // Case i.
    SimpleBcastReduction bcastReduction;
    int64_t reductionDim;
    if (isSimpleBcastReduction(operand.getDefiningOp(), &reductionDim,
                               &bcastReduction)) {
      if (commonSource && commonSource != bcastReduction.operand) {
        return rewriter.notifyMatchFailure(bcastReduction.bcast,
                                           "no common reduction source");
      }
      commonSource = bcastReduction.operand;
      if (commonReductionDim && *commonReductionDim != reductionDim) {
        return rewriter.notifyMatchFailure(bcastReduction.reduction,
                                           "no common reduction dim");
      }
      commonReductionDim = reductionDim;
      simpleBcastReductions.push_back(bcastReduction);
      continue;
    }

    // Case ii.
    if (commonSource && commonSource != operand)
      return rewriter.notifyMatchFailure(op, "common source != operand");
    commonSource = operand;
    simpleBcastReductions.push_back(std::nullopt);
  }

  if (!commonReductionDim || !commonSource)
    return rewriter.notifyMatchFailure(op, "no common dim/src");

  // Tile or fuse cwise root op.
  FailureOr<Operation *> tiledOp = tileOperationFn(op, *commonReductionDim);
  if (failed(tiledOp))
    return rewriter.notifyMatchFailure(op, "call to tileOperationFn failed");
  setLabel(*tiledOp, kTileSoftmaxAppliedLabel);

  // Fuse through the bcast reduction chains.
  Value commonTiledSource;
  for (int64_t i = 0; i < static_cast<int64_t>(simpleBcastReductions.size());
       i++) {
    if (!simpleBcastReductions[i]) continue;

    // Fuse.
    Operation *tiledBcast = fuseIthOperandInPlace(rewriter, *tiledOp, i);
    Operation *tiledReduction =
        fuseIthOperandInPlace(rewriter, tiledBcast, /*i=*/0);

    // Use common tiled source value.
    if (commonTiledSource) {
      tiledReduction->setOperand(0, commonTiledSource);
    } else {
      commonTiledSource = tiledReduction->getOperands().front();
    }
  }

  // Also use the common tiled source value for the remaining operands.
  for (size_t i = 0; i < simpleBcastReductions.size(); i++) {
    if (simpleBcastReductions[i]) continue;
    (*tiledOp)->setOperand(i, commonTiledSource);
  }

  return success();
}

struct TilePartialSoftmaxPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern<TilingInterface>::OpInterfaceRewritePattern;

  TilePartialSoftmaxPattern(MLIRContext *ctx, SmallVector<int64_t> tileSizes,
                            PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(ctx, benefit),
        tileSizes(std::move(tileSizes)) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kTileSoftmaxAppliedLabel))
      return rewriter.notifyMatchFailure(op, "has tranformation attr");

    // Only apply to non-fusable occurrences.
    bool hasFusableOccurrences = llvm::any_of(
        op->getUsers(),
        [](Operation *op) { return llvm::isa<tensor::ExtractSliceOp>(op); });
    if (hasFusableOccurrences)
      return rewriter.notifyMatchFailure(op, "has fusable occurrences");

    return tilePartialSoftmax(
        op, rewriter,
        [&](Operation *op,
            int64_t commonReductionDim) -> FailureOr<Operation *> {
          // Populate tiling options.
          scf::SCFTilingOptions tilingOptions;
          tilingOptions.setTileSizeComputationFunction(
              [&](OpBuilder &b, Operation *op) -> SmallVector<Value> {
                Location loc = op->getLoc();
                SmallVector<Value> tileSizeValues;
                for (int64_t i = 0; i < static_cast<int64_t>(tileSizes.size());
                     i++) {
                  // Skip tiling the reduction dimension. By convention, this is
                  // a tile size of 0.
                  int64_t tileSizeInDim =
                      i == commonReductionDim ? 0 : tileSizes[i];
                  tileSizeValues.push_back(
                      b.create<arith::ConstantIndexOp>(loc, tileSizeInDim));
                }
                return tileSizeValues;
              });
          // Tile.
          FailureOr<TilingResult> tilingResult =
              tileUsingSCFForallOp(rewriter, op, tilingOptions);
          if (failed(tilingResult)) return failure();

          rewriter.replaceOp(op, tilingResult->loop->getResults());
          setLabel(tilingResult->tiledOps.front(), kTileSoftmaxAppliedLabel);
          return tilingResult->tiledOps.front();
        });
  }

 private:
  SmallVector<int64_t> tileSizes;
};

struct FusePartialSoftmaxPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    Value source = op.getSource();
    Operation *def = source.getDefiningOp();
    if (!def) return failure();

    if (!llvm::isa<TilingInterface>(def)) return failure();

    return tilePartialSoftmax(
        def, rewriter,
        [&](Operation *cwiseOp,
            int64_t /*commonReductionDim*/) -> FailureOr<Operation *> {
          auto iface = llvm::dyn_cast_or_null<TilingInterface>(cwiseOp);
          if (!iface) {
            return rewriter.notifyMatchFailure(
                cwiseOp, "doesn't implement tiling iface");
          }

          // By construction, we assume that the tile spans the operand in the
          // common reduction dimension (`commonReductionDim`).
          // TODO(frgossen): Assert this assumption when we have moved to
          // unnested tiles.

          // Fuse.
          SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
          SmallVector<OpFoldResult> sizes = op.getMixedSizes();
          FailureOr<Value> result =
              iface.generateResultTileValue(rewriter, 0, offsets, sizes);
          if (failed(result)) {
            return rewriter.notifyMatchFailure(
                cwiseOp, "failed to generate result tile");
          }

          rewriter.replaceOp(op, *result);
          return result->getDefiningOp();
        });
  }
};

struct FuseUnaryCwisePattern : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter &rewriter) const override {
    // Match unary cwise ops.
    Operation *source = op.getSource().getDefiningOp();
    auto mapOp = dyn_cast_or_null<linalg::MapOp>(source);
    if (!mapOp || mapOp.getNumDpsInputs() != 1) return failure();
    // Fuse.
    return fuse(rewriter, op);
  }
};

struct TilingSoftmaxPass
    : public impl::TilingSoftmaxPassBase<TilingSoftmaxPass> {
  TilingSoftmaxPass() = default;
  explicit TilingSoftmaxPass(ArrayRef<int64_t> ts) { this->tileSizes = ts; }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect, linalg::LinalgDialect, tensor::TensorDialect,
                    scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    // Populate tiling and fusion patterns for partial softmax and unary cwise
    // ops.
    RewritePatternSet patterns(ctx);
    SmallVector<int64_t> tileSizes(this->tileSizes.begin(),
                                   this->tileSizes.end());
    patterns.insert<TilePartialSoftmaxPattern>(ctx, tileSizes);
    patterns.insert<FuseUnaryCwisePattern, FusePartialSoftmaxPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    f.walk([](Operation *op) { removeLabel(op, kTileSoftmaxAppliedLabel); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass() {
  return std::make_unique<TilingSoftmaxPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass(
    ArrayRef<int64_t> tileSizes) {
  return std::make_unique<TilingSoftmaxPass>(tileSizes);
}

}  // namespace mlir::gml_st
