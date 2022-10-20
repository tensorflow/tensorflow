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

#include "mlir-hlo/Dialect/gml_st/transforms/fusion.h"
#include "mlir-hlo/Dialect/gml_st/transforms/linalg_utils.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling.h"
#include "mlir-hlo/Dialect/gml_st/transforms/tiling_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_TILINGSOFTMAXPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

Operation *fuseIthOperandInPlace(PatternRewriter &rewriter, Operation *op,
                                 int64_t i) {
  auto matOp = llvm::cast<MaterializeOp>(op->getOperand(i).getDefiningOp());
  FailureOr<Value> fused = createFusedOp(rewriter, matOp);
  assert(succeeded(fused) && "expect success after matching");
  rewriter.replaceOp(matOp, *fused);
  return fused->getDefiningOp();
}

LogicalResult tilePartialSoftmax(
    TilingInterface op, PatternRewriter &rewriter,
    llvm::function_ref<FailureOr<Operation *>(Operation *, int64_t)>
        tileOperationFn) {
  // Match cwise root op.
  int64_t arity;
  if (!isCwiseGenericOp(op, arity)) return failure();

  // Match all operands to be derived from the same source value in one of two
  // ways:
  //   i)  by a reduction and subsequent bcast in one dimension, or
  //   ii) by using the source value as is.
  Value commonSource;
  Optional<int64_t> commonReductionDim;
  SmallVector<Optional<SimpleBcastReduction>> simpleBcastReductions;
  auto genericOp = llvm::dyn_cast_or_null<linalg::GenericOp>(op.getOperation());
  for (Value operand : genericOp.getInputs()) {
    // Case i.
    SimpleBcastReduction bcastReduction;
    int64_t reductionDim;
    if (isSimpleBcastReduction(operand.getDefiningOp(), reductionDim,
                               bcastReduction)) {
      if (commonSource && commonSource != bcastReduction.operand) {
        return failure();
      }
      commonSource = bcastReduction.operand;
      if (commonReductionDim && *commonReductionDim != reductionDim) {
        return failure();
      }
      commonReductionDim = reductionDim;
      simpleBcastReductions.push_back(bcastReduction);
      // foundBcastReduction = true;
      continue;
    }

    // Case ii.
    if (commonSource && commonSource != operand) return failure();
    commonSource = operand;
    simpleBcastReductions.push_back(llvm::None);
  }

  if (!commonReductionDim || !commonSource) return failure();

  // Tile or fuse cwise root op.
  FailureOr<Operation *> tiledOp = tileOperationFn(op, *commonReductionDim);
  if (failed(tiledOp)) return failure();
  setTransformationAttr(rewriter, *tiledOp);

  // Fuse through the bcast reduction chains.
  Value commonTiledSource;
  for (int64_t i = 0; i < simpleBcastReductions.size(); i++) {
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
  for (int64_t i = 0; i < simpleBcastReductions.size(); i++) {
    if (simpleBcastReductions[i]) continue;
    (*tiledOp)->setOperand(i, commonTiledSource);
  }

  return success();
}

struct TilePartialSoftmaxPattern
    : public OpInterfaceRewritePattern<TilingInterface> {
  using OpInterfaceRewritePattern<TilingInterface>::OpInterfaceRewritePattern;

  TilePartialSoftmaxPattern(MLIRContext *ctx, bool distribute,
                            SmallVector<int64_t> tileSizes,
                            PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(ctx, benefit),
        distribute(distribute),
        tileSizes(std::move(tileSizes)) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (hasTransformationAttr(op)) return failure();

    // Only apply to non-fusable occurrences.
    bool hasFusableOccurrences = llvm::any_of(
        op->getUsers(),
        [](Operation *op) { return llvm::isa<MaterializeOp>(op); });
    if (hasFusableOccurrences) return failure();

    return tilePartialSoftmax(
        op, rewriter,
        [&](Operation *op,
            int64_t commonReductionDim) -> FailureOr<Operation *> {
          // Populate tiling options.
          TilingOptions tilingOptions;
          tilingOptions.tileSizeComputationFn =
              [&](OpBuilder &b, Operation *op) -> SmallVector<Value> {
            Location loc = op->getLoc();
            SmallVector<Value> tileSizeValues;
            for (int64_t i = 0; i < tileSizes.size(); i++) {
              // Skip tiling the reduction dimension. By convention, this is a
              // tile size of 0.
              int64_t tileSizeInDim =
                  i == commonReductionDim ? 0 : tileSizes[i];
              tileSizeValues.push_back(
                  b.create<arith::ConstantIndexOp>(loc, tileSizeInDim));
            }
            return tileSizeValues;
          };
          tilingOptions.distribute = distribute;

          // Tile.
          FailureOr<TilingResult> tilingResult =
              tile(tilingOptions, rewriter, op);
          if (failed(tilingResult)) return failure();

          rewriter.replaceOp(op, tilingResult->loop->getResults());
          setTransformationAttr(rewriter, tilingResult->tiledOp);
          return tilingResult->tiledOp;
        });
  }

 private:
  bool distribute;
  SmallVector<int64_t> tileSizes;
};

struct FusePartialSoftmaxPattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
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
          if (!iface) return failure();

          // By construction, we assume that the tile spans the operand in the
          // common reduction dimension (`commonReductionDim`).
          // TODO(frgossen): Assert this assumption when we have moved to
          // unnested tiles.

          // Extract tile offsets and sizes.
          auto tile = op.getSet().getDefiningOp<TileOp>();
          if (!tile) return failure();

          // Fuse.
          SmallVector<OpFoldResult> offsets = tile.getMixedOffsets();
          SmallVector<OpFoldResult> sizes = tile.getMixedSizes();
          FailureOr<Value> result =
              iface.generateResultTileValue(rewriter, 0, offsets, sizes);
          if (failed(result)) return failure();

          rewriter.replaceOp(op, *result);
          return result->getDefiningOp();
        });
  }
};

struct FuseUnaryCwisePattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter &rewriter) const override {
    // Match unary cwise ops.
    Operation *source = op.getSource().getDefiningOp();
    if (!isUnaryCwiseGenericOp(source)) return failure();

    // Fuse.
    FailureOr<Value> fused = createFusedOp(rewriter, op);
    if (failed(fused)) return failure();

    rewriter.replaceOp(op, *fused);
    return success();
  }
};

struct TilingSoftmaxPass
    : public impl::TilingSoftmaxPassBase<TilingSoftmaxPass> {
  TilingSoftmaxPass() = default;
  TilingSoftmaxPass(bool distr, ArrayRef<int64_t> ts) {
    this->distribute = distr;
    this->tileSizes = ts;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<GmlStDialect, linalg::LinalgDialect, tensor::TensorDialect>();
    registerGmlStTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    // Populate tiling and fusion patterns for partial softmax and unary cwise
    // ops.
    RewritePatternSet patterns(ctx);
    SmallVector<int64_t> tileSizes(this->tileSizes.begin(),
                                   this->tileSizes.end());
    patterns.insert<TilePartialSoftmaxPattern>(ctx, distribute, tileSizes);
    patterns.insert<FuseUnaryCwisePattern, FusePartialSoftmaxPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Clean up by removing temporary attributes.
    f.walk([](Operation *op) { removeTransformationAttr(op); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass() {
  return std::make_unique<TilingSoftmaxPass>();
}

std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass(
    bool distribute, ArrayRef<int64_t> tileSizes) {
  return std::make_unique<TilingSoftmaxPass>(distribute, tileSizes);
}

}  // namespace gml_st
}  // namespace mlir
