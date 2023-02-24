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

#include "gml_st/transforms/tiling/tiling.h"

#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {

void TilingOptions::setTileSizeComputationFn(ArrayRef<int64_t> ts) {
  SmallVector<int64_t, 4> tileSizes(ts.begin(), ts.end());
  tileSizeComputationFn = [tileSizes](OpBuilder &b, Operation *op) {
    return llvm::to_vector<4>(map_range(tileSizes, [&](int64_t s) {
      return b.create<arith::ConstantIndexOp>(op->getLoc(), s).getResult();
    }));
  };
}

namespace {

#define GEN_PASS_DEF_TILINGPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kTileAppliedLabel = "__tile_applied_label__";

// Compute tile size for the tile that starts at `offset`, has size `tileSize`
// for the tensor with the dimension size `dimSize`.
// The tile size is static when `tileSize` divides `dimSize` or when the
// `tileSize` is 1.
// Otherwise, it is minimum of `tileSize` and `dimSize - offset` to avoid out of
// bounds access.
OpFoldResult computeTileSizeInDim(OpBuilder &builder, Location loc,
                                  OpFoldResult tileSize, OpFoldResult dimSize,
                                  OpFoldResult offset) {
  std::optional<int64_t> tileCst = getConstantIntValue(tileSize);
  std::optional<int64_t> dimCst = getConstantIntValue(dimSize);

  bool hasTileSizeOne = tileCst && *tileCst == 1;
  bool dividesEvenly = tileCst && dimCst && ((*dimCst % *tileCst) == 0);
  if (hasTileSizeOne || dividesEvenly) return builder.getIndexAttr(*tileCst);

  AffineExpr d0, s0;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0);
  OpFoldResult residualTileSize =
      makeComposedFoldedAffineApply(builder, loc, s0 - d0, {offset, dimSize});

  return makeComposedFoldedAffineMin(
      builder, loc, AffineMap::getMultiDimIdentityMap(2, loc.getContext()),
      {residualTileSize, tileSize});
}

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the tile processed within the inner most loop.
ParallelOp generateTileLoopNest(OpBuilder &builder, Location loc,
                                ArrayRef<Range> loopRanges,
                                ArrayRef<Value> tileSizeVals,
                                ArrayRef<Value> dstOperands,
                                SmallVector<OpFoldResult> &offsets,
                                SmallVector<OpFoldResult> &sizes) {
  assert(!loopRanges.empty() && "expected at least one loop range");
  assert(loopRanges.size() == tileSizeVals.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(builder);

  SmallVector<OpFoldResult> lbs, ubs, steps;
  SmallVector<unsigned> nonemptyRangeIndices;
  for (auto &loopRange : llvm::enumerate(loopRanges)) {
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().offset);
    Value size =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().size);
    // No loops if tile size is zero. Set offset and size to the loop offset and
    // size.
    offsets.push_back(offset);
    sizes.push_back(size);
    if (matchPattern(tileSizeVals[loopRange.index()], m_Zero())) continue;
    lbs.push_back(offset);
    ubs.push_back(size);
    steps.push_back(tileSizeVals[loopRange.index()]);
    nonemptyRangeIndices.push_back(loopRange.index());
  }

  auto buildBody = [&](OpBuilder &nestedBuilder, Location bodyLoc,
                       ValueRange ivs) {
    for (const auto &[index, iv] : llvm::enumerate(ivs)) {
      offsets[nonemptyRangeIndices[index]] = iv;
      sizes[nonemptyRangeIndices[index]] = computeTileSizeInDim(
          nestedBuilder, bodyLoc, steps[index], ubs[index], iv);
    }
  };
  return builder.create<gml_st::ParallelOp>(
      loc, TypeRange(ValueRange{dstOperands}),
      getValueOrCreateConstantIndexOp(builder, loc, lbs),
      getValueOrCreateConstantIndexOp(builder, loc, ubs),
      getValueOrCreateConstantIndexOp(builder, loc, steps), dstOperands,
      std::nullopt,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange ivs,
          ValueRange /*outputs*/) { buildBody(nestedBuilder, bodyLoc, ivs); });
}

/// Pattern to tile an op that implements the `TilingInterface` using
/// `gml_st.for` for iterating over the tiles.
struct TilingPattern : public OpInterfaceRewritePattern<TilingInterface> {
  TilingPattern(MLIRContext *context,
                llvm::function_ref<LogicalResult(TilingInterface)> filterFn,
                TilingOptions options, bool distribute,
                PatternBenefit benefit = 1)
      : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
        filterFn(filterFn),
        options(std::move(options)),
        distribute(distribute) {}

  LogicalResult matchAndRewrite(TilingInterface op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn || failed(filterFn(op)) || hasLabel(op, kTileAppliedLabel))
      return failure();

    if (distribute) {
      auto tilingResult = tileUsingGmlSt(options, rewriter, op);
      if (failed(tilingResult)) return failure();

      if (tilingResult->loop != nullptr) {
        rewriter.replaceOp(op, tilingResult->loop->getResults());
      }
      setLabel(tilingResult->tiledOps.front(), kTileAppliedLabel);
    } else {
      scf::SCFTilingOptions opts;
      opts.setTileSizeComputationFunction(options.tileSizeComputationFn);
      auto tilingResult = scf::tileUsingSCFForOp(rewriter, op, opts);
      if (failed(tilingResult)) return failure();
      if (!tilingResult->loops.empty()) {
        rewriter.replaceOp(op, tilingResult->replacements);
      }
      setLabel(tilingResult->tiledOps.front(), kTileAppliedLabel);
    }
    return success();
  }

 private:
  llvm::function_ref<LogicalResult(TilingInterface)> filterFn;
  TilingOptions options;
  bool distribute;
};

struct TilingPass : public impl::TilingPassBase<TilingPass> {
  TilingPass() = default;
  TilingPass(StringRef name, StringRef label, bool distributeFlag,
             llvm::ArrayRef<int64_t> sizes) {
    opName = name.str();
    opLabel = label.str();
    distribute = distributeFlag;
    tileSizes = sizes;
  }

  void getDependentDialects(DialectRegistry &registry) const final {
    registry.insert<GmlStDialect, tensor::TensorDialect, linalg::LinalgDialect,
                    scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    TilingOptions opts;
    SmallVector<int64_t> ts(tileSizes.begin(), tileSizes.end());
    opts.tileSizeComputationFn = [ts](OpBuilder &b, Operation *op) {
      OpBuilder::InsertionGuard guard(b);
      b.setInsertionPointToStart(
          &op->getParentOfType<func::FuncOp>().getBody().front());
      return llvm::to_vector<4>(llvm::map_range(ts, [&](int64_t s) {
        Value v = b.create<arith::ConstantIndexOp>(op->getLoc(), s);
        return v;
      }));
    };

    auto filterFn = [&](TilingInterface op) {
      if (!opName.empty() && op->getName().getStringRef() != opName)
        return failure();
      if (!opLabel.empty() && !hasMatchingLabel(op, opLabel)) return failure();
      return success();
    };
    RewritePatternSet patterns(ctx);
    patterns.add<TilingPattern>(ctx, filterFn, opts, distribute);
    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns))))
      return signalPassFailure();

    // Clean up by removing temporary attributes.
    removeTilingLabels(f);
  }
};

template <typename LoopTy>
void insertTerminatorAndUpdateOutputs(PatternRewriter &rewriter,
                                      const TilingResult &tilingResult,
                                      SetYieldOp terminator,
                                      ValueRange dstOperands,
                                      ValueRange outputTiles) {
  auto parallelLoop = cast<LoopTy>(tilingResult.loop);
  rewriter.replaceOpWithNewOp<SetYieldOp>(
      terminator, tilingResult.tiledOps.front()->getResults(),
      parallelLoop.getRegionOutputArgs(), outputTiles);

  if (auto dstOp = dyn_cast<DestinationStyleOpInterface>(
          tilingResult.tiledOps.front())) {
    for (auto [dst, regionArg] :
         llvm::zip(dstOperands, parallelLoop.getRegionOutputArgs())) {
      dst.replaceUsesWithIf(regionArg, [&](OpOperand &operand) {
        Operation *owner = operand.getOwner();
        return isa<tensor::ExtractSliceOp, TilingInterface>(owner) &&
               owner->getParentOfType<LoopTy>() == parallelLoop.getOperation();
      });
    }
  }
}

}  // namespace

FailureOr<TilingResult> tileUsingGmlSt(const TilingOptions &options,
                                       PatternRewriter &rewriter,
                                       TilingInterface op) {
  rewriter.setInsertionPoint(op);
  if (!options.tileSizeComputationFn) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }
  Location loc = op.getLoc();

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0) return failure();

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by
  // zero" skips tiling a particular dimension. This convention is
  // significantly simpler to handle instead of adjusting affine maps to
  // account for missing dimensions.
  SmallVector<Value> tileSizeVector;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    tileSizeVector = options.tileSizeComputationFn(rewriter, op);
  }

  if (tileSizeVector.size() < iterationDomain.size()) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }

  if (llvm::all_of(tileSizeVector, mlir::gml_st::isZero)) {
    return TilingResult{{op}, nullptr};
  }

  // 3. Materialize an empty loop nest that iterates over the tiles.
  SmallVector<Value> dstOperands;
  if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, dstOperands)))
    return rewriter.notifyMatchFailure(op, "failed to get destinations");
  SmallVector<OpFoldResult> offsets, sizes;
  TilingResult tilingResult;
  tilingResult.loop =
      generateTileLoopNest(rewriter, loc, iterationDomain, tileSizeVector,
                           dstOperands, offsets, sizes);
  Block *loopBody = &tilingResult.loop->getRegion(0).front();
  auto terminator = cast<SetYieldOp>(loopBody->getTerminator());
  rewriter.setInsertionPoint(terminator);

  // 4. Insert the tiled implementation within the loop.
  tilingResult.tiledOps = op.getTiledImplementation(rewriter, offsets, sizes);

  // 5. Compute tiles for the insertion.
  int64_t numResults = op->getNumResults();
  SmallVector<Value> outputTiles;
  auto oneAttr = rewriter.getI64IntegerAttr(1);
  for (const auto &result : llvm::enumerate(op->getResults())) {
    SmallVector<OpFoldResult> resultOffsetsList(numResults),
        resultSizesList(numResults);
    if (failed(op.getResultTilePosition(rewriter, result.index(), offsets,
                                        sizes, resultOffsetsList,
                                        resultSizesList))) {
      return rewriter.notifyMatchFailure(
          op, "failed to get slice of result produced");
    }
    outputTiles.push_back(rewriter.createOrFold<TileOp>(
        loc, resultOffsetsList, resultSizesList,
        SmallVector<OpFoldResult>(resultSizesList.size(), oneAttr)));
  }

  // 6. Add a `set_yield` terminator, update the uses of `outputs` with the
  // output bbArgs.
  insertTerminatorAndUpdateOutputs<ParallelOp>(
      rewriter, tilingResult, terminator, dstOperands, outputTiles);
  return tilingResult;
}

void populateTilingPatterns(
    MLIRContext *context,
    llvm::function_ref<LogicalResult(TilingInterface)> filterFn,
    const TilingOptions &opts, RewritePatternSet *patterns) {}

void removeTilingLabels(Operation *op) {
  op->walk([](Operation *op) { removeLabel(op, kTileAppliedLabel); });
}

std::unique_ptr<OperationPass<func::FuncOp>> createTilingPass(
    StringRef opName, StringRef opLabel, bool distribute,
    ArrayRef<int64_t> tileSizes) {
  return std::make_unique<TilingPass>(opName, opLabel, distribute, tileSizes);
}

}  // namespace gml_st
}  // namespace mlir
