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

#include "mlir-hlo/Dialect/gml_st/transforms/tiling_using_interface.h"

#include <memory>
#include <tuple>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Arithmetic/Utils/Utils.h"
#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace gml_st {
namespace {

constexpr llvm::StringLiteral kOpLabel = "op_label";

/// Generate an empty loop nest that represents the tiled loop nest shell.
/// - `loopRanges` specifies the lb, ub and step of the untiled iteration space.
/// - `tileSizeVals` is the tile sizes to use. Zero represent untiled loops.
/// - In `offsets` and `sizes` return the multi-dimensional offset and size of
/// the tile processed within the inner most loop.
gml_st::ForOp generateTileLoopNest(OpBuilder &builder, Location loc,
                                   ArrayRef<Range> loopRanges,
                                   ArrayRef<Value> tileSizeVals,
                                   ArrayRef<Value> dstOperands,
                                   SmallVector<OpFoldResult> &offsets,
                                   SmallVector<OpFoldResult> &sizes) {
  assert(!loopRanges.empty() && "expected at least one loop range");
  assert(loopRanges.size() == tileSizeVals.size() &&
         "expected as many tile sizes as loop ranges");
  OpBuilder::InsertionGuard guard(builder);

  // The tile size to use (to avoid out of bounds access) is  minimum of
  // `tileSize` and `ub - iv`, where `iv` is the induction variable
  // of the tiled loop.
  AffineExpr s0, s1, d0;
  bindDims(builder.getContext(), d0);
  bindSymbols(builder.getContext(), s0, s1);
  AffineMap minMap = AffineMap::get(1, 2, {s0, s1 - d0}, builder.getContext());

  SmallVector<Value> lbs, ubs, steps;
  SmallVector<unsigned> nonemptyRangeIndices;
  for (auto &loopRange : llvm::enumerate(loopRanges)) {
    Value offset =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().offset);
    Value size =
        getValueOrCreateConstantIndexOp(builder, loc, loopRange.value().size);
    // No loops if tile size is zero. Set offset and size to the loop
    // offset and size.
    offsets.push_back(offset);
    sizes.push_back(size);
    if (matchPattern(tileSizeVals[loopRange.index()], m_Zero())) continue;
    lbs.push_back(offset);
    ubs.push_back(size);
    steps.push_back(tileSizeVals[loopRange.index()]);
    nonemptyRangeIndices.push_back(loopRange.index());
  }

  auto loop = builder.create<gml_st::ForOp>(
      loc, TypeRange(ValueRange{dstOperands}), lbs, ubs, steps, dstOperands,
      [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange ivs,
          ValueRange /*inits*/) {
        for (const auto &en : llvm::enumerate(ivs)) {
          Value iv = en.value();
          size_t index = en.index();
          Value boundedTileSize = nestedBuilder.create<AffineMinOp>(
              bodyLoc, minMap, ValueRange{iv, steps[index], ubs[index]});
          sizes[nonemptyRangeIndices[index]] = boundedTileSize;
          offsets[nonemptyRangeIndices[index]] = iv;
        }
      });
  return loop;
}

}  // namespace

TileToGmlStLoops::TileToGmlStLoops(MLIRContext *context, StringRef tilingTarget,
                                   GmlStTilingOptions options,
                                   PatternBenefit benefit)
    : OpInterfaceRewritePattern<TilingInterface>(context, benefit),
      tilingTarget(tilingTarget),
      options(std::move(options)) {}

LogicalResult TileToGmlStLoops::matchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  auto opLabelAttr = op->getAttr(kOpLabel);
  if (!opLabelAttr) return failure();

  auto opLabelStrAttr = opLabelAttr.cast<StringAttr>();
  if (!opLabelStrAttr || opLabelStrAttr.getValue() != tilingTarget)
    return failure();

  if (hasTransformationAttr(op)) return failure();

  auto tilingResult = returningMatchAndRewrite(op, rewriter);
  if (failed(tilingResult)) return failure();

  setTransformationAttr(rewriter, tilingResult->tiledOp);
  return success();
}

FailureOr<GmlStTilingResult> TileToGmlStLoops::returningMatchAndRewrite(
    TilingInterface op, PatternRewriter &rewriter) const {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(op);

  if (!options.tileSizeComputationFunction) {
    return rewriter.notifyMatchFailure(
        op, "missing tile size computation function");
  }

  // 1. Get the range of the loops that are represented by the operation.
  SmallVector<Range> iterationDomain = op.getIterationDomain(rewriter);
  size_t numLoops = iterationDomain.size();
  if (numLoops == 0)
    return rewriter.notifyMatchFailure(op, "missing iteration domain");

  // 2. Materialize the tile sizes. Enforce the convention that "tiling by zero"
  // skips tiling a particular dimension. This convention is significantly
  // simpler to handle instead of adjusting affine maps to account for missing
  // dimensions.
  SmallVector<Value> tileSizeVector =
      options.tileSizeComputationFunction(rewriter, op);
  if (tileSizeVector.size() < iterationDomain.size()) {
    auto zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(numLoops - tileSizeVector.size(), zero);
  }

  // 3. Materialize an empty loop nest that iterates over the tiles.
  auto dstOperands = op.getDestinationOperands(rewriter);
  SmallVector<OpFoldResult> offsets, sizes;
  GmlStTilingResult tilingResult;
  tilingResult.loop =
      generateTileLoopNest(rewriter, op.getLoc(), iterationDomain,
                           tileSizeVector, dstOperands, offsets, sizes);
  Block *loopBody = tilingResult.loop.getBody();
  Operation *terminator = loopBody->getTerminator();
  rewriter.setInsertionPoint(terminator);

  // 4. Insert the tiled implementation within the loop.
  tilingResult.tiledOp =
      op.getTiledImplementation(rewriter, dstOperands, offsets, sizes, true);

  // 5. Add `gml_st.set_yield` terminator.
  SmallVector<Value> dstSubsets;
  for (Value dst : tilingResult.tiledOp.getDestinationOperands(rewriter))
    dstSubsets.push_back(dst.getDefiningOp<MaterializeOp>().set());
  rewriter.replaceOpWithNewOp<SetYieldOp>(
      terminator, tilingResult.tiledOp->getResults(), dstOperands, dstSubsets);

  // 6. Replace the uses of `outputs` with the output block arguments.
  for (auto [dst, regionArg] :
       llvm::zip(dstOperands, tilingResult.loop.getRegionOutputArgs())) {
    dst.replaceUsesWithIf(regionArg, [&](OpOperand &operand) {
      return operand.getOwner()->getBlock() == loopBody;
    });
  }
  rewriter.replaceOp(op, tilingResult.loop.getResults());
  return tilingResult;
}

}  // namespace gml_st
}  // namespace mlir
