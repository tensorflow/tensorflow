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

#include "mlir-hlo/Dialect/gml_st/transforms/transforms.h"

#include <utility>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/Utils/AffineCanonicalizationUtils.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BlockAndValueMapping.h"

namespace mlir {
namespace gml_st {
namespace {

/// Rewrite a LoopOp with bounds/step that potentially do not divide evenly
/// into two LoopOps: One where the step divides the iteration space
/// evenly, followed another one for the last (partial) iteration (if any). This
/// function only rewrites the `idx`-th loop of the loop nest represented by
/// the LoopOp. To peel the entire loop nest, this function must be called
/// multiple times.
///
/// This function rewrites the given LoopOp in-place and creates a new
/// LoopOp for the last iteration. It replaces all uses of the original
/// LoopOp with the results of the newly generated one.
///
/// The newly generated LoopOp is returned via `result`. The boundary
/// at which the loop is split (new upper bound) is returned via `splitBound`.
/// The return value indicates whether the LoopOp was rewritten or not.
static LogicalResult peelLoop(RewriterBase &b, LoopOp loopOp, int64_t idx,
                              LoopOp &result, Value &splitBound) {
  Value lb = loopOp.lowerBound()[idx], ub = loopOp.upperBound()[idx],
        step = loopOp.step()[idx];
  auto ubInt = getConstantIntValue(ub);

  auto loc = loopOp.getLoc();
  AffineExpr exprLb, exprUb, exprStep;
  bindSymbols(b.getContext(), exprLb, exprUb, exprStep);
  // New upper bound: %ub - (%ub - %lb) mod %step
  auto modMap = AffineMap::get(0, 3, {exprUb - ((exprUb - exprLb) % exprStep)});
  SmallVector<Value> operands{lb, ub, step};
  canonicalizeMapAndOperands(&modMap, &operands);
  modMap = simplifyAffineMap(modMap);
  RewriterBase::InsertionGuard guard(b);
  b.setInsertionPoint(loopOp);
  splitBound = b.createOrFold<AffineApplyOp>(loc, modMap, operands);
  // No specialization necessary if step already divides upper bound evenly.
  if (splitBound == ub || (ubInt && ubInt == getConstantIntValue(splitBound)))
    return failure();

  // Create remainder loop.
  b.setInsertionPointAfter(loopOp);
  auto remainderLoop = cast<LoopOp>(b.clone(*loopOp.getOperation()));
  loopOp.replaceAllUsesWith(remainderLoop->getResults());
  // Outputs: Take tensors from main loop's results. Take memrefs from main
  // loop's outputs.
  SmallVector<Value> remainderOutputs;
  for (unsigned o = 0, t = 0; o < loopOp.getNumOutputs(); ++o) {
    remainderOutputs.push_back(loopOp.outputs()[o].getType().isa<MemRefType>()
                                   ? loopOp.outputs()[o]
                                   : loopOp->getResult(t++));
  }
  remainderLoop.outputsMutable().assign(remainderOutputs);

  // Set new loop bounds.
  b.updateRootInPlace(loopOp, [&]() {
    SmallVector<Value> ubs = loopOp.upperBound();
    ubs[idx] = splitBound;
    loopOp.upperBoundMutable().assign(ubs);
  });
  SmallVector<Value> lbs = remainderLoop.lowerBound();
  lbs[idx] = splitBound;
  remainderLoop.lowerBoundMutable().assign(lbs);

  result = remainderLoop;
  return success();
}

template <typename OpTy, bool IsMin>
static void rewriteAffineOpAfterPeeling(RewriterBase &rewriter, LoopOp mainLoop,
                                        LoopOp remainderLoop, Value mainIv,
                                        Value remainderIv, Value ub,
                                        Value step) {
  mainLoop.walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.operands(), IsMin, mainIv, ub,
                                     step, /*insideLoop=*/true);
  });
  remainderLoop.walk([&](OpTy affineOp) {
    AffineMap map = affineOp.getAffineMap();
    (void)scf::rewritePeeledMinMaxOp(rewriter, affineOp, map,
                                     affineOp.operands(), IsMin, remainderIv,
                                     ub, step, /*insideLoop=*/false);
  });
}

bool isZero(Value v) {
  if (auto cst = v.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value() == 0;
  return false;
}
using ::mlir::linalg::LinalgOp;

void generateLoopNest(OpBuilder &b, Location loc, ArrayRef<Range> loopRanges,
                      LinalgOp linalgOp, ArrayRef<Attribute> iteratorTypes,
                      function_ref<scf::ValueVector(OpBuilder &, Location,
                                                    ValueRange, ValueRange)>
                          bodyBuilderFn,
                      ArrayRef<StringRef> distributionTypes) {
  SmallVector<Value, 4> lbs, ubs, steps;
  for (Range range : loopRanges) {
    lbs.emplace_back(range.offset);
    ubs.emplace_back(range.size);
    steps.emplace_back(range.stride);
  }

  auto wrappedBuilderFn = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                              ValueRange ivs, ValueRange inputs,
                              ValueRange outputs) {
    SmallVector<Value> operandValuesToUse = inputs;
    operandValuesToUse.append(outputs.begin(), outputs.end());
    scf::ValueVector results =
        bodyBuilderFn(nestedBuilder, nestedLoc, ivs, operandValuesToUse);
    nestedBuilder.create<gml_st::YieldOp>(nestedLoc, results);
  };

  SmallVector<Value> inputOperands = linalgOp.getInputOperands();
  SmallVector<Value> outputOperands = linalgOp.getOutputOperands();
  auto tiledLoop =
      b.create<LoopOp>(loc, lbs, ubs, steps, inputOperands, outputOperands,
                       b.getArrayAttr(iteratorTypes), wrappedBuilderFn);
  if (!distributionTypes.empty())
    tiledLoop.setDistributionTypes(b, distributionTypes);
}

// Insert a tile `source` into the destination tensor `dest`. The position at
// which the tile is inserted (as well as size of tile) is taken from a given
// ExtractSliceOp `sliceOp`.
Value insertSliceIntoTensor(RewriterBase &b, Location loc,
                            tensor::ExtractSliceOp sliceOp, Value source,
                            Value dest) {
  return b.create<tensor::InsertSliceOp>(
      loc, sliceOp.source().getType(), source, dest, sliceOp.offsets(),
      sliceOp.sizes(), sliceOp.strides(), sliceOp.static_offsets(),
      sliceOp.static_sizes(), sliceOp.static_strides());
}

FailureOr<linalg::TiledLinalgOp> tileLinalgOpImpl(
    RewriterBase &b, LinalgOp op, ValueRange tileSizes,
    const linalg::LinalgTilingOptions &options) {
  auto nLoops = op.getNumLoops();
  // Initial tile sizes may be too big, only take the first nLoops.
  tileSizes = tileSizes.take_front(nLoops);

  if (llvm::all_of(tileSizes, isZero)) {
    linalg::TiledLinalgOp tiledOp;
    tiledOp.op = cast<LinalgOp>(b.clone(*op.getOperation()));
    tiledOp.tensorResults.assign(tiledOp.op->result_begin(),
                                 tiledOp.op->result_end());
    return tiledOp;
  }

  // 1. Build the tiled loop ranges.
  auto allShapeSizes = op.createFlatListOfOperandDims(b, op.getLoc());
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap) return failure();

  SmallVector<Range, 4> loopRanges;
  mlir::linalg::LoopIndexToRangeIndexMap loopIndexToRangeIndex;
  std::tie(loopRanges, loopIndexToRangeIndex) =
      mlir::linalg::makeTiledLoopRanges(b, op.getLoc(), shapeSizesToLoopsMap,
                                        allShapeSizes, tileSizes);

  SmallVector<Attribute, 4> iteratorTypes;
  for (const auto &attr :
       enumerate(op.iterator_types().cast<ArrayAttr>().getValue())) {
    if (loopIndexToRangeIndex.count(attr.index()))
      iteratorTypes.push_back(attr.value());
  }

  // 2. Create the tiled loops.
  LinalgOp res = op;
  SmallVector<Value, 4> ivs, tensorResults;
  auto tiledLoopBodyBuilder =
      [&](OpBuilder & /*builder*/, Location loc, ValueRange localIvs,
          ValueRange operandValuesToUse) -> scf::ValueVector {
    ivs.assign(localIvs.begin(), localIvs.end());

    // Tile the `operandValuesToUse` that either match the `op` operands
    // themselves or the tile loop arguments forwarding them.
    assert(operandValuesToUse.size() ==
               static_cast<size_t>(op.getNumInputsAndOutputs()) &&
           "expect the number of operands and inputs and outputs to match");
    SmallVector<Value> valuesToTile = operandValuesToUse;
    auto sizeBounds =
        applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
    SmallVector<Value, 4> tiledOperands =
        makeTiledShapes(b, loc, op, valuesToTile, ivs, tileSizes, sizeBounds,
                        /*omitPartialTileCheck=*/false);

    SmallVector<Type, 4> resultTensorTypes;
    for (OpOperand *opOperand : op.getOutputTensorOperands())
      resultTensorTypes.push_back(
          tiledOperands[opOperand->getOperandNumber()].getType());

    res = op.clone(b, loc, resultTensorTypes, tiledOperands);

    // Insert a insert_slice for each output tensor.
    unsigned resultIdx = 0;
    for (OpOperand *opOperand : op.getOutputTensorOperands()) {
      Value outputTensor = tiledOperands[opOperand->getOperandNumber()];
      IRRewriter rewriter(b);
      if (auto sliceOp = outputTensor.getDefiningOp<tensor::ExtractSliceOp>()) {
        tensorResults.push_back(insertSliceIntoTensor(rewriter, loc, sliceOp,
                                                      res->getResult(resultIdx),
                                                      sliceOp.source()));
      } else {
        tensorResults.push_back(res->getResult(resultIdx));
      }
      ++resultIdx;
    }
    return scf::ValueVector(tensorResults.begin(), tensorResults.end());
  };
  generateLoopNest(b, op.getLoc(), loopRanges, op, iteratorTypes,
                   tiledLoopBodyBuilder, options.distributionTypes);

  // 3. Transform IndexOp results w.r.t. the tiling.
  mlir::linalg::transformIndexOps(b, res, ivs, loopIndexToRangeIndex);

  // 4. Gather the newly created loops and return them with the new op.
  SmallVector<Operation *, 8> loops;
  loops.reserve(ivs.size());
  for (auto iv : ivs) {
    if (iv.isa<BlockArgument>()) {
      loops.push_back(iv.cast<BlockArgument>().getOwner()->getParentOp());
      assert(loops.back() && "no owner found for induction variable!");
    } else {
      loops.push_back(nullptr);
    }
  }

  // 5. Get the tensor results from the outermost loop if available. Otherwise
  // use the previously captured `tensorResults`.
  Operation *outermostLoop = nullptr;
  for (Operation *loop : loops)
    if ((outermostLoop = loop)) break;

  return linalg::TiledLinalgOp{
      res, loops, outermostLoop ? outermostLoop->getResults() : tensorResults};
}

}  // namespace

LogicalResult peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                           LoopOp loopOp, int64_t idx,
                                           LoopOp &result) {
  int64_t numLoops = loopOp.iterator_types().size();
  if (idx < 0 || numLoops <= idx) return failure();

  Value ub = loopOp.upperBound()[idx];
  LoopOp remainderLoop;
  Value splitBound;
  if (failed(peelLoop(rewriter, loopOp, idx, remainderLoop, splitBound)))
    return failure();

  // Rewrite affine.min and affine.max ops.
  Value mainIv = loopOp.getInductionVars()[idx], step = loopOp.step()[idx],
        remainderIv = remainderLoop.getInductionVars()[idx];

  rewriteAffineOpAfterPeeling<AffineMinOp, /*IsMin=*/true>(
      rewriter, loopOp, remainderLoop, mainIv, remainderIv, ub, step);
  rewriteAffineOpAfterPeeling<AffineMaxOp, /*IsMin=*/false>(
      rewriter, loopOp, remainderLoop, mainIv, remainderIv, ub, step);

  result = remainderLoop;
  return success();
}

FailureOr<linalg::TiledLinalgOp> tileLinalgOp(
    RewriterBase &b, linalg::LinalgOp op,
    const linalg::LinalgTilingOptions &options) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);

  if (!options.tileSizeComputationFunction) return failure();

  // Enforce the convention that "tiling by zero" skips tiling a particular
  // dimension. This convention is significantly simpler to handle instead of
  // adjusting affine maps to account for missing dimensions.
  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector =
      options.tileSizeComputationFunction(b, op);
  if (tileSizeVector.size() < nLoops) {
    auto zero = b.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero);
  }

  return tileLinalgOpImpl(b, op, tileSizeVector, options);
}

namespace {
TileOp createTile(OpBuilder &b, Location loc, ValueRange ivs,
                  ValueRange upperBounds, ValueRange steps,
                  ArrayRef<int64_t> staticDims, ArrayRef<int64_t> tileSizes,
                  Value space) {
  // Compute the actual sizes of the tile.
  auto rank = ivs.size();
  SmallVector<Value> dynamicTileSizes;
  SmallVector<int64_t> staticTileSizes;
  staticTileSizes.reserve(rank);
  for (int i = 0; i < rank; ++i) {
    // Check if this dimension can be perfectly tiled.
    if (tileSizes[i] == 1 || (staticDims[i] != ShapedType::kDynamicSize &&
                              staticDims[i] % tileSizes[i] == 0)) {
      staticTileSizes.push_back(tileSizes[i]);
      continue;
    }
    auto nextIv = b.create<arith::AddIOp>(loc, ivs[i], steps[i]);
    auto isPartialTile = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                                 nextIv, upperBounds[i]);
    auto rem = b.create<arith::SubIOp>(loc, upperBounds[i], ivs[i]);
    auto size = b.create<arith::SelectOp>(loc, isPartialTile, rem, steps[i]);
    dynamicTileSizes.push_back(size);
    staticTileSizes.push_back(ShapedType::kDynamicSize);
  }
  SmallVector<int64_t> allDynamic(rank, ShapedType::kDynamicStrideOrOffset);
  auto staticSizes = b.getI64ArrayAttr(staticTileSizes);
  auto staticOffsets = b.getI64ArrayAttr(allDynamic);
  auto staticStrides = b.getI64ArrayAttr(SmallVector<int64_t>(rank, 1));
  auto tileTy = b.getType<TileType>(staticTileSizes);
  return b.create<TileOp>(loc, tileTy, space, ivs, dynamicTileSizes,
                          ValueRange{}, staticOffsets, staticSizes,
                          staticStrides);
}
}  // namespace

FailureOr<TilingResult> tileToTiles(RewriterBase &b, linalg::LinalgOp linalgOp,
                                    ArrayRef<int64_t> tileSizes) {
  Location loc = linalgOp.getLoc();
  Value output = linalgOp.getOutputOperand(0)->get();
  auto outputType = output.getType().cast<RankedTensorType>();

  SmallVector<Value> dynamicDims =
      tensor::createDynamicDimValues(b, loc, output);
  auto spaceType = b.getType<TileType>(outputType.getShape());
  auto space = b.create<SpaceOp>(loc, spaceType, dynamicDims,
                                 b.getI64ArrayAttr(outputType.getShape()));

  SmallVector<Value> steps;
  steps.reserve(outputType.getRank());
  for (int64_t size : tileSizes) {
    steps.push_back(b.create<arith::ConstantIndexOp>(loc, size));
  }

  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lowerBounds(outputType.getRank(), zero);

  SmallVector<Value> upperBounds;
  for (int i = 0; i < outputType.getRank(); ++i) {
    if (outputType.isDynamicDim(i)) {
      upperBounds.push_back(b.create<tensor::DimOp>(loc, output, i));
    } else {
      upperBounds.push_back(
          b.create<arith::ConstantIndexOp>(loc, outputType.getDimSize(i)));
    }
  }
  LinalgOp result;
  auto loop = b.create<ParallelOp>(
      loc, output.getType(), lowerBounds, upperBounds, steps,
      [&](OpBuilder &b, Location nestedLoc, ValueRange ivs) {
        auto operandTile = createTile(b, nestedLoc, ivs, upperBounds, steps,
                                      outputType.getShape(), tileSizes, space);
        auto tiledResultTy = RankedTensorType::get(
            operandTile.getType().getShape(), outputType.getElementType());
        SmallVector<Value> tiledOperands;
        tiledOperands.reserve(linalgOp.getNumInputs());

        for (auto input : linalgOp.inputs()) {
          tiledOperands.push_back(
              b.create<MaterializeOp>(nestedLoc, input, operandTile));
        }
        for (auto output : linalgOp.outputs()) {
          tiledOperands.push_back(
              b.create<MaterializeOp>(nestedLoc, output, operandTile));
        }
        // Clone 'linalgOp' and replace the original operands with
        // 'tiledOperands'.
        result = linalgOp.clone(b, loc, tiledResultTy, tiledOperands);
        SmallVector<Value> tiledResults = result->getResults();
        b.create<SubsetYieldOp>(
            nestedLoc, tiledResults, SmallVector<Value>{output},
            SmallVector<Value>(tiledResults.size(), operandTile));
      });

  return TilingResult{loop.getOperation(), result};
}

FailureOr<TilingResult> tileToPoints(RewriterBase &b,
                                     linalg::LinalgOp linalgOp) {
  Location loc = linalgOp.getLoc();
  auto output = linalgOp.getOutputOperand(0)->get();
  auto outputType = output.getType().cast<RankedTensorType>();

  SmallVector<Value> dynamicDims =
      tensor::createDynamicDimValues(b, loc, output);
  auto spaceType = b.getType<TileType>(outputType.getShape());
  Value space = b.create<SpaceOp>(loc, spaceType, dynamicDims,
                                  b.getI64ArrayAttr(outputType.getShape()));

  Value one = b.create<arith::ConstantIndexOp>(loc, 1);
  SmallVector<Value> steps(outputType.getRank(), one);

  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> lowerBounds(outputType.getRank(), zero);

  SmallVector<Value> upperBounds;
  for (int i = 0; i < outputType.getRank(); ++i) {
    if (outputType.isDynamicDim(i)) {
      upperBounds.push_back(b.create<tensor::DimOp>(loc, output, i));
    } else {
      upperBounds.push_back(
          b.create<arith::ConstantIndexOp>(loc, outputType.getDimSize(i)));
    }
  }
  auto loop = b.create<ParallelOp>(
      loc, output.getType(), lowerBounds, upperBounds, steps,
      [&](OpBuilder &b, Location nestedLoc, ValueRange ivs) {
        Value point = b.create<PointOp>(
            nestedLoc, b.getType<PointType>(), space, ivs,
            b.getI64ArrayAttr(SmallVector<int64_t>(
                outputType.getRank(), ShapedType::kDynamicStrideOrOffset)));

        BlockAndValueMapping bvm;
        for (auto &en : llvm::enumerate(linalgOp.getInputOperands()))
          bvm.map(linalgOp.getTiedBlockArgument(en.value()),
                  b.create<MaterializeOp>(nestedLoc, en.value()->get(), point));

        Block &block = linalgOp->getRegion(0).front();
        for (auto &op : block.without_terminator()) b.clone(op, bvm);

        SmallVector<Value> yieldedValues;
        for (Value val : block.getTerminator()->getOperands())
          yieldedValues.push_back(bvm.lookup(val));
        b.create<SubsetYieldOp>(
            nestedLoc, yieldedValues, SmallVector<Value>{output},
            SmallVector<Value>(yieldedValues.size(), point));
      });

  return TilingResult{loop.getOperation(), nullptr};
}

}  // namespace gml_st
}  // namespace mlir
