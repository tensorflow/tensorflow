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

#include "gml_st/transforms/scalarization/scalarization.h"

#include <memory>
#include <optional>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_SCALARIZATIONPASS
#include "gml_st/transforms/passes.h.inc"

using linalg::LinalgOp;
using tensor::ExtractOp;
using tensor::FromElementsOp;
using tensor::InsertOp;

// Fold `tensor.insert_slice(tensor.from_elements(x), dst)` into
//      `tensor.insert(x, dst)` for single-element tensors.
struct FoldTensorFromElementsIntoInsertSlice
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp insertSliceOp,
                                PatternRewriter &rewriter) const override {
    auto fromElementsOp =
        insertSliceOp.getSource().getDefiningOp<FromElementsOp>();
    if (!fromElementsOp || !hasSingleElement(fromElementsOp.getType())) {
      return failure();
    }
    SmallVector<Value> indices = getValueOrCreateConstantIndexOp(
        rewriter, insertSliceOp.getLoc(), insertSliceOp.getMixedOffsets());
    rewriter.replaceOpWithNewOp<tensor::InsertOp>(
        insertSliceOp, fromElementsOp.getElements().front(),
        insertSliceOp.getDest(), indices);
    return success();
  }
};

// Fold `gml_st.set_yield(tensor.from_elements(x) -> tensor<1x1xf32>)` into
//      `gml_st.set_yield(x)` for single-element tensors.
struct FoldTensorFromElementsIntoSetYield
    : public OpRewritePattern<gml_st::SetYieldOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(gml_st::SetYieldOp yieldOp,
                                PatternRewriter &rewriter) const override {
    bool isFoldingPossible = false;
    SmallVector<Value> newSrcs;
    for (auto [src, set] : llvm::zip(yieldOp.getSrcs(), yieldOp.getSets())) {
      auto fromElementsOp = src.getDefiningOp<FromElementsOp>();
      if (!fromElementsOp) continue;

      if (hasSingleElement(fromElementsOp.getType())) {
        newSrcs.push_back(fromElementsOp.getElements().front());
        isFoldingPossible = true;
        continue;
      }
      newSrcs.push_back(src);
    }

    if (!isFoldingPossible) return failure();

    // Update in-place to make sure that the accumulator regions don't get lost.
    rewriter.updateRootInPlace(
        yieldOp, [&]() { yieldOp.getSrcsMutable().assign(newSrcs); });
    return success();
  }
};

LogicalResult inlinePayload(PatternRewriter &rewriter, Location loc,
                            LinalgOp linalgOp, ValueRange argValues) {
  // Clone everything but terminator.
  Block *body = linalgOp.getBlock();
  IRMapping map;
  map.map(body->getArguments(), argValues);
  for (auto &op : body->without_terminator()) {
    if (auto indexOp = dyn_cast<linalg::IndexOp>(&op)) {
      Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      map.map(indexOp.getResult(), zero);
      continue;
    }
    rewriter.clone(op, map);
  }

  // Wrap every scalar result into a tensor using `tensor.from_elements`.
  SmallVector<Value> newResults;
  for (auto [resultType, yieldOperand] : llvm::zip(
           linalgOp->getResultTypes(), body->getTerminator()->getOperands())) {
    auto scalarValue = map.lookupOrDefault(yieldOperand);
    newResults.push_back(
        rewriter.create<FromElementsOp>(loc, resultType, scalarValue));
  }
  rewriter.replaceOp(linalgOp, newResults);
  return success();
}

// `scalarizeLinalgOp` has to be wrapped in OpInterfaceRewritePattern, because
// `patterns.add` does not support adding interface rewriter patterns yet.
struct ScalarizeLinalgOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    return scalarizeLinalgOp(linalgOp, rewriter);
  }
};

// Returns `startIndices`[0, :] for `startIndices` of shape 1xn. Returns None if
// startIndices has a different shape.
Optional<SmallVector<Value>> extractStartIndices(
    ImplicitLocOpBuilder &b, TypedValue<ShapedType> startIndices) {
  if (startIndices.getType().getRank() != 2 ||
      startIndices.getType().getDimSize(0) != 1) {
    return std::nullopt;
  }

  int64_t indexVectorSize = startIndices.getType().getDimSize(1);
  SmallVector<Value> result;
  result.reserve(indexVectorSize);
  Value zero = b.create<arith::ConstantIndexOp>(0);
  for (int64_t i = 0; i < indexVectorSize; ++i) {
    result.push_back(b.create<ExtractOp>(
        startIndices, ValueRange{zero, b.create<arith::ConstantIndexOp>(i)}));
  }
  return result;
}

// Return i1 value after checking that 0 <= indices < dims(tensor).
Value isValidIndex(OpBuilder &b, Location loc, ArrayRef<Value> indices,
                   ArrayRef<Value> tensorDims, Value &zero) {
  auto i1Type = b.getI1Type();
  Value isValid = b.create<arith::ConstantOp>(
      loc, i1Type, IntegerAttr::get(i1Type, APInt(1, 1)));

  for (auto [dim, index] : llvm::zip(tensorDims, indices)) {
    Value geZero =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge, index, zero);
    Value ltDim =
        b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, index, dim);
    Value dimInBounds = b.create<arith::AndIOp>(loc, geZero, ltDim);
    isValid = b.create<arith::AndIOp>(loc, isValid, dimInBounds);
  }
  return isValid;
}

Value isIndexInBounds(ImplicitLocOpBuilder &b, Location loc,
                      ArrayRef<Value> updatesDimValues,
                      ArrayRef<Value> scatterIndices,
                      ArrayRef<Value> initDimValues, Value &zero, Value &one) {
  SmallVector<Value> limitIndex{updatesDimValues.drop_front()};
  for (const auto &en : llvm::enumerate(scatterIndices)) {
    limitIndex[en.index()] =
        b.create<arith::AddIOp>(loc, limitIndex[en.index()], en.value());
  }
  for (auto &value : limitIndex) {
    value = b.create<arith::SubIOp>(loc, value, one);
  }

  Value inBounds = isValidIndex(b, loc, limitIndex, initDimValues, zero);
  return b.create<arith::AndIOp>(
      loc, inBounds, isValidIndex(b, loc, scatterIndices, initDimValues, zero));
}

Value tensorHasElement(OpBuilder &b, Location loc, Value input,
                       int64_t concatDim) {
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  Value concatDimSize = b.create<tensor::DimOp>(loc, input, concatDim);
  return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, concatDimSize,
                                 zero);
}

Value extractElementFromInputs(
    OpBuilder &b, Location loc, ValueRange inputs, Type resultType,
    int64_t concatDim,
    llvm::function_ref<Value(OpBuilder &, Location, Value)>
        materializeAndInsert) {
  if (inputs.size() == 1) {
    return materializeAndInsert(b, loc, inputs.front());
  }

  return b
      .create<scf::IfOp>(
          loc, tensorHasElement(b, loc, inputs.front(), concatDim),
          [&](OpBuilder &thenBuilder, Location thenLoc) {
            thenBuilder.create<scf::YieldOp>(
                thenLoc,
                materializeAndInsert(thenBuilder, thenLoc, inputs.front()));
          },
          [&](OpBuilder &elseBuilder, Location elseLoc) {
            elseBuilder.create<scf::YieldOp>(
                elseLoc, extractElementFromInputs(
                             elseBuilder, elseLoc, inputs.drop_front(),
                             resultType, concatDim, materializeAndInsert));
          })
      .getResult(0);
}

LogicalResult scalarizeOp(Operation *op, PatternRewriter &rewriter,
                          TypedValue<ShapedType> &input,
                          TypedValue<ShapedType> &output) {
  ImplicitLocOpBuilder b(op->getLoc(), rewriter);

  auto outputType = output.getType().dyn_cast<RankedTensorType>();
  if (!outputType) {
    return rewriter.notifyMatchFailure(
        op, "failed to cast output to RankedTensorType");
  }
  if (!hasSingleElement(outputType)) {
    return rewriter.notifyMatchFailure(
        op, "has output with number of elements not equal to 1");
  }

  auto inputType = input.getType().dyn_cast<RankedTensorType>();
  if (!inputType) {
    return rewriter.notifyMatchFailure(
        op, "failed to cast input to RankedTensorType");
  }

  Value zero = b.create<arith::ConstantIndexOp>(0);
  llvm::SmallVector<Value> indicesInput(inputType.getRank(), zero);
  llvm::SmallVector<Value> indicesOutput(outputType.getRank(), zero);

  Value extractedValue = b.create<ExtractOp>(input, indicesInput);
  Value result = b.create<tensor::FromElementsOp>(outputType, extractedValue);

  rewriter.replaceOp(op, result);
  return success();
}

LogicalResult hoistTensorExtractFromForOp(scf::ForOp forOp,
                                          PatternRewriter &rewriter) {
  if (forOp.getNumIterOperands() != 1) return failure();
  OpOperand &iterOperand = forOp.getIterOpOperands().front();
  auto iterArgTensorTy =
      dyn_cast<RankedTensorType>(iterOperand.get().getType());
  if (!iterArgTensorTy || !hasSingleElement(iterArgTensorTy)) return failure();

  Value bbArg = forOp.getRegionIterArgForOpOperand(iterOperand);

  if (!bbArg.hasOneUse()) return failure();

  Operation *user = *bbArg.getUsers().begin();
  auto extractOp = dyn_cast<tensor::ExtractOp>(user);
  if (!extractOp) return failure();

  Operation *terminator = forOp.getBody()->getTerminator();
  auto fromTensorOp =
      terminator->getOperand(0).getDefiningOp<tensor::FromElementsOp>();
  if (!fromTensorOp) return failure();

  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);
  Location loc = forOp.getLoc();
  Value extractedElement = rewriter.create<tensor::ExtractOp>(
      loc, iterOperand.get(), extractOp.getIndices());
  auto newForOp = rewriter.create<scf::ForOp>(
      loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      ValueRange{extractedElement});
  newForOp->setAttrs(forOp->getAttrs());
  Block *newLoopBody = newForOp.getBody();

  // Move old body into new for loop.
  rewriter.setInsertionPointToStart(newLoopBody);
  SmallVector<Value> blockArgs{
      newForOp.getInductionVar(),
      rewriter.create<tensor::FromElementsOp>(loc, iterArgTensorTy,
                                              newForOp.getRegionIterArg(0))};
  rewriter.mergeBlocks(forOp.getBody(), newLoopBody, blockArgs);

  // Replace terminator that yields a tensor with the one that yields the
  // element.
  Operation *newTerminator = newForOp.getBody()->getTerminator();
  rewriter.setInsertionPointAfter(newTerminator);
  Value elemOfYieldedTensor = rewriter.create<tensor::ExtractOp>(
      loc, terminator->getOperand(0), extractOp.getIndices());
  rewriter.replaceOpWithNewOp<scf::YieldOp>(newTerminator, elemOfYieldedTensor);

  // Replace the old loop with the new loop result wrapped in a tensor.
  rewriter.setInsertionPointAfter(newForOp);
  rewriter.replaceOpWithNewOp<tensor::FromElementsOp>(
      forOp, forOp.getResultTypes().front(), newForOp.getResult(0));

  return success();
}

LogicalResult hoistTensorExtractFromIfOp(scf::IfOp ifOp,
                                         PatternRewriter &rewriter) {
  // Analyse result types and determine what we can scalarize.
  int64_t numResults = ifOp.getNumResults();
  SmallVector<bool> isScalarizableResult(numResults, false);
  SmallVector<Type> unscalarizedResultType =
      llvm::to_vector(ifOp.getResultTypes());
  SmallVector<Type> scalarizedResultType =
      llvm::to_vector(ifOp.getResultTypes());
  bool isAnyResultScalarizable = false;
  for (int64_t i = 0; i < numResults; ++i) {
    auto rankedTy = scalarizedResultType[i].dyn_cast<RankedTensorType>();
    if (!rankedTy || !hasSingleElement(rankedTy)) continue;
    isScalarizableResult[i] = true;
    scalarizedResultType[i] = rankedTy.getElementType();
    isAnyResultScalarizable = true;
  }

  if (!isAnyResultScalarizable) {
    return rewriter.notifyMatchFailure(ifOp, "cannot scalarize any result");
  }

  // Create new if ifOp.
  Location loc = ifOp.getLoc();
  Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  auto scalarizedOp = rewriter.create<scf::IfOp>(loc, scalarizedResultType,
                                                 ifOp.getCondition());
  scalarizedOp.getThenRegion().takeBody(ifOp.getThenRegion());
  scalarizedOp.getElseRegion().takeBody(ifOp.getElseRegion());
  for (int64_t i = 0; i < numResults; ++i) {
    if (!isScalarizableResult[i]) continue;

    // Insert `extract` ops to yield value as a scalar.
    llvm::SmallVector<Value> zeroIndices(
        unscalarizedResultType[i].cast<RankedTensorType>().getRank(), zero);
    rewriter.setInsertionPoint(scalarizedOp.thenYield());
    Value thenScalar = rewriter.createOrFold<tensor::ExtractOp>(
        loc, scalarizedOp.thenYield().getOperand(i), zeroIndices);
    scalarizedOp.thenYield().setOperand(i, thenScalar);
    rewriter.setInsertionPoint(scalarizedOp.elseYield());
    Value elseScalar = rewriter.createOrFold<tensor::ExtractOp>(
        loc, scalarizedOp.elseYield().getOperand(i), zeroIndices);
    scalarizedOp.elseYield().setOperand(i, elseScalar);
  }

  // Insert `from_elements` ifOp to be type compatible.
  rewriter.setInsertionPointAfter(scalarizedOp);
  SmallVector<Value> results(scalarizedOp.getResults());
  for (int64_t i = 0; i < numResults; ++i) {
    if (!isScalarizableResult[i]) continue;

    // Wrap scalar.
    results[i] = rewriter.create<tensor::FromElementsOp>(
        loc, unscalarizedResultType[i], results[i]);
  }

  rewriter.replaceOp(ifOp, results);
  return success();
}

struct ScalarizationPass
    : public impl::ScalarizationPassBase<ScalarizationPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ScalarizeLinalgOp, FoldTensorFromElementsIntoInsertSlice,
                 FoldTensorFromElementsIntoSetYield>(context);
    patterns.add(hoistTensorExtractFromForOp);
    patterns.add(hoistTensorExtractFromIfOp);
    patterns.add(scalarizeConcatenateOp);
    patterns.add(scalarizeDynamicBroadcastInDimOp);
    patterns.add(scalarizeGatherOp);
    patterns.add(scalarizeReverseOp);
    patterns.add(scalarizeScatterOp);

    FromElementsOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};

}  // namespace

LogicalResult scalarizeConcatenateOp(thlo::ConcatenateOp concatenateOp,
                                     PatternRewriter &rewriter) {
  Location loc = concatenateOp.getLoc();
  int64_t concatDim = concatenateOp.getDimension().getSExtValue();

  auto initTensor = concatenateOp.getInit();
  auto initType = initTensor.getType();
  int64_t rank = initTensor.getType().getRank();

  // Only scalarize when it's statically known that output concatenation dim
  // size is one.
  if (initType.getShape()[concatDim] != 1) {
    return failure();
  }

  IntegerAttr oneAttr = rewriter.getIndexAttr(1);
  SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> strides(rank, oneAttr);

  SmallVector<OpFoldResult> sizes;
  for (int i = 0; i < rank; ++i) {
    if (i == concatDim) {
      sizes.push_back(oneAttr);
    } else {
      sizes.emplace_back(rewriter.create<tensor::DimOp>(loc, initTensor, i));
    }
  }

  auto materializeAndInsert = [&](OpBuilder &b, Location l, Value input) {
    Value slice =
        b.create<tensor::ExtractSliceOp>(l, input, offsets, sizes, strides);
    return b.create<tensor::InsertSliceOp>(l, slice, initTensor, offsets, sizes,
                                           strides);
  };

  Value res =
      extractElementFromInputs(rewriter, loc, concatenateOp.getInputs(),
                               initType, concatDim, materializeAndInsert);

  rewriter.replaceOp(concatenateOp, res);

  return success();
}

LogicalResult scalarizeDynamicBroadcastInDimOp(
    thlo::DynamicBroadcastInDimOp broadcastOp, PatternRewriter &rewriter) {
  auto input = broadcastOp.getOperand();
  auto output = broadcastOp.getInit();
  return scalarizeOp(broadcastOp, rewriter, input, output);
}

LogicalResult scalarizeGatherOp(thlo::GatherOp gatherOp,
                                PatternRewriter &rewriter) {
  Location loc = gatherOp.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);
  auto startIndices = extractStartIndices(b, gatherOp.getStartIndices());
  if (!startIndices) return failure();

  TypedValue<ShapedType> init = gatherOp.getInit();
  ShapedType initTy = init.getType();
  int64_t initRank = initTy.getRank();
  SmallVector<OpFoldResult> initDimSizes = tensor::getMixedSizes(b, loc, init);
  SmallVector<Value> initDimSizeValues =
      getValueOrCreateConstantIndexOp(b, loc, initDimSizes);

  IntegerAttr oneAttr = b.getI64IntegerAttr(1);

  TypedValue<ShapedType> operand = gatherOp.getOperand();
  auto operandSizes = getValueOrCreateConstantIndexOp(
      b, loc, tensor::createDimValues(b, loc, operand));
  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);

  SmallVector<Value> sliceSizes{initDimSizeValues.begin() + 1,
                                initDimSizeValues.end()};
  while (sliceSizes.size() < startIndices->size()) {
    sliceSizes.push_back(one);
  }

  // Clamp the indices.
  for (auto &&[startIndex, max, sliceSize] :
       llvm::zip(*startIndices, operandSizes, sliceSizes)) {
    auto maxMinusSize = b.createOrFold<arith::SubIOp>(loc, max, sliceSize);
    startIndex = b.create<arith::MinSIOp>(loc, startIndex, maxMinusSize);
    startIndex = b.create<arith::MaxSIOp>(loc, startIndex, zero);
  }

  SmallVector<Value> lbs(initRank, zero);
  SmallVector<Value> steps(initRank, one);

  scf::LoopNest loopNest = scf::buildLoopNest(
      rewriter, loc, lbs, initDimSizeValues, steps, ValueRange{init},
      [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange ivs,
          ValueRange loopInits) {
        // Compute the index in the operand.
        SmallVector<Value> readIndices(operand.getType().getRank(), zero);
        llvm::copy(ivs.drop_front(1), readIndices.begin());
        for (auto &&[readIndex, startIndex] :
             llvm::zip(readIndices, *startIndices)) {
          readIndex = nestedBuilder.create<arith::AddIOp>(bodyLoc, readIndex,
                                                          startIndex);
        }

        // Materialize the value and yield it.
        SmallVector<OpFoldResult> ones(initRank, oneAttr);
        Value val = nestedBuilder.create<tensor::ExtractOp>(bodyLoc, operand,
                                                            readIndices);
        Value updatedInit = nestedBuilder.create<tensor::InsertOp>(
            bodyLoc, val, loopInits.front(), ivs);

        return scf::ValueVector({updatedInit});
      });

  rewriter.replaceOp(gatherOp, loopNest.results);
  return success();
}

LogicalResult scalarizeLinalgOp(LinalgOp linalgOp, PatternRewriter &rewriter) {
  // Fail if not every argument is a scalar or a single-element tensor.
  if (!hasSingleElementOperandsAndResults(linalgOp)) return failure();

  // Load the data corresponding to the block arguments that
  // represent input operands.
  SmallVector<Value> indexedValues;
  indexedValues.reserve(linalgOp->getNumOperands());
  Location loc = linalgOp->getLoc();
  auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  for (OpOperand &operand : linalgOp->getOpOperands()) {
    if (!linalgOp.payloadUsesValueFromOperand(&operand)) {
      indexedValues.push_back(nullptr);
      continue;
    }
    if (linalgOp.isScalar(&operand)) {
      indexedValues.push_back(operand.get());
      continue;
    }
    Value operandValue = operand.get();
    Type operandType = operandValue.getType();
    SmallVector<Value> indices(operandType.cast<RankedTensorType>().getRank(),
                               zero);
    Value load = rewriter.create<ExtractOp>(loc, operandValue, indices);
    indexedValues.push_back(load);
  }

  // Inline the op payload and rewrite the operation.
  return inlinePayload(rewriter, loc, linalgOp, indexedValues);
}

LogicalResult scalarizeReverseOp(thlo::ReverseOp reverseOp,
                                 PatternRewriter &rewriter) {
  auto input = reverseOp.getInput();
  auto output = reverseOp.getInit();
  return scalarizeOp(reverseOp, rewriter, input, output);
}

LogicalResult scalarizeScatterOp(thlo::ScatterOp scatterOp,
                                 PatternRewriter &rewriter) {
  Location loc = scatterOp.getLoc();
  ImplicitLocOpBuilder b(loc, rewriter);
  b.setInsertionPoint(scatterOp);

  auto scatterIndices = extractStartIndices(b, scatterOp.getIndices());
  if (!scatterIndices) return failure();
  Value updates = scatterOp.getUpdates();
  auto updatesType = updates.getType().dyn_cast<RankedTensorType>();
  if (!updatesType) return failure();
  unsigned updatesRank = updatesType.getRank();

  SmallVector<OpFoldResult> updatesDimSizes =
      tensor::getMixedSizes(b, loc, updates);
  SmallVector<Value> updatesDimValues =
      getValueOrCreateConstantIndexOp(b, loc, updatesDimSizes);

  Value init = scatterOp.getInit();
  auto initType = init.getType().dyn_cast<RankedTensorType>();
  if (!initType) return failure();
  SmallVector<Value> initDimValues = getValueOrCreateConstantIndexOp(
      b, loc, tensor::getMixedSizes(b, loc, init));

  Value zero = b.create<arith::ConstantIndexOp>(0);
  Value one = b.create<arith::ConstantIndexOp>(1);

  Value indexIsInBounds =
      isIndexInBounds(b, loc, updatesDimValues, scatterIndices.value(),
                      initDimValues, zero, one);
  auto ifOp = b.create<scf::IfOp>(
      loc, indexIsInBounds,
      [&](OpBuilder &thenBuilder, Location thenLoc) {
        SmallVector<OpFoldResult> collapsedOffsets;
        for (size_t i = 0; i < updatesRank - 1; ++i) {
          collapsedOffsets.push_back(
              i < (scatterIndices->size()) ? (*scatterIndices)[i] : zero);
        }
        SmallVector<OpFoldResult> collapsedSizes;
        for (size_t i = 1; i < updatesRank; ++i) {
          collapsedSizes.push_back(updatesDimSizes[i]);
        }

        auto collapsedStrides = SmallVector<OpFoldResult>(updatesRank - 1, one);

        // If body consists only from terminator, then insert the update
        // slice into `init`, otherwise reduce the update slice with the same
        // body.
        if (scatterOp.getBody()->getOperations().size() == 1) {
          SmallVector<OpFoldResult> offsets(updatesRank, zero);
          SmallVector<OpFoldResult> strides(updatesRank, one);

          // Create rank-reducing `tensor.extract_slice` to avoid insertion of
          // `tensor.collapse_shape` to get rid of the outer size-1 dimension.
          RankedTensorType resultType =
              tensor::ExtractSliceOp::inferCanonicalRankReducedResultType(
                  /*resultRank=*/updatesRank - 1,
                  updatesType.cast<RankedTensorType>(), offsets,
                  updatesDimSizes, strides);
          Value extracted = thenBuilder.create<tensor::ExtractSliceOp>(
              thenLoc, resultType, updates, offsets, updatesDimSizes, strides);

          // Insert resized `updates` into `init`.
          Value inserted = thenBuilder.create<tensor::InsertSliceOp>(
              thenLoc, extracted, init, collapsedOffsets, collapsedSizes,
              collapsedStrides);
          thenBuilder.create<scf::YieldOp>(thenLoc, inserted);
          return;
        }

        // Extract a slice form `init`.
        Value extracted = thenBuilder.create<tensor::ExtractSliceOp>(
            thenLoc, init, collapsedOffsets, collapsedSizes, collapsedStrides);

        // Reduce `updates` into that slice.
        auto reduced = thenBuilder.create<linalg::ReduceOp>(
            thenLoc, extracted.getType().cast<RankedTensorType>(), updates,
            extracted, ArrayRef<int64_t>({0}));
        reduced.getRegion().takeBody(scatterOp.getBodyRegion());

        Operation *yield = reduced.getBlock()->getTerminator();

        OpBuilder::InsertionGuard g(rewriter);
        rewriter.setInsertionPoint(yield);
        rewriter.replaceOpWithNewOp<linalg::YieldOp>(yield,
                                                     yield->getOperands());
        // Put that slice back.
        auto inserted = thenBuilder.create<tensor::InsertSliceOp>(
            thenLoc, reduced.getResults().front(), init, collapsedOffsets,
            collapsedSizes, collapsedStrides);
        thenBuilder.create<scf::YieldOp>(thenLoc, inserted.getResult());
      },
      [&](OpBuilder &elseBuilder, Location elseLoc) {
        elseBuilder.create<scf::YieldOp>(elseLoc, init);
      });
  rewriter.replaceOp(scatterOp, ifOp.getResults());
  return success();
}

std::unique_ptr<OperationPass<func::FuncOp>> createScalarizationPass() {
  return std::make_unique<ScalarizationPass>();
}

}  // namespace gml_st
}  // namespace mlir
