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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

Value materializePoint(OpBuilder &b, Location loc, Value valueToTile,
                       ArrayRef<OpFoldResult> offsets) {
  auto tensorType = valueToTile.getType().cast<RankedTensorType>();
  int64_t rank = tensorType.getRank();

  IntegerAttr oneAttr = b.getIndexAttr(1);
  SmallVector<OpFoldResult> sizes(rank, oneAttr);
  SmallVector<OpFoldResult> strides(rank, oneAttr);

  Value slice = b.create<tensor::ExtractSliceOp>(loc, valueToTile, offsets,
                                                 sizes, strides);
  Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
  return b.create<tensor::ExtractOp>(loc, slice,
                                     SmallVector<Value>(rank, zero));
}

struct ScalarizeLinalgOp : public OpInterfaceRewritePattern<LinalgOp> {
  using OpInterfaceRewritePattern<LinalgOp>::OpInterfaceRewritePattern;

  static LogicalResult inlinePayload(PatternRewriter &rewriter, Location loc,
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
    for (auto [resultType, yieldOperand] :
         llvm::zip(linalgOp->getResultTypes(),
                   body->getTerminator()->getOperands())) {
      auto scalarValue = map.lookupOrDefault(yieldOperand);
      newResults.push_back(
          rewriter.create<FromElementsOp>(loc, resultType, scalarValue));
    }
    rewriter.replaceOp(linalgOp, newResults);
    return success();
  }

  LogicalResult matchAndRewrite(LinalgOp linalgOp,
                                PatternRewriter &rewriter) const override {
    // Fail if not every argument is a scalar or a single-element tensor.
    if (!hasSingleElementOperandsAndResults(linalgOp)) return failure();

    // TODO(aliia): fix scalarization of FillOp.
    if (auto *fillOp = dyn_cast<linalg::FillOp>(&linalgOp)) return failure();

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

struct ScalarizeScatterOp : public OpRewritePattern<thlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    Location loc = scatterOp.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    auto scatterIndices = extractStartIndices(b, scatterOp.getIndices());
    if (!scatterIndices) return failure();

    // Create the loop nest that spans window dimensions of `updates`.
    Value updates = scatterOp.getUpdates();
    auto updatesType = updates.getType().dyn_cast<RankedTensorType>();
    if (!updatesType) return failure();
    int64_t updatesRank = updatesType.getRank();

    SmallVector<OpFoldResult> updatesDimSizes =
        tensor::getMixedSizes(b, loc, updates);
    auto updatesDimValues =
        getValueOrCreateConstantIndexOp(b, loc, updatesDimSizes);

    Value init = scatterOp.getInit();
    auto initType = init.getType().dyn_cast<RankedTensorType>();
    if (!initType) return failure();

    SmallVector<OpFoldResult> initDimSizes =
        tensor::getMixedSizes(b, loc, init);
    auto initDimValues = getValueOrCreateConstantIndexOp(b, loc, initDimSizes);

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);

    // Create a loop that spans the dimensions of the update slice.
    SmallVector<Value> lbs(updatesRank, zero);
    SmallVector<Value> steps(updatesRank, one);

    SmallVector<Value> limitIndex{
        ArrayRef<Value>(updatesDimValues).drop_front()};
    for (const auto &en : llvm::enumerate(*scatterIndices)) {
      limitIndex[en.index()] =
          b.create<arith::AddIOp>(loc, limitIndex[en.index()], en.value());
    }
    for (auto &value : limitIndex) {
      value = b.create<arith::SubIOp>(loc, value, one);
    }

    Value indexIsInBounds =
        isValidIndex(b, loc, limitIndex, initDimValues, zero);
    indexIsInBounds = b.create<arith::AndIOp>(
        loc, indexIsInBounds,
        isValidIndex(b, loc, *scatterIndices, initDimValues, zero));
    auto ifOp = b.create<scf::IfOp>(
        loc, indexIsInBounds,
        [&](OpBuilder &thenBuilder, Location thenLoc) {
          scf::LoopNest loopNest = scf::buildLoopNest(
              thenBuilder, thenLoc, lbs, updatesDimValues, steps,
              ValueRange{init},
              [&](OpBuilder &nestedBuilder, Location bodyLoc,
                  ValueRange updateIndex, ValueRange loopInits) {
                Value initBlockArg = loopInits.front();

                auto initIndex = llvm::to_vector(updateIndex.drop_front());
                for (const auto &en : llvm::enumerate(*scatterIndices)) {
                  initIndex[en.index()] = nestedBuilder.create<arith::AddIOp>(
                      bodyLoc, initIndex[en.index()], en.value());
                }

                Value updateValue = materializePoint(
                    thenBuilder, loc, updates, getAsOpFoldResult(updateIndex));
                Value currentValue =
                    materializePoint(thenBuilder, loc, initBlockArg,
                                     getAsOpFoldResult(initIndex));

                // Combine update with the value in the output.
                Block *body = scatterOp.getBody();
                IRMapping bvm;
                bvm.map(body->getArgument(0), updateValue);
                bvm.map(body->getArgument(1), currentValue);

                for (Operation &op : body->without_terminator())
                  thenBuilder.clone(op, bvm);

                // Wrap every scalar result into a tensor using
                // `tensor.from_elements`.
                auto combinedValue =
                    bvm.lookup(body->getTerminator()->getOperand(0));
                Value updatedInit = thenBuilder.create<InsertOp>(
                    thenLoc, combinedValue, initBlockArg, initIndex);

                return scf::ValueVector({updatedInit});
              });

          thenBuilder.create<scf::YieldOp>(thenLoc, loopNest.results);
        },
        [&](OpBuilder &elseBuilder, Location elseLoc) {
          elseBuilder.create<scf::YieldOp>(elseLoc, init);
        });
    rewriter.replaceOp(scatterOp, ifOp.getResults());
    return success();
  }

 private:
  // Return i1 value after checking that 0 <= indices < dims(tensor).
  Value isValidIndex(OpBuilder &b, Location loc, ArrayRef<Value> indices,
                     ArrayRef<Value> tensorDims, Value zero) const {
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
};

struct ScalarizeGatherOp : public OpRewritePattern<thlo::GatherOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::GatherOp gatherOp,
                                PatternRewriter &rewriter) const override {
    Location loc = gatherOp.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);
    auto startIndices = extractStartIndices(b, gatherOp.getStartIndices());
    if (!startIndices) return failure();

    TypedValue<ShapedType> init = gatherOp.getInit();
    ShapedType initTy = init.getType();
    int64_t initRank = initTy.getRank();
    SmallVector<OpFoldResult> initDimSizes =
        tensor::getMixedSizes(b, loc, init);
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
    rewriter.replaceOpWithNewOp<gml_st::ForOp>(
        gatherOp, TypeRange(ValueRange{init}), lbs, initDimSizeValues, steps,
        init,
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
          Value tile = nestedBuilder.create<gml_st::TileOp>(
              bodyLoc, SmallVector<OpFoldResult>(ivs), ones, ones);
          Value val = materializePoint(nestedBuilder, bodyLoc, operand,
                                       getAsOpFoldResult(readIndices));
          nestedBuilder.create<gml_st::SetYieldOp>(bodyLoc, val,
                                                   loopInits.front(), tile);
        });
    return success();
  }
};

// Replace `thlo.concatenate` that is statically known to have only one element
// in concatenation dimension in all the inputs with
// `gml_st.materialize/tensor.insert_slice`.
struct ScalarizeConcatenateOp : public OpRewritePattern<thlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ConcatenateOp concatenateOp,
                                PatternRewriter &rewriter) const override {
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
      return b.create<tensor::InsertSliceOp>(l, slice, initTensor, offsets,
                                             sizes, strides);
    };

    Value res =
        extractElementFromInputs(rewriter, loc, concatenateOp.getInputs(),
                                 initType, concatDim, materializeAndInsert);

    rewriter.replaceOp(concatenateOp, res);

    return success();
  }

 private:
  Value tensorHasElement(OpBuilder &b, Location loc, Value input,
                         int64_t concatDim) const {
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value concatDimSize = b.create<tensor::DimOp>(loc, input, concatDim);
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, concatDimSize,
                                   zero);
  }

  Value extractElementFromInputs(
      OpBuilder &b, Location loc, ValueRange inputs, Type resultType,
      int64_t concatDim,
      llvm::function_ref<Value(OpBuilder &, Location, Value)>
          materializeAndInsert) const {
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
};

namespace {
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

}  // namespace

struct ScalarizeDynamicBroadcastInDimOp
    : public OpRewritePattern<thlo::DynamicBroadcastInDimOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::DynamicBroadcastInDimOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto input = broadcastOp.getOperand();
    auto output = broadcastOp.getInit();
    return scalarizeOp(broadcastOp, rewriter, input, output);
  }
};

struct ScalarizeReverseOp : public OpRewritePattern<thlo::ReverseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ReverseOp reverseOp,
                                PatternRewriter &rewriter) const override {
    auto input = reverseOp.getInput();
    auto output = reverseOp.getInit();
    return scalarizeOp(reverseOp, rewriter, input, output);
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

void populateTensorInsertExtractFoldingPatterns(RewritePatternSet *patterns) {
  patterns->add<FoldTensorFromElementsIntoSetYield>(patterns->getContext());
}

struct ScalarizationPass
    : public impl::ScalarizationPassBase<ScalarizationPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();

    RewritePatternSet patterns(context);
    // clang-format off
    patterns.add<
        ScalarizeConcatenateOp,
        ScalarizeDynamicBroadcastInDimOp,
        ScalarizeGatherOp,
        ScalarizeLinalgOp,
        ScalarizeReverseOp,
        ScalarizeScatterOp
    >(context);
    // clang-format on
    populateTensorInsertExtractFoldingPatterns(&patterns);
    FromElementsOp::getCanonicalizationPatterns(patterns, context);
    gml_st::ForOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createScalarizationPass() {
  return std::make_unique<ScalarizationPass>();
}

}  // namespace gml_st
}  // namespace mlir
