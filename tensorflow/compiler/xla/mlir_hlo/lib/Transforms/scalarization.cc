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

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/thlo/IR/thlo_ops.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_SCALARIZATIONPASS
#include "mlir-hlo/Transforms/passes.h.inc"

using linalg::GenericOp;
using tensor::ExtractOp;
using tensor::FromElementsOp;
using tensor::InsertOp;

template <typename ShapedTy>
bool hasSingleElement(ShapedTy type) {
  return type.hasStaticShape() && type.getNumElements() == 1;
}

struct ScalarizeGenericOp : public OpRewritePattern<GenericOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto isNonScalar = [](Type type) {
      return type.isa<TensorType>() &&
             !hasSingleElement(type.cast<TensorType>());
    };
    if (llvm::any_of(genericOp.getOperandTypes(), isNonScalar) ||
        llvm::any_of(genericOp.getResultTypes(), isNonScalar))
      return failure();

    // Map block arguments of genericOp to tensor.extract ops of its args.
    Location loc = genericOp.getLoc();
    BlockAndValueMapping bvm;
    for (OpOperand &opOperand : genericOp->getOpOperands()) {
      Value operandValue = opOperand.get();
      Type operandType = operandValue.getType();
      auto bbArg = genericOp.getMatchingBlockArgument(&opOperand);
      if (!operandType.isa<ShapedType>()) continue;

      SmallVector<Value> indices(
          operandType.cast<RankedTensorType>().getRank(),
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
      Value extractedElement =
          rewriter.create<ExtractOp>(loc, operandValue, indices);
      bvm.map(bbArg, extractedElement);
    }

    // Clone everything but terminator.
    Block *body = genericOp.getBody();
    for (Operation &op : body->without_terminator()) {
      // `linalg.index` can only result in 0 for scalar linalg.generic.
      if (auto indexOp = dyn_cast<linalg::IndexOp>(op)) {
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
        bvm.map(indexOp.getResult(), zero);
        continue;
      }
      rewriter.clone(op, bvm);
    }

    // Wrap every scalar result into a tensor using `tensor.from_elements`.
    SmallVector<Value> newResults;
    for (auto [resultType, yieldOperand] :
         llvm::zip(genericOp->getResultTypes(),
                   body->getTerminator()->getOperands())) {
      auto scalarValue = bvm.lookupOrDefault(yieldOperand);
      newResults.push_back(
          rewriter.create<FromElementsOp>(loc, resultType, scalarValue));
    }
    rewriter.replaceOp(genericOp, newResults);

    return success();
  }
};

// Extracts a point using gml_st.materialize and gml_st.tile with 1 element.
Value getPoint(OpBuilder &b, Location loc, Value tensor, Value space,
               ValueRange indices) {
  IntegerAttr oneAttr = b.getIndexAttr(1);

  auto tensorType = tensor.getType().cast<RankedTensorType>();
  int64_t tensorRank = tensorType.getRank();

  SmallVector<OpFoldResult> offsets(indices.begin(), indices.end());
  SmallVector<OpFoldResult> sizes(tensorRank, oneAttr);
  SmallVector<OpFoldResult> strides(tensorRank, oneAttr);

  Value tile = b.create<gml_st::TileOp>(loc, space, offsets, sizes, strides);
  return b.create<gml_st::MaterializeOp>(loc, tensorType.getElementType(),
                                         tensor, tile);
}

struct ScalarizeScatterOp : public OpRewritePattern<thlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    if (scatterOp.getIndicesCount() != 1) return failure();

    Location loc = scatterOp.getLoc();
    ImplicitLocOpBuilder b(loc, rewriter);

    // Create the loop nest that spans window dimensions of `updates`.
    Value updates = scatterOp.getUpdates();
    auto updatesType = updates.getType().dyn_cast<RankedTensorType>();
    int64_t updatesRank = updatesType.getRank();

    SmallVector<OpFoldResult> updatesDimSizes =
        tensor::getMixedSizes(b, loc, updates);
    auto updatesDimValues =
        getValueOrCreateConstantIndexOp(b, loc, updatesDimSizes);
    Value updatesSpace = b.create<gml_st::SpaceOp>(updatesDimSizes);

    Value indices = scatterOp.getIndices();

    Value init = scatterOp.getInit();
    auto initType = init.getType().dyn_cast<RankedTensorType>();

    SmallVector<OpFoldResult> initDimSizes =
        tensor::getMixedSizes(b, loc, init);
    auto initDimValues = getValueOrCreateConstantIndexOp(b, loc, initDimSizes);
    Value initSpace = b.create<gml_st::SpaceOp>(initDimSizes);

    Value zero = b.create<arith::ConstantIndexOp>(0);
    Value one = b.create<arith::ConstantIndexOp>(1);

    // Extract scatter indices.
    Type indexType = rewriter.getIndexType();
    SmallVector<Value> scatterIndices;
    SmallVector<Value> currentIndexInIndicesTensor(2, zero);
    for (int64_t i = 0, e = scatterOp.getIndexVectorDimSize(); i < e; ++i) {
      currentIndexInIndicesTensor.back() = b.create<arith::ConstantIndexOp>(i);
      Value index = b.create<ExtractOp>(indices, currentIndexInIndicesTensor);
      if (index.getType() != indexType)
        index = b.create<arith::IndexCastOp>(indexType, index);
      scatterIndices.push_back(index);
    }

    // Create a loop that spans the dimensions of the update slice.
    SmallVector<Value> lbs(updatesRank, zero);
    SmallVector<Value> steps(updatesRank, one);

    auto loop = b.create<gml_st::ForOp>(
        TypeRange(ValueRange{init}), lbs, updatesDimValues, steps, init,
        [&](OpBuilder &nestedBuilder, Location bodyLoc, ValueRange updateIndex,
            ValueRange loopInits) {
          Value initBlockArg = loopInits.front();

          auto initIndex = llvm::to_vector(updateIndex.drop_front());
          for (const auto &en : llvm::enumerate(scatterIndices)) {
            initIndex[en.index()] = nestedBuilder.create<arith::AddIOp>(
                bodyLoc, initIndex[en.index()], en.value());
          }

          Value indexIsInBounds =
              isValidIndex(nestedBuilder, loc, initIndex, initDimValues, zero);
          Value maybeUpdatedInit =
              nestedBuilder
                  .create<scf::IfOp>(
                      loc, initType, indexIsInBounds,
                      [&](OpBuilder &thenBuilder, Location thenLoc) {
                        Value updateValue = getPoint(thenBuilder, loc, updates,
                                                     updatesSpace, updateIndex);
                        Value currentValue =
                            getPoint(thenBuilder, loc, initBlockArg, initSpace,
                                     initIndex);

                        // Combine update with the value in the output.
                        Block *body = scatterOp.getBody();
                        BlockAndValueMapping bvm;
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
                        thenBuilder.create<scf::YieldOp>(thenLoc, updatedInit);
                      },
                      [&](OpBuilder &elseBuilder, Location elseLoc) {
                        elseBuilder.create<scf::YieldOp>(elseLoc, initBlockArg);
                      })
                  .getResult(0);

          nestedBuilder.create<gml_st::SetYieldOp>(bodyLoc, maybeUpdatedInit,
                                                   initBlockArg, initSpace);
        });
    rewriter.replaceOp(scatterOp, loop.getResults());
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

// Replace `thlo.concatenate` that has only one element in total in all the
// inputs with `tensor.extract/insert`.
struct ScalarizeConcatenateOp : public OpRewritePattern<thlo::ConcatenateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ConcatenateOp concatenateOp,
                                PatternRewriter &rewriter) const override {
    Location loc = concatenateOp.getLoc();
    int64_t concatDim = concatenateOp.getDimension();

    auto initTensor = concatenateOp.getInit();
    auto initType = initTensor.getType();
    int64_t rank = initTensor.getType().getRank();

    // Only scalarize an op that has 1-element tensor output.
    if (!hasSingleElement(initType)) {
      return failure();
    }

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> indices(rank, zero);

    // All inputs have only one element in total. Only one input has the concat
    // dimension equal to 1, other inputs have 0.
    Value element = extractElementFromInputs(
        rewriter, loc, concatenateOp.getInputs(), indices,
        initType.getElementType(), concatDim, zero);

    Value res = rewriter.create<InsertOp>(loc, element, initTensor, indices);

    rewriter.replaceOp(concatenateOp, res);

    return success();
  }

 private:
  Value tensorHasElement(OpBuilder &b, Location loc, Value input,
                         int64_t concatDim, Value zero) const {
    Value concatDimSize = b.create<tensor::DimOp>(loc, input, concatDim);
    return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ne, concatDimSize,
                                   zero);
  }

  Value extractElementFromInputs(OpBuilder &b, Location loc, ValueRange inputs,
                                 ArrayRef<Value> indices, Type resultType,
                                 int64_t concatDim, Value zero) const {
    if (inputs.size() == 1) {
      return b.create<ExtractOp>(loc, inputs.front(), indices);
    }

    return b
        .create<scf::IfOp>(
            loc, resultType,
            tensorHasElement(b, loc, inputs.front(), concatDim, zero),
            [&](OpBuilder &thenBuilder, Location thenLoc) {
              Value el = thenBuilder.create<ExtractOp>(thenLoc, inputs.front(),
                                                       indices);
              b.create<scf::YieldOp>(loc, el);
            },
            [&](OpBuilder &elseBuilder, Location elseLoc) {
              b.create<scf::YieldOp>(
                  loc, extractElementFromInputs(elseBuilder, elseLoc,
                                                inputs.drop_front(), indices,
                                                resultType, concatDim, zero));
            })
        .getResult(0);
  }
};

// Fold `tensor.extract(gml_st.materialize -> tensor<1x1xf32>)` into
//      `gml_st.materialize -> f32` for single-element tensors.
struct FoldTensorExtractIntoMaterialize : public OpRewritePattern<ExtractOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExtractOp extractOp,
                                PatternRewriter &rewriter) const override {
    auto materializeOp =
        extractOp.getTensor().getDefiningOp<gml_st::MaterializeOp>();
    if (!materializeOp) return failure();

    auto tileType =
        materializeOp.getSet().getType().dyn_cast<gml_st::TileType>();
    if (!tileType || !hasSingleElement(tileType)) return failure();

    rewriter.replaceOpWithNewOp<gml_st::MaterializeOp>(
        extractOp, extractOp.getType(), materializeOp.getSource(),
        materializeOp.getSet());
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

void populateTensorInsertExtractFoldingPatterns(RewritePatternSet *patterns) {
  patterns->add<FoldTensorExtractIntoMaterialize,
                FoldTensorFromElementsIntoSetYield>(patterns->getContext());
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
        ScalarizeGenericOp,
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

}  // namespace mlir
