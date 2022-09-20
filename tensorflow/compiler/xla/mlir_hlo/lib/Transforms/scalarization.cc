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
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
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
      auto bbArg = genericOp.getTiedBlockArgument(&opOperand);
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
    for (Operation &op : body->without_terminator()) rewriter.clone(op, bvm);

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

struct ScalarizeScatterOp : public OpRewritePattern<thlo::ScatterOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(thlo::ScatterOp scatterOp,
                                PatternRewriter &rewriter) const override {
    Location loc = scatterOp.getLoc();
    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Extract update.
    Value updates = scatterOp.getUpdates();
    auto updatesType = updates.getType().dyn_cast<RankedTensorType>();
    if (!updatesType || !hasSingleElement(updatesType)) return failure();
    SmallVector<Value> updateIndices(updatesType.getRank(), zero);
    Value updateValue = rewriter.create<ExtractOp>(loc, updates, updateIndices);

    // Extract/compute index.
    Value indices = scatterOp.getIndices();
    auto indicesType = indices.getType().dyn_cast<RankedTensorType>();
    SmallVector<Value> indicesIndices(indicesType.getRank(), zero);

    Value init = scatterOp.getInit();
    auto initType = init.getType().dyn_cast<RankedTensorType>();

    SmallVector<Value> scatterIndices;
    Type indexType = rewriter.getIndexType();
    for (int64_t i = 0, e = initType.getRank(); i < e; ++i) {
      indicesIndices.back() = rewriter.create<arith::ConstantIndexOp>(loc, i);
      Value index = rewriter.create<ExtractOp>(loc, indices, indicesIndices);
      if (index.getType() != indexType)
        index = rewriter.create<arith::IndexCastOp>(loc, indexType, index);
      scatterIndices.push_back(index);
    }

    // Check if the computed index is within bounds.
    Value indexIsInBounds = isValidIndex(rewriter, loc, init, scatterIndices);
    Value maybeUpdatedInit =
        rewriter
            .create<scf::IfOp>(
                loc, initType, indexIsInBounds,
                [&](OpBuilder &thenBuilder, Location thenLoc) {
                  // Extract the current value from the output tensor.
                  Value currentValue =
                      rewriter.create<ExtractOp>(loc, init, scatterIndices);

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
                      thenLoc, combinedValue, init, scatterIndices);
                  rewriter.create<scf::YieldOp>(thenLoc, updatedInit);
                },
                [&](OpBuilder &elseBuilder, Location elseLoc) {
                  elseBuilder.create<scf::YieldOp>(elseLoc, init);
                })
            .getResult(0);
    rewriter.replaceOp(scatterOp, maybeUpdatedInit);
    return success();
  }

 private:
  // Return i1 value after checking that 0 <= indices < dims(tensor).
  Value isValidIndex(OpBuilder &b, Location loc, Value tensor,
                     ArrayRef<Value> indices) const {
    auto i1Type = b.getIntegerType(1);
    Value isValid = b.create<arith::ConstantOp>(
        loc, i1Type, IntegerAttr::get(i1Type, APInt(1, 1)));

    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> tensorDims = tensor::createDimValues(b, loc, tensor);
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
    patterns.add<ScalarizeGenericOp, ScalarizeScatterOp>(context);
    populateTensorInsertExtractFoldingPatterns(&patterns);
    FromElementsOp::getCanonicalizationPatterns(patterns, context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createScalarizationPass() {
  return std::make_unique<ScalarizationPass>();
}

}  // namespace mlir
