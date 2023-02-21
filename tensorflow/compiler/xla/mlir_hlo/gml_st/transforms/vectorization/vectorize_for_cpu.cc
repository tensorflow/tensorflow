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

#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "gml_st/transforms/vectorization/vectorization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "thlo/IR/thlo_ops.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_VECTORIZEFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

using mlir::linalg::BroadcastOp;
using mlir::linalg::DotOp;
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::MapOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::MatvecOp;
using mlir::linalg::Mmt4DOp;
using mlir::linalg::ReduceOp;
using mlir::linalg::TransposeOp;
using mlir::linalg::VecmatOp;
using mlir::tensor::ExpandShapeOp;
using mlir::thlo::ReverseOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

struct VectorizeIfOpPattern : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  VectorizeIfOpPattern(MLIRContext *ctx,
                       llvm::function_ref<bool(scf::IfOp)> matchFn,
                       PatternBenefit benefit = 1)
      : OpRewritePattern<scf::IfOp>(ctx, benefit), filterFn(matchFn) {}

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    if (!filterFn(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");

    int64_t numResults = op.getNumResults();
    if (numResults == 0) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot vectorize if op w/o results");
    }

    // Derive vectorized types.
    SmallVector<Type> vectorizedTypes;
    vectorizedTypes.reserve(numResults);
    for (Type ty : op.getResultTypes()) {
      auto rankedTy = ty.dyn_cast<RankedTensorType>();
      if (rankedTy && rankedTy.hasStaticShape()) {
        vectorizedTypes.push_back(
            VectorType::get(rankedTy.getShape(), rankedTy.getElementType()));
      } else {
        vectorizedTypes.push_back(ty);
      }
    }

    // Create vectorized if op and steal bodies.
    Location loc = op.getLoc();
    auto vectorizedIfOp =
        rewriter.create<scf::IfOp>(loc, vectorizedTypes, op.getCondition());
    vectorizedIfOp.getThenRegion().takeBody(op.getThenRegion());
    vectorizedIfOp.getElseRegion().takeBody(op.getElseRegion());

    // Insert `transfer_read/write` ops for type compatibility.
    auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    SmallVector<Value> replacements(vectorizedIfOp.getResults());
    for (int64_t i = 0; i < numResults; ++i) {
      // Skip non-vectorizable values.
      auto vectorTy = vectorizedTypes[i].dyn_cast<VectorType>();
      if (!vectorTy) continue;

      // Yield vectorized value in then-case.
      rewriter.setInsertionPoint(vectorizedIfOp.thenYield());
      SmallVector<Value> indices(vectorTy.getRank(), zero);
      Value unvectorizedThen = vectorizedIfOp.thenYield().getOperand(i);
      Value vectorizedThen = rewriter.create<vector::TransferReadOp>(
          loc, vectorTy, unvectorizedThen, indices);
      vectorizedIfOp.thenYield().setOperand(i, vectorizedThen);

      // Yield vectorized value in else-case.
      rewriter.setInsertionPoint(vectorizedIfOp.elseYield());
      Value unvectorizedElse = vectorizedIfOp.elseYield().getOperand(i);
      Value vectorizedElse = rewriter.create<vector::TransferReadOp>(
          loc, vectorTy, unvectorizedElse, indices);
      vectorizedIfOp.elseYield().setOperand(i, vectorizedElse);

      // Insert `transfer_write` op after the vectorized if op for type
      // compatibility.
      rewriter.setInsertionPointAfter(vectorizedIfOp);
      Value init = rewriter.create<tensor::EmptyOp>(
          loc, vectorTy.getShape(), vectorTy.getElementType(), ValueRange{});
      replacements[i] = rewriter
                            .create<vector::TransferWriteOp>(
                                loc, vectorizedIfOp.getResult(i), init, indices)
                            .getResult();
    }

    // Replace op.
    rewriter.replaceOp(op, replacements);
    return success();
  }

 private:
  llvm::function_ref<bool(scf::IfOp)> filterFn;
};

// TODO(b/269643522): Upstream this as a canonicalization for `scf.if`.
struct InlineCastInIfOpPattern : public OpRewritePattern<tensor::CastOp> {
  using OpRewritePattern<tensor::CastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::CastOp op,
                                PatternRewriter &rewriter) const override {
    auto srcTy = op.getSource().getType().cast<RankedTensorType>();
    auto dstTy = op.getType().cast<RankedTensorType>();
    if (srcTy.hasStaticShape() || !dstTy.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          op, "not cast from dynamic to static shape");
    }

    if (!op.getSource().hasOneUse())
      return rewriter.notifyMatchFailure(op, "source has more than one use");

    auto ifOp = op.getSource().getDefiningOp<scf::IfOp>();
    if (!ifOp || ifOp.getNumResults() != 1) {
      return rewriter.notifyMatchFailure(
          op, "source is not an if op with a unique result");
    }

    // Determine result types for the new if op.
    SmallVector<Type> newResultTypes(ifOp.getResultTypes());
    auto ifOpResult = llvm::cast<OpResult>(op.getSource());
    int64_t resultIdx = ifOpResult.getResultNumber();
    newResultTypes[resultIdx] = dstTy;

    // Create new if op and steal bodies.
    rewriter.setInsertionPoint(ifOp);
    Location loc = ifOp.getLoc();
    auto newIfOp =
        rewriter.create<scf::IfOp>(loc, newResultTypes, ifOp.getCondition());
    newIfOp.getThenRegion().takeBody(ifOp.getThenRegion());
    newIfOp.getElseRegion().takeBody(ifOp.getElseRegion());

    // Insert inner casts.
    rewriter.setInsertionPoint(newIfOp.thenYield());
    newIfOp.thenYield().setOperand(
        resultIdx, rewriter.create<tensor::CastOp>(
                       loc, dstTy, newIfOp.thenYield().getOperand(resultIdx)));
    rewriter.setInsertionPoint(newIfOp.elseYield());
    newIfOp.elseYield().setOperand(
        resultIdx, rewriter.create<tensor::CastOp>(
                       loc, dstTy, newIfOp.elseYield().getOperand(resultIdx)));

    // Replace op.
    rewriter.replaceOp(op, newIfOp.getResults());
    rewriter.eraseOp(ifOp);
    return success();
  }
};

// Rewrite `vector.transfer_read(linalg.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfOneDimExpandShape
    : public mlir::OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      vector::TransferReadOp vectorRead,
      mlir::PatternRewriter &rewriter) const override {
    auto expand = vectorRead.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand) return failure();

    auto expandSrc = expand.getSrc();
    auto expandSrcType = expand.getSrcType();
    auto expandDstType = expand.getResultType();
    if (expandSrcType.getRank() != 1 || expandDstType.getRank() != 2)
      return failure();

    auto resultType = vectorRead.getType().dyn_cast<mlir::ShapedType>();
    if (!resultType || resultType.getShape() != expandDstType.getShape())
      return failure();

    auto zero = rewriter.create<arith::ConstantIndexOp>(vectorRead.getLoc(), 0);
    auto map = mlir::AffineMap::get(1, 0, {rewriter.getAffineDimExpr(0)},
                                    vectorRead.getContext());
    // TODO(pifon): Also support canonicalization in case the map is not an
    // identity.
    if (!map.isIdentity()) return failure();

    auto newRead = rewriter.create<vector::TransferReadOp>(
        vectorRead.getLoc(),
        mlir::VectorType::get(expandSrcType.getShape(),
                              expandSrcType.getElementType()),
        expandSrc, mlir::ValueRange{zero}, mlir::AffineMapAttr::get(map),
        vectorRead.getPadding(),
        /*mask=*/mlir::Value(), rewriter.getBoolArrayAttr({true}));
    rewriter.replaceOpWithNewOp<mlir::vector::ShapeCastOp>(
        vectorRead, vectorRead.getType(), newRead);
    return success();
  }
};

// This currently matches for all thlo.reverse of the form 1x1x..x1xVectorSize.
// DimSize < kNumElementsVectorization will be handled by Scalarization.
bool isPerfectlyTiledReverse(thlo::ReverseOp reverseOp) {
  auto inputType = reverseOp.getInput().getType();
  for (unsigned i = 0; i < inputType.getRank(); ++i) {
    if (inputType.isDynamicDim(i)) {
      return false;
    }
    if (i == inputType.getRank() - 1) {
      return inputType.getDimSize(i) == kNumElementsVectorization &&
             llvm::is_contained(reverseOp.getReverseDimensions(), i);
    }
    if (inputType.getDimSize(i) != 1) {
      return false;
    }
  }
  return false;
}

// Rewrite thlo.reverse of pattern 1x1x..x1xVectorSize as vector.transfer_read
// followed by vector.shuffle followed by vector.transfer_write.
struct ThloReverseVectorizationPattern
    : public mlir::OpRewritePattern<thlo::ReverseOp> {
  explicit ThloReverseVectorizationPattern(MLIRContext *context,
                                           mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<thlo::ReverseOp>(context, benefit) {}

  LogicalResult matchAndRewrite(thlo::ReverseOp op,
                                PatternRewriter &rewriter) const override {
    if (!isPerfectlyTiledReverse(op))
      return rewriter.notifyMatchFailure(op, "did not match filter");

    auto inputType = op.getInput().getType();
    auto vecTargetType =
        RankedTensorType::get(inputType.getShape()[inputType.getRank() - 1],
                              inputType.getElementType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Value> indices(op.getInit().getType().getRank(), zero);

    auto readInput = rewriter.create<vector::TransferReadOp>(
        op.getLoc(),
        VectorType::get(vecTargetType.getShape(),
                        vecTargetType.getElementType()),
        op.getInput(), indices);

    SmallVector<int64_t> mask;
    int64_t maskSize = inputType.getShape()[inputType.getRank() - 1];
    mask.reserve(maskSize);
    for (int64_t i = maskSize - 1; i >= 0; --i) {
      mask.push_back(i);
    }
    auto shuffle = rewriter.create<vector::ShuffleOp>(op.getLoc(), readInput,
                                                      readInput, mask);

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, shuffle.getResult(), op.getInit(), indices);
    return success();
  }
};

struct IdentityTransposeOpFoldingPattern
    : public OpRewritePattern<TransposeOp> {
  explicit IdentityTransposeOpFoldingPattern(MLIRContext *context,
                                             PatternBenefit benefit = 1)
      : OpRewritePattern(context, benefit) {}

  LogicalResult matchAndRewrite(TransposeOp op,
                                PatternRewriter & /*rewriter*/) const override {
    auto perm = op.getPermutation();
    for (int64_t i = 0; static_cast<uint64_t>(i) < perm.size(); ++i) {
      if (perm[i] != i) return failure();
    }

    if (!hasSingleElementOperandsAndResults(op)) return failure();

    op.replaceAllUsesWith(SmallVector<Value>(1, op.getInput()));
    return success();
  }
};

bool isSmallTensorOrScalar(Type ty) {
  auto rankedTy = ty.dyn_cast<mlir::RankedTensorType>();
  bool isSmallTensor = rankedTy && rankedTy.hasStaticShape() &&
                       rankedTy.getNumElements() < kNumElementsThreshold;
  bool isScalar = !isa<ShapedType>(ty);
  return isSmallTensor || isScalar;
}

struct VectorizeForCPUPass
    : public impl::VectorizeForCPUPassBase<VectorizeForCPUPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    auto hasSmallStaticallyShapedTensorResults = [&](Operation *op) {
      return llvm::all_of(op->getOperandTypes(), isSmallTensorOrScalar) &&
             llvm::all_of(op->getResultTypes(), isSmallTensorOrScalar);
    };
    auto isInsidePerfectlyTiledLoop = [&](Operation *op) {
      Operation *parent = op->getParentOp();
      return (isa<ParallelOp, scf::ForOp>(parent)) &&
             hasLabel(parent, kPerfectlyTiledLoopLabel);
    };
    auto isInsidePerfectlyTiledLoopOrSmall = [&](Operation *op) {
      return !hasSingleElementOperandsAndResults(op) &&
             (isInsidePerfectlyTiledLoop(op) ||
              hasSmallStaticallyShapedTensorResults(op));
    };
    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      // clang-format off
      patterns.add<
        VectorizeIfOpPattern,
        VectorizationPattern<BroadcastOp>,
        VectorizationPattern<FillOp>,
        VectorizationPattern<GenericOp>,
        VectorizationPattern<DotOp>,
        VectorizationPattern<MapOp>,
        VectorizationPattern<MatmulOp>,
        VectorizationPattern<MatvecOp>,
        VectorizationPattern<Mmt4DOp>,
        VectorizationPattern<ReduceOp>,
        VectorizationPattern<TransposeOp>,
        VectorizationPattern<VecmatOp>
      >(ctx, isInsidePerfectlyTiledLoopOrSmall);
      // clang-format on
      populateTransferReadOfOneDimExpandShapePattern(patterns);
      patterns.add<InlineCastInIfOpPattern, ThloReverseVectorizationPattern>(
          ctx);
      tensor::CastOp::getCanonicalizationPatterns(patterns, ctx);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      linalg::populatePadOpVectorizationPatterns(patterns);
      patterns.add<IdentityTransposeOpFoldingPattern>(ctx);
      if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    // Hoisting transfer_read/transfer_write.
    linalg::hoistRedundantVectorTransfersOnTensor(func);
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeForCPUPass() {
  return std::make_unique<VectorizeForCPUPass>();
}

}  // namespace gml_st
}  // namespace mlir
