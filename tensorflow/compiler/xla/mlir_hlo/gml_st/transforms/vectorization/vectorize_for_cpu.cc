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
#include "llvm/Support/Casting.h"
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

struct PassVectorizedValuesThroughIfOpPattern
    : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern<scf::IfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp op,
                                PatternRewriter &rewriter) const override {
    int64_t numResults = op.getNumResults();
    if (numResults == 0) {
      return rewriter.notifyMatchFailure(op,
                                         "cannot vectorize if op w/o results");
    }

    // Derive vectorized types.
    SmallVector<Type> vectorizedTypes(op.getResultTypes());
    int64_t numActuallyVectorizedTypes = 0;
    scf::YieldOp thenYieldOp = op.thenYield();
    scf::YieldOp elseYieldOp = op.elseYield();
    for (int64_t i = 0; i < numResults; ++i) {
      Value result = op.getResult(i);

      // Can only vectorized statically shaped results.
      auto rankedTy = result.getType().dyn_cast<RankedTensorType>();
      if (!rankedTy || !rankedTy.hasStaticShape()) continue;

      // Vectorize only results that are either always used as a vector or
      // always produced as a vector.
      bool allVectorConsumers =
          llvm::all_of(result.getUsers(), [](Operation *user) {
            return llvm::isa_and_nonnull<vector::TransferReadOp>(user);
          });
      bool allVectorProducers =
          llvm::isa_and_nonnull<vector::TransferWriteOp>(
              thenYieldOp.getOperand(i).getDefiningOp()) &&
          llvm::isa_and_nonnull<vector::TransferWriteOp>(
              elseYieldOp.getOperand(i).getDefiningOp());
      if (!allVectorProducers && !allVectorConsumers) continue;

      // Derive vectorized type.
      vectorizedTypes[i] =
          VectorType::get(rankedTy.getShape(), rankedTy.getElementType());
      numActuallyVectorizedTypes++;
    }

    // Fail if there isn't anything to vectorize.
    if (numActuallyVectorizedTypes == 0) {
      return rewriter.notifyMatchFailure(op, "nothing to vectorize");
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
    if (!VectorType::isValidElementType(inputType.getElementType())) {
      return rewriter.notifyMatchFailure(op, "cannot be vectorized");
    }
    auto vecTargetType =
        VectorType::get(inputType.getShape()[inputType.getRank() - 1],
                        inputType.getElementType());
    Value zero = rewriter.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    SmallVector<Value> indices(op.getInit().getType().getRank(), zero);

    auto readInput = rewriter.create<vector::TransferReadOp>(
        op.getLoc(), vecTargetType, op.getInput(), indices);

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

struct VectorizeForCPUPass
    : public impl::VectorizeForCPUPassBase<VectorizeForCPUPass> {
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    auto isNonComplexSmallTensorOrScalar = [&](Type ty) {
      if (getElementTypeOrSelf(ty).isa<ComplexType>()) return false;
      if (auto rankedTy = ty.dyn_cast<mlir::RankedTensorType>()) {
        return rankedTy.hasStaticShape() &&
               rankedTy.getNumElements() < numElementsThreshold;
      }

      return !isa<ShapedType>(ty);
    };

    auto isOpOnNonComplexSmallTensorOrScalar = [&](Operation *op) {
      return llvm::all_of(op->getOperandTypes(),
                          isNonComplexSmallTensorOrScalar) &&
             llvm::all_of(op->getResultTypes(),
                          isNonComplexSmallTensorOrScalar);
    };
    auto isInsidePerfectlyTiledLoop = [&](Operation *op) {
      Operation *parent = op->getParentOp();
      return (isa<scf::ForallOp, scf::ForOp>(parent)) &&
             hasLabel(parent, kPerfectlyTiledLoopLabel);
    };
    auto isInsidePerfectlyTiledLoopOrSmall = [&](Operation *op) {
      return !hasSingleElementOperandsAndResults(op) &&
             (isInsidePerfectlyTiledLoop(op) ||
              isOpOnNonComplexSmallTensorOrScalar(op));
    };
    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      // clang-format off
      patterns.add<
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
      patterns.add<PassVectorizedValuesThroughIfOpPattern>(ctx);
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

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeForCPUPass(
    int64_t numElementsThreshold) {
  VectorizeForCPUPassOptions opts;
  opts.numElementsThreshold = numElementsThreshold;
  return std::make_unique<VectorizeForCPUPass>(opts);
}

}  // namespace gml_st
}  // namespace mlir
