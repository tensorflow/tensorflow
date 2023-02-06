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
using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::linalg::MapOp;
using mlir::linalg::MatmulOp;
using mlir::linalg::Mmt4DOp;
using mlir::linalg::ReduceOp;
using mlir::linalg::TransposeOp;
using mlir::tensor::ExpandShapeOp;
using mlir::thlo::ReverseOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

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

bool isInsideGmlStLoop(Operation *op) {
  Operation *parent = op->getParentOp();
  return isa<ParallelOp>(parent) || isa<ForOp>(parent);
}

bool isFillTiledOrSmall(linalg::FillOp fill) {
  if (isInsideGmlStLoop(fill)) return true;

  // Allow vectorization for static shapes with low number of elements.
  auto outputType = fill.output().getType().dyn_cast<mlir::RankedTensorType>();
  return outputType && outputType.hasStaticShape() &&
         outputType.getNumElements() < kNumElementsThreshold;
}

struct VectorizeForCPUPass
    : public impl::VectorizeForCPUPassBase<VectorizeForCPUPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    auto hasSmallStaticOutputs = [&](Operation *op) {
      return llvm::all_of(op->getResultTypes(), [](Type type) {
        auto outputType = type.dyn_cast<mlir::RankedTensorType>();
        return outputType && outputType.hasStaticShape() &&
               outputType.getNumElements() < kNumElementsThreshold;
      });
    };
    auto isPerfectlyTiledLoop = [&](Operation *op) {
      return (isa<ForOp, ParallelOp, scf::ForOp>(op)) &&
             hasLabel(op, kPerfectlyTiledLoopLabel);
    };
    auto isInsidePerfectlyTiledLoop = [&](Operation *op) {
      return isPerfectlyTiledLoop(op->getParentOp());
    };
    auto isInsidePerfectlyTiledLoopOrSmall = [&](Operation *op) {
      return !hasSingleElementOperandsAndResults(op) &&
             (isInsidePerfectlyTiledLoop(op) || hasSmallStaticOutputs(op));
    };
    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      // clang-format off
      patterns.add<
        VectorizationPattern<BroadcastOp>,
        VectorizationPattern<GenericOp>,
        VectorizationPattern<MapOp>,
        VectorizationPattern<MatmulOp>,
        VectorizationPattern<Mmt4DOp>,
        VectorizationPattern<ReduceOp>,
        VectorizationPattern<TransposeOp>
      >(ctx, isInsidePerfectlyTiledLoopOrSmall);
      // clang-format on
      patterns.add<VectorizationPattern<FillOp>>(ctx, isFillTiledOrSmall);
      populateTransferReadOfOneDimExpandShapePattern(patterns);
      patterns.add<ThloReverseVectorizationPattern>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }

    {
      RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
      TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
      linalg::populatePadOpVectorizationPatterns(patterns);
      patterns.add<IdentityTransposeOpFoldingPattern>(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    }

    // Hoisting transfer_read/transfer_write.
    {
      RewritePatternSet patterns(ctx);
      (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

      hoistRedundantVectorTransfersOnTensor(func);
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
