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
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_VECTORIZEGMLSTLOOPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

using mlir::linalg::FillOp;
using mlir::linalg::GenericOp;
using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

// The upper limit for vectorization of untiled `linalg.fill`. If a tensor has a
// static shape with more elements, then `linalg.fill` won't be vectorized. It
// is expected that such operations are tiled to get to small static shapes.
constexpr int64_t kNumElementsThreshold = 1024;

// Rewrite `vector.transfer_read(linalg.expand_shape)` as
// `vector.shape_cast(vector.transfer_read)`.
struct TransferReadOfOneDimExpandShape
    : public mlir::OpRewritePattern<TransferReadOp> {
  using OpRewritePattern<TransferReadOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      TransferReadOp vectorRead,
      mlir::PatternRewriter &rewriter) const override {
    auto expand = vectorRead.getSource().getDefiningOp<ExpandShapeOp>();
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

    auto newRead = rewriter.create<TransferReadOp>(
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

template <typename OpTy>
struct VectorizationPattern : public mlir::OpRewritePattern<OpTy> {
  VectorizationPattern(MLIRContext *context,
                       llvm::function_ref<bool(OpTy)> matchFn,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<OpTy>(context, benefit), matchFn(matchFn) {}

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (!matchFn(op)) return failure();
    return mlir::linalg::vectorize(rewriter, op);
  }

 private:
  llvm::function_ref<bool(OpTy)> matchFn;
};

RewritePatternSet getDefaultVectorizationPatterns(MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  mlir::vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<mlir::linalg::LinalgCopyVTRForwardingPattern,
               mlir::linalg::LinalgCopyVTWForwardingPattern>(ctx,
                                                             /*benefit=*/2);
  TransferReadOp::getCanonicalizationPatterns(patterns, ctx);
  TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  return patterns;
}

bool isInsideGmlStLoop(Operation *op) {
  Operation *parent = op->getParentOp();
  return isa<LoopOp>(parent) || isa<ParallelOp>(parent) || isa<ForOp>(parent);
}
bool isFillTiledOrSmall(FillOp fill) {
  if (isInsideGmlStLoop(fill)) return true;

  // Allow vectorization for static shapes with low number of elements.
  auto outputType = fill.output().getType().cast<mlir::RankedTensorType>();
  return outputType.hasStaticShape() &&
         outputType.getNumElements() < kNumElementsThreshold;
}

bool isGenericOpTiledOrOneDimReduction(GenericOp generic) {
  if (isInsideGmlStLoop(generic)) return true;

  // Allow vectorization of 1D reductions.
  return generic.getNumLoops() == 1 && generic.getNumReductionLoops() == 1;
}

struct VectorizeGmlStLoopsPass
    : public impl::VectorizeGmlStLoopsPassBase<VectorizeGmlStLoopsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns = getDefaultVectorizationPatterns(ctx);
    patterns.add<TransferReadOfOneDimExpandShape>(func.getContext());
    patterns.add<VectorizationPattern<FillOp>>(ctx, isFillTiledOrSmall);
    patterns.add<VectorizationPattern<GenericOp>>(
        ctx, isGenericOpTiledOrOneDimReduction);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeGmlStLoopsPass() {
  return std::make_unique<VectorizeGmlStLoopsPass>();
}

}  // namespace gml_st
}  // namespace mlir
