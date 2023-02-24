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

#include "gml_st/transforms/vectorization/vectorization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpDefinition.h"

namespace mlir {
namespace gml_st {
namespace {

using mlir::tensor::ExpandShapeOp;
using mlir::vector::TransferReadOp;
using mlir::vector::TransferWriteOp;

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

}  // namespace

void populateTransferReadOfOneDimExpandShapePattern(
    RewritePatternSet &patterns) {
  patterns.add<TransferReadOfOneDimExpandShape>(patterns.getContext());
}

RewritePatternSet getDefaultVectorizationPatterns(MLIRContext *ctx) {
  RewritePatternSet patterns(ctx);
  vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
  vector::populateVectorReductionToContractPatterns(patterns);
  patterns.add<linalg::LinalgCopyVTRForwardingPattern,
               linalg::LinalgCopyVTWForwardingPattern>(ctx, /*benefit=*/2);
  TransferWriteOp::getCanonicalizationPatterns(patterns, ctx);
  return patterns;
}

}  // namespace gml_st
}  // namespace mlir
