/* Copyright 2023 The OpenXLA Authors.

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

#include <algorithm>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace {

#define GEN_PASS_DEF_VECTORIZECOPYPASS
#include "transforms/passes.h.inc"

/// Transforms a big non-contiguous `memref.copy` into a loop over smaller
/// copies that are either contiguous or can be vectorized.
struct TileCopyPattern : public OpRewritePattern<memref::CopyOp> {
  TileCopyPattern(MLIRContext *context, int64_t tileSize,
                  mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<memref::CopyOp>(context, benefit),
        tileSize(tileSize) {}
  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(op.getTarget().getType());

    if (!srcType || !targetType) return failure();

    if (!srcType.hasStaticShape() || !targetType.hasStaticShape())
      return failure();

    if (srcType.getShape() != targetType.getShape()) return failure();

    if (memref::isStaticShapeAndContiguousRowMajor(srcType) &&
        memref::isStaticShapeAndContiguousRowMajor(targetType)) {
      return failure();
    }

    if (srcType.getNumElements() <= tileSize) return failure();

    auto rank = srcType.getRank();
    auto shape = srcType.getShape();

    SmallVector<OpFoldResult> offsets(rank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> sizes;
    for (auto s : shape) sizes.push_back(rewriter.getIndexAttr(s));
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    createLoopsNest(rewriter, op.getLoc(), 0, op.getSource(), op.getTarget(),
                    shape, offsets, sizes, strides);

    rewriter.eraseOp(op);

    return success();
  }

 private:
  void createLoopsNest(PatternRewriter &rewriter, Location loc, int64_t dim,
                       Value src, Value target, ArrayRef<int64_t> shape,
                       SmallVector<OpFoldResult> &offsets,
                       SmallVector<OpFoldResult> &sizes,
                       SmallVector<OpFoldResult> &strides) const {
    auto srcType = dyn_cast<MemRefType>(src.getType());
    auto targetType = dyn_cast<MemRefType>(target.getType());

    const bool isContiguous =
        memref::isStaticShapeAndContiguousRowMajor(srcType) &&
        memref::isStaticShapeAndContiguousRowMajor(targetType);
    const bool isSmall = srcType.getNumElements() <= tileSize &&
                         targetType.getNumElements() <= tileSize;

    if (isContiguous || isSmall) {
      rewriter.create<memref::CopyOp>(loc, src, target);
      return;
    }

    const int64_t dimSize = shape[dim];
    const int64_t sliceSize =
        std::max((int64_t)1, tileSize * dimSize / srcType.getNumElements());

    const int64_t remainderSize = dimSize % sliceSize;
    const int64_t upperBound = shape[dim] - remainderSize;

    Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value tileSizeValue =
        rewriter.create<arith::ConstantIndexOp>(loc, sliceSize);
    Value upperBoundValue =
        rewriter.create<arith::ConstantIndexOp>(loc, upperBound);

    auto loop = rewriter.create<scf::ForOp>(loc, zero, upperBoundValue,
                                            tileSizeValue, target);

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(loop.getBody());
    offsets[dim] = loop.getInductionVar();
    sizes[dim] = rewriter.getIndexAttr(sliceSize);

    Value srcSubview =
        getSubView(rewriter, loc, src, shape, offsets, sizes, strides);
    Value targetSubview = getSubView(rewriter, loc, loop.getRegionIterArgs()[0],
                                     shape, offsets, sizes, strides);

    offsets[dim] = rewriter.getIndexAttr(0);

    createLoopsNest(rewriter, loc, dim + 1, srcSubview, targetSubview, shape,
                    offsets, sizes, strides);

    rewriter.create<scf::YieldOp>(loc, loop.getRegionIterArgs()[0]);

    // Remainder copy can only be created for the innermost loop, for other
    // loops remainder size is guaranteed to be 0.
    if (remainderSize > 0) {
      rewriter.setInsertionPointAfter(loop);

      offsets[dim] = rewriter.getIndexAttr(upperBound);
      sizes[dim] = rewriter.getIndexAttr(remainderSize);

      Value srcRemainderSubview =
          getSubView(rewriter, loc, src, shape, offsets, sizes, strides);
      Value targetRemainderSubview =
          getSubView(rewriter, loc, target, shape, offsets, sizes, strides);

      rewriter.create<memref::CopyOp>(loc, srcRemainderSubview,
                                      targetRemainderSubview);
    }
  }

  memref::SubViewOp getSubView(PatternRewriter &rewriter, Location loc,
                               Value val, ArrayRef<int64_t> shape,
                               SmallVector<OpFoldResult> &offsets,
                               SmallVector<OpFoldResult> &sizes,
                               SmallVector<OpFoldResult> &strides) const {
    auto valType = cast<MemRefType>(val.getType());

    auto valSubviewType =
        cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            shape, valType, offsets, sizes, strides));

    return rewriter.create<memref::SubViewOp>(loc, valSubviewType, val, offsets,
                                              sizes, strides);
  }

  int64_t tileSize;
};

/// Custom vectorization pattern for small and non-contiguous memref::CopyOp.
struct CopyVectorizationPattern : public OpRewritePattern<memref::CopyOp> {
  CopyVectorizationPattern(MLIRContext *context, int64_t numElementsThreshold,
                           mlir::PatternBenefit benefit = 1)
      : OpRewritePattern<memref::CopyOp>(context, benefit),
        numElementsThreshold(numElementsThreshold) {}

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<MemRefType>(op.getSource().getType());
    auto targetType = dyn_cast<MemRefType>(op.getTarget().getType());

    if (!srcType || !targetType) return failure();

    if (!srcType.hasStaticShape() || !targetType.hasStaticShape())
      return failure();

    // If memref has an identity layout or is contiguous with an arbitrary
    // offset, it will be turned into llvm.memcpy intrinsic later, do not
    // vectorize it.
    if (memref::isStaticShapeAndContiguousRowMajor(srcType) &&
        memref::isStaticShapeAndContiguousRowMajor(targetType)) {
      return failure();
    }

    auto isSmallMemrefType = [&](MemRefType memrefType) {
      return memrefType.getNumElements() > 0 &&
             memrefType.getNumElements() <= numElementsThreshold;
    };

    // If memref is too big, vectorizing it actually explodes the compilation
    // time. Also, ignore empty memrefs, which will be handled by memrefCopy
    // function.
    if (!isSmallMemrefType(srcType) || !isSmallMemrefType(targetType)) {
      return failure();
    }
    return linalg::vectorizeCopy(rewriter, op);
  }

 private:
  int64_t numElementsThreshold;
};

struct VectorizeCopyPass
    : public impl::VectorizeCopyPassBase<VectorizeCopyPass> {
  using Base::Base;

  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<TileCopyPattern, CopyVectorizationPattern>(
        ctx, /*numElementsThreshold = */ 8);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeCopyPass() {
  return std::make_unique<VectorizeCopyPass>();
}

}  // namespace mlir
