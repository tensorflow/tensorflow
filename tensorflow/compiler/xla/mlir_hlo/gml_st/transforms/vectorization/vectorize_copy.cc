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

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/vectorization/vectorization.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_VECTORIZECOPYPASS
#include "gml_st/transforms/passes.h.inc"

/// Custom vectorization pattern for small and non-contiguous memref::CopyOp.
struct CopyVectorizationPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto srcType = op.getSource().getType().cast<BaseMemRefType>();
    auto targetType = op.getTarget().getType().cast<BaseMemRefType>();

    auto isStaticShapeAndContiguousRowMajor = [](MemRefType type) {
      if (!type.hasStaticShape()) return false;

      SmallVector<int64_t> strides;
      int64_t offset;
      if (failed(getStridesAndOffset(type, strides, offset))) return false;

      int64_t runningStride = 1;
      for (unsigned i = strides.size(); i > 0; --i) {
        if (strides[i - 1] != runningStride) return false;
        runningStride *= type.getDimSize(i - 1);
      }
      return true;
    };

    auto isContiguousMemrefType = [&](BaseMemRefType type) {
      auto memrefType = type.dyn_cast<mlir::MemRefType>();
      return memrefType && (memrefType.getLayout().isIdentity() ||
                            isStaticShapeAndContiguousRowMajor(memrefType));
    };

    auto isSmallMemrefType = [&](BaseMemRefType type) {
      auto memrefType = type.dyn_cast<mlir::MemRefType>();
      return memrefType && memrefType.hasStaticShape() &&
             memrefType.getNumElements() > 0 &&
             memrefType.getNumElements() < kNumElementsThreshold;
    };

    // If memref has an identity layout or is contiguous with an arbitrary
    // offset, it will be turned into llvm.memcpy intrinsic later, do not
    // vectorize it.
    if (isContiguousMemrefType(srcType) && isContiguousMemrefType(targetType)) {
      return failure();
    }

    // If memref is too big, vectorizing it actually explodes the compilation
    // time. Also, ignore empty memrefs, which will be handled by memrefCopy
    // function.
    if (!isSmallMemrefType(srcType) || !isSmallMemrefType(targetType)) {
      return failure();
    }
    return linalg::vectorizeCopy(rewriter, op);
  }
};

struct VectorizeCopyPass
    : public impl::VectorizeCopyPassBase<VectorizeCopyPass> {
  void runOnOperation() override {
    auto func = getOperation();
    auto *ctx = func.getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<CopyVectorizationPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeCopyPass() {
  return std::make_unique<VectorizeCopyPass>();
}

}  // namespace gml_st
}  // namespace mlir
