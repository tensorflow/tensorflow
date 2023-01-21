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

#include <iterator>
#include <memory>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/rewriters.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_COLLAPSEMATERIALIZEOPSPASS
#include "gml_st/transforms/passes.h.inc"

// Collapse extract_slice operations
//   `extract_slice(extract_slice(tensor1, slice_params1), slice_params)
// ... as ...
//   `extract_slice(tensor1, composed_slice_params)
struct CollapseExtractSliceOpPattern
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern<tensor::ExtractSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp op,
                                PatternRewriter& rewriter) const override {
    auto producerExtractSliceOp =
        op.getSource().getDefiningOp<tensor::ExtractSliceOp>();
    if (!producerExtractSliceOp) return failure();

    // Compose tileOp and producerTileOp.
    auto loc = op.getLoc();
    SmallVector<OpFoldResult> newOffsets, newSizes, newStrides;
    if (failed(mergeOffsetsSizesAndStrides(
            rewriter, loc, producerExtractSliceOp, op,
            producerExtractSliceOp.getDroppedDims(), newOffsets, newSizes,
            newStrides)))
      return failure();

    rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
        op, op.getType(), producerExtractSliceOp.getSource(), newOffsets,
        newSizes, newStrides);
    return success();
  }
};

struct CollapseMaterializeOpsPass
    : public impl::CollapseMaterializeOpsPassBase<CollapseMaterializeOpsPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateCollapseMaterializeOpsPatterns(ctx, &patterns);

    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateCollapseMaterializeOpsPatterns(MLIRContext* ctx,
                                            RewritePatternSet* patterns) {
  patterns->add<CollapseExtractSliceOpPattern>(ctx);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createCollapseMaterializeOpsPass() {
  return std::make_unique<CollapseMaterializeOpsPass>();
}

}  // namespace gml_st
}  // namespace mlir
