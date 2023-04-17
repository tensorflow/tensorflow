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

#include <memory>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Affine/ViewLikeInterfaceUtils.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

using tensor::ExtractOp;
using tensor::ExtractSliceOp;

#define GEN_PASS_DEF_COMPOSEEXTRACTINSERTSLICEPASS
#include "gml_st/transforms/passes.h.inc"

LogicalResult composeExtractOfExtractSlice(ExtractOp extractOp,
                                           PatternRewriter& rewriter) {
  auto sliceOp = extractOp.getTensor().getDefiningOp<ExtractSliceOp>();
  if (!sliceOp) return failure();

  Location loc = extractOp.getLoc();
  SmallVector<OpFoldResult> combinedOffsets, combinedSizes, combinedStrides;

  // ExtractOp can be viewed as ExtractSliceOp as extracts 1x...x1 slice.
  int64_t rank = extractOp.getTensor().getType().getRank();
  SmallVector<OpFoldResult> consumerOffsets(
      getAsOpFoldResult(extractOp.getIndices()));
  SmallVector<OpFoldResult> consumerSizes(rank, rewriter.getIndexAttr(1));
  SmallVector<OpFoldResult> consumerStrides(rank, rewriter.getIndexAttr(1));

  if (failed(mergeOffsetsSizesAndStrides(
          rewriter, loc, sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
          sliceOp.getMixedStrides(), sliceOp.getDroppedDims(), consumerOffsets,
          consumerSizes, consumerStrides, combinedOffsets, combinedSizes,
          combinedStrides)))
    return failure();

  rewriter.replaceOpWithNewOp<ExtractOp>(
      extractOp, sliceOp.getSource(),
      getValueOrCreateConstantIndexOp(rewriter, loc, combinedOffsets));
  return success();
}

struct ComposeExtractInsertSlicePass
    : public impl::ComposeExtractInsertSlicePassBase<
          ComposeExtractInsertSlicePass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add(composeExtractOfExtractSlice);
    tensor::populateMergeConsecutiveInsertExtractSlicePatterns(patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createComposeExtractInsertSlicePass() {
  return std::make_unique<ComposeExtractInsertSlicePass>();
}

}  // namespace mlir::gml_st
