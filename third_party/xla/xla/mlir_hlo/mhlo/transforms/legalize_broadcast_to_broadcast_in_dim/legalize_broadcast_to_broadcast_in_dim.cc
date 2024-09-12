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

#include <cstdint>
#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEBROADCASTTOBROADCASTINDIMPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct BroadcastToBroadcastInDimPattern : public OpRewritePattern<BroadcastOp> {
  using OpRewritePattern<BroadcastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BroadcastOp broadcastOp,
                                PatternRewriter &rewriter) const override {
    auto resultType = broadcastOp.getType();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(
          broadcastOp, "cannot convert broadcast with dynamic result type");
    }

    auto operandRank =
        resultType.getRank() - broadcastOp.getBroadcastSizes().size();
    SmallVector<int64_t> broadcastDimensionsArgs;
    for (auto i = 0; i < operandRank; ++i) {
      broadcastDimensionsArgs.push_back(i +
                                        broadcastOp.getBroadcastSizes().size());
    }

    auto broadcastInDimOp = rewriter.create<BroadcastInDimOp>(
        broadcastOp.getLoc(), resultType, broadcastOp.getOperand(),
        rewriter.getI64TensorAttr(broadcastDimensionsArgs));
    rewriter.replaceOp(broadcastOp, broadcastInDimOp);
    return success();
  }
};

struct LegalizeBroadcastToBroadcastInDimPass
    : public impl::LegalizeBroadcastToBroadcastInDimPassBase<
          LegalizeBroadcastToBroadcastInDimPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateBroadcastToBroadcastInDimPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateBroadcastToBroadcastInDimPatterns(mlir::MLIRContext *context,
                                               RewritePatternSet *patterns) {
  patterns->add<BroadcastToBroadcastInDimPattern>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeBroadcastToBroadcastInDimPass() {
  return std::make_unique<LegalizeBroadcastToBroadcastInDimPass>();
}

}  // namespace mhlo
}  // namespace mlir
