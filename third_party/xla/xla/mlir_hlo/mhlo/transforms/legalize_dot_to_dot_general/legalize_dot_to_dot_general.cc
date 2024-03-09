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

#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZEDOTTODOTGENERALPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct DotToDotGeneralPattern : public OpRewritePattern<DotOp> {
  using OpRewritePattern<DotOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotOp dotOp,
                                PatternRewriter &rewriter) const override {
    auto lhs = dotOp.getLhs();
    auto rhs = dotOp.getRhs();

    if (!lhs.getType().hasRank() || !rhs.getType().hasRank()) {
      return rewriter.notifyMatchFailure(dotOp, "unranked operands");
    }

    auto dotDimensionNumbers = DotDimensionNumbersAttr::get(
        dotOp.getContext(),
        /*lhsBatchingDimensions=*/{},
        /*rhsBatchingDimensions=*/{},
        /*lhsContractingDimensions=*/{lhs.getType().getRank() - 1},
        /*rhsContractingDimensions=*/{0});

    rewriter.replaceOpWithNewOp<DotGeneralOp>(dotOp, dotOp.getType(), lhs, rhs,
                                              dotDimensionNumbers,
                                              dotOp.getPrecisionConfigAttr());
    return success();
  }
};

struct LegalizeDotToDotGeneralPass
    : public impl::LegalizeDotToDotGeneralPassBase<
          LegalizeDotToDotGeneralPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateDotToDotGeneralPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateDotToDotGeneralPatterns(mlir::MLIRContext *context,
                                     RewritePatternSet *patterns) {
  patterns->add<DotToDotGeneralPattern>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeDotToDotGeneralPass() {
  return std::make_unique<LegalizeDotToDotGeneralPass>();
}

}  // namespace mhlo
}  // namespace mlir
