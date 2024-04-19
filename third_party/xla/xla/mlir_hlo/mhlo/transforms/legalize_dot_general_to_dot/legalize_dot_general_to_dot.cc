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

// This file implements logic for simplifying HLO dot.

#include <memory>
#include <utility>

#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_LEGALIZEDOTGENERALTODOTPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

constexpr char kFrontendAttributesAttr[] = "mhlo.frontend_attributes";

// Handle the generic case of DotGeneral and convert to a regulat DotOp.
struct DotGeneralToDot : public OpRewritePattern<DotGeneralOp> {
  using OpRewritePattern<DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(DotGeneralOp dot,
                                PatternRewriter& rewriter) const override {
    auto lhs = dot.getLhs();
    auto rhs = dot.getRhs();
    auto lhsTy = lhs.getType().cast<ShapedType>();
    auto rhsTy = rhs.getType().cast<ShapedType>();

    int64_t lhsRank = lhsTy.getRank();
    int64_t rhsRank = rhsTy.getRank();
    if ((lhsRank != 1 && lhsRank != 2) || (rhsRank != 1 && rhsRank != 2)) {
      return rewriter.notifyMatchFailure(
          dot, "input tensors must have rank of 1 or 2");
    }

    auto nums = dot.getDotDimensionNumbers();
    if ((!nums.getLhsBatchingDimensions().empty()) ||
        (!nums.getRhsBatchingDimensions().empty())) {
      return rewriter.notifyMatchFailure(dot, "cannot have batch dimensions");
    }

    auto lhsContract = nums.getLhsContractingDimensions();
    auto rhsContract = nums.getRhsContractingDimensions();

    if (lhsContract.size() != 1 || rhsContract.size() != 1) {
      return rewriter.notifyMatchFailure(
          dot, "input tensors must only have 1 contracting dimension");
    }
    if (rhsContract.front() != 0) {
      return rewriter.notifyMatchFailure(
          dot, "rhs must contract the first dimension");
    }
    if (lhsContract.front() != lhsRank - 1) {
      return rewriter.notifyMatchFailure(
          dot, "lhs must contract the last dimension");
    }

    DictionaryAttr frontendAttributes =
        dot->getAttrOfType<DictionaryAttr>(kFrontendAttributesAttr);
    auto newDotOp = rewriter.replaceOpWithNewOp<mhlo::DotOp>(
        dot, dot.getType(), lhs, rhs,
        dot.getPrecisionConfig().value_or(nullptr));
    if (frontendAttributes) {
      newDotOp->setAttr(kFrontendAttributesAttr, frontendAttributes);
    }

    return success();
  }
};

struct LegalizeDotGeneralToDotPass
    : impl::LegalizeDotGeneralToDotPassBase<LegalizeDotGeneralToDotPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<DotGeneralToDot>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeDotGeneralToDotPass() {
  return std::make_unique<LegalizeDotGeneralToDotPass>();
}

}  // namespace mhlo
}  // namespace mlir
