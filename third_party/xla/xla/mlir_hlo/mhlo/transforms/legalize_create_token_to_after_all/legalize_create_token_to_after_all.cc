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
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZECREATETOKENTOAFTERALLPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

struct CreateTokenToAfterAllPattern : public OpRewritePattern<CreateTokenOp> {
  using OpRewritePattern<CreateTokenOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateTokenOp createTokenOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<AfterAllOp>(
        createTokenOp, createTokenOp.getType(), ValueRange{});
    return success();
  }
};

struct LegalizeCreateTokenToAfterAllPass
    : public impl::LegalizeCreateTokenToAfterAllPassBase<
          LegalizeCreateTokenToAfterAllPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateCreateTokenToAfterAllPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

void populateCreateTokenToAfterAllPatterns(mlir::MLIRContext *context,
                                           RewritePatternSet *patterns) {
  patterns->add<CreateTokenToAfterAllPattern>(context);
}

std::unique_ptr<OperationPass<func::FuncOp>>
createLegalizeCreateTokenToAfterAllPass() {
  return std::make_unique<LegalizeCreateTokenToAfterAllPass>();
}

}  // namespace mhlo
}  // namespace mlir
