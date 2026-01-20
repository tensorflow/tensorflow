/* Copyright 2025 The OpenXLA Authors.

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

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/triton/transforms/passes.h"

namespace mlir::triton::xla {

#define GEN_PASS_DEF_UNSUPPORTEDELEMENTWISETOTRITONPASS
#include "xla/backends/gpu/codegen/triton/transforms/passes.h.inc"

namespace {

class RewriteNegFToSubtract : public OpRewritePattern<mlir::arith::NegFOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::arith::NegFOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Type element_type = getElementTypeOrSelf(op.getType());
    auto type = mlir::dyn_cast<mlir::FloatType>(element_type);

    if (!type) {
      return rewriter.notifyMatchFailure(op, "expected float type");
    }

    const llvm::fltSemantics& semantics = type.getFloatSemantics();
    mlir::Value zero_value =
        mlir::createScalarOrSplatConstant(rewriter, op->getLoc(), op.getType(),
                                          mlir::APFloat::getZero(semantics));

    rewriter.replaceOpWithNewOp<mlir::arith::SubFOp>(op, zero_value,
                                                     op.getOperand());
    return success();
  }
};

struct UnsupportedElementwiseToTritonPass
    : public impl::UnsupportedElementwiseToTritonPassBase<
          UnsupportedElementwiseToTritonPass> {
  void runOnOperation() override {
    auto module = getOperation();
    mlir::RewritePatternSet patterns(
        &getContext(), std::make_unique<RewriteNegFToSubtract>(&getContext()));
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateUnsupportedElementwiseToTritonPass() {
  return std::make_unique<UnsupportedElementwiseToTritonPass>();
}

}  // namespace mlir::triton::xla
