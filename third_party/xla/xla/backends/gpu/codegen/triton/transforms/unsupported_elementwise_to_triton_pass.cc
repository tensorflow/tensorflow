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

#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
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

class RewriteIsFinite : public OpRewritePattern<mlir::math::IsFiniteOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::math::IsFiniteOp op,
                                PatternRewriter& rewriter) const override {
    mlir::Value operand = op.getOperand();
    mlir::Type operand_type = operand.getType();
    mlir::Type element_type = getElementTypeOrSelf(operand_type);
    auto float_type = mlir::dyn_cast<mlir::FloatType>(element_type);
    if (!float_type || !(float_type.isF16() || float_type.isBF16() ||
                         float_type.isF32() || float_type.isF64())) {
      return rewriter.notifyMatchFailure(
          op, "expected f16, bf16, f32, or f64 type");
    }

    const llvm::fltSemantics& semantics = float_type.getFloatSemantics();
    mlir::Value pos_inf = mlir::createScalarOrSplatConstant(
        rewriter, op->getLoc(), operand_type, mlir::APFloat::getInf(semantics));
    mlir::Value abs_operand =
        mlir::math::AbsFOp::create(rewriter, op->getLoc(), operand);
    rewriter.replaceOpWithNewOp<mlir::arith::CmpFOp>(
        op, mlir::arith::CmpFPredicate::ONE, abs_operand, pos_inf);
    return success();
  }
};

struct UnsupportedElementwiseToTritonPass
    : public impl::UnsupportedElementwiseToTritonPassBase<
          UnsupportedElementwiseToTritonPass> {
  void runOnOperation() override {
    auto module = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteNegFToSubtract, RewriteIsFinite>(&getContext());
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

}  // namespace mlir::triton::xla
