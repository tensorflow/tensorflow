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

#include <cassert>
#include <cstdint>
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"

namespace xla::emitters {

#define GEN_PASS_DEF_SAFEINTEGERARITHMETICPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

inline mlir::Value GetConstantOrSplat(mlir::ImplicitLocOpBuilder& builder,
                                      mlir::Type type, mlir::TypedAttr value) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(type)) {
    value = mlir::SplatElementsAttr::get(vector_type, value);
  }
  return mlir::arith::ConstantOp::create(builder, type, value);
}

inline mlir::Value GetConstantOrSplat(mlir::ImplicitLocOpBuilder& builder,
                                      mlir::Type type, mlir::APInt value) {
  return GetConstantOrSplat(
      builder, type,
      builder.getIntegerAttr(mlir::getElementTypeOrSelf(type), value));
}

template <typename OpT>
inline mlir::Value MakeSafeSignedIntOp(mlir::ImplicitLocOpBuilder& builder,
                                       mlir::Value lhs, mlir::Value rhs,
                                       mlir::Value on_zero,
                                       mlir::Value on_overflow) {
  mlir::Type type = lhs.getType();
  auto element_type =
      mlir::cast<mlir::IntegerType>(mlir::getElementTypeOrSelf(type));
  mlir::Value zero =
      builder.create<mlir::arith::ConstantOp>(builder.getZeroAttr(type));
  mlir::Value one =
      GetConstantOrSplat(builder, type, builder.getOneAttr(element_type));
  mlir::Value rhs_is_zero = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, rhs, zero);

  // For signed also check for INT_MIN / -1.
  mlir::Value smin = GetConstantOrSplat(
      builder, type, mlir::APInt::getSignedMinValue(element_type.getWidth()));
  mlir::Value lhs_is_smin = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, lhs, smin);
  mlir::Value minus_one = GetConstantOrSplat(
      builder, type, mlir::APInt::getAllOnes(element_type.getWidth()));
  mlir::Value rhs_is_minus_one = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, rhs, minus_one);
  mlir::Value has_int_min_overflow =
      builder.create<mlir::arith::AndIOp>(lhs_is_smin, rhs_is_minus_one);
  mlir::Value rhs_is_unsafe =
      builder.create<mlir::arith::OrIOp>(rhs_is_zero, has_int_min_overflow);
  mlir::Value safe_rhs =
      builder.create<mlir::arith::SelectOp>(rhs_is_unsafe, one, rhs);
  mlir::Value safe_div = builder.create<OpT>(lhs, safe_rhs);
  mlir::Value safe_smin = builder.create<mlir::arith::SelectOp>(
      has_int_min_overflow, on_overflow, safe_div);
  return builder.create<mlir::arith::SelectOp>(rhs_is_zero, on_zero, safe_smin);
}

template <typename OpT>
inline mlir::Value MakeSafeUnsignedIntOp(mlir::ImplicitLocOpBuilder& builder,
                                         mlir::Value lhs, mlir::Value rhs,
                                         mlir::Value on_zero) {
  mlir::Type type = lhs.getType();
  auto element_type =
      mlir::cast<mlir::IntegerType>(mlir::getElementTypeOrSelf(type));
  mlir::Value zero =
      builder.create<mlir::arith::ConstantOp>(builder.getZeroAttr(type));
  auto make_constant = [&](const mlir::APInt& i) {
    return GetConstantOrSplat(builder, type,
                              builder.getIntegerAttr(element_type, i));
  };
  mlir::Value one = make_constant(mlir::APInt(element_type.getWidth(), 1));
  mlir::Value rhs_is_zero = builder.create<mlir::arith::CmpIOp>(
      mlir::arith::CmpIPredicate::eq, rhs, zero);

  // For unsigned just set the divisor to 1 when it would be 0.
  mlir::Value safe_rhs =
      builder.create<mlir::arith::SelectOp>(rhs_is_zero, one, rhs);
  mlir::Value safe_div = builder.create<OpT>(lhs, safe_rhs);
  return builder.create<mlir::arith::SelectOp>(rhs_is_zero, on_zero, safe_div);
}

// Integer division overflow behavior:
//
// x / 0 == -1
// INT_SMIN / -1 = INT_SMIN
struct MakeSignedDivSafe : mlir::OpRewritePattern<mlir::arith::DivSIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::DivSIOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op->hasAttr("xla.guard_ub")) {
      return rewriter.notifyMatchFailure(op, "already safe");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    mlir::Type type = op.getType();
    auto element_type =
        mlir::cast<mlir::IntegerType>(mlir::getElementTypeOrSelf(type));
    int64_t width = element_type.getWidth();
    auto minus_one =
        GetConstantOrSplat(builder, type, mlir::APInt::getAllOnes(width));
    auto smin = GetConstantOrSplat(builder, type,
                                   mlir::APInt::getSignedMinValue(width));

    rewriter.replaceOp(op,
                       MakeSafeSignedIntOp<mlir::arith::DivSIOp>(
                           builder, op.getLhs(), op.getRhs(), minus_one, smin));

    return mlir::success();
  }
};

// Unsigned integer division overflow behavior:
//
// x / 0 == -1
struct MakeUnsignedDivSafe : mlir::OpRewritePattern<mlir::arith::DivUIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::DivUIOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op->hasAttr("xla.guard_ub")) {
      return rewriter.notifyMatchFailure(op, "already safe");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    mlir::Type type = op.getType();
    auto element_type =
        mlir::cast<mlir::IntegerType>(mlir::getElementTypeOrSelf(type));
    int64_t width = element_type.getWidth();
    auto minus_one =
        GetConstantOrSplat(builder, type, mlir::APInt::getAllOnes(width));

    rewriter.replaceOp(op, MakeSafeUnsignedIntOp<mlir::arith::DivUIOp>(
                               builder, op.getLhs(), op.getRhs(), minus_one));

    return mlir::success();
  }
};

// Integer remainder overflow behavior:
//
// x % 0 == x
// INT_SMIN % -1 = 0
struct MakeSignedRemSafe : mlir::OpRewritePattern<mlir::arith::RemSIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::RemSIOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op->hasAttr("xla.guard_ub")) {
      return rewriter.notifyMatchFailure(op, "already safe");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    mlir::Type type = op.getType();
    auto element_type =
        mlir::cast<mlir::IntegerType>(mlir::getElementTypeOrSelf(type));
    int64_t width = element_type.getWidth();
    auto zero = GetConstantOrSplat(builder, type, mlir::APInt::getZero(width));

    rewriter.replaceOp(
        op, MakeSafeSignedIntOp<mlir::arith::RemSIOp>(
                builder, op.getLhs(), op.getRhs(), op.getLhs(), zero));

    return mlir::success();
  }
};

// Unsigned integer remainder overflow behavior:
//
// x % 0 == x
struct MakeUnsignedRemSafe : mlir::OpRewritePattern<mlir::arith::RemUIOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::RemUIOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op->hasAttr("xla.guard_ub")) {
      return rewriter.notifyMatchFailure(op, "already safe");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    rewriter.replaceOp(op, MakeSafeUnsignedIntOp<mlir::arith::RemUIOp>(
                               builder, op.getLhs(), op.getRhs(), op.getLhs()));

    return mlir::success();
  }
};

class SafeIntegerArithmeticPass
    : public impl::SafeIntegerArithmeticPassBase<SafeIntegerArithmeticPass> {
 public:
  using SafeIntegerArithmeticPassBase::SafeIntegerArithmeticPassBase;

  void runOnOperation() override {
    mlir::MLIRContext* context = &getContext();
    mlir::RewritePatternSet patterns(context);
    patterns.add<MakeSignedDivSafe, MakeUnsignedDivSafe, MakeSignedRemSafe,
                 MakeUnsignedRemSafe>(context);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateSafeIntegerArithmeticPass() {
  return std::make_unique<SafeIntegerArithmeticPass>();
}

}  // namespace xla::emitters
