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
#include <memory>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace xla::cpu {

#define GEN_PASS_DECL_EXPANDFLOATOPSPASS
#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;

// Get a constant value, if the type is a vector, splat the value to the vector
// type.
mlir::Value GetConst(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                     mlir::TypedAttr value) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(type)) {
    value =
        mlir::SplatElementsAttr::get(mlir::cast<mlir::ShapedType>(type), value);
  }
  return mlir::arith::ConstantOp::create(b, type, value);
}

mlir::Value EmitBF16ToF32(mlir::Type dst_ty, mlir::Value in,
                          mlir::ImplicitLocOpBuilder& b) {
  auto get_type = [&](mlir::Type element_type) -> mlir::Type {
    if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(in.getType())) {
      return vector_type.clone(element_type);
    }
    return element_type;
  };

  mlir::Type i16_type = get_type(b.getI16Type());
  mlir::Type i32_type = get_type(b.getI32Type());

  mlir::Value i16 = ma::BitcastOp::create(b, i16_type, in);
  mlir::Value i32 = ma::ExtUIOp::create(b, i32_type, i16);

  mlir::TypedAttr shift_value = b.getI32IntegerAttr(16);
  mlir::Value shift_const = GetConst(b, i32_type, shift_value);

  mlir::Value i32_shl = mlir::arith::ShLIOp::create(b, i32, shift_const);
  return ma::BitcastOp::create(b, dst_ty, i32_shl);
}

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = op.getType();

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (mlir::isa<mlir::BFloat16Type>(
            mlir::getElementTypeOrSelf(src.getType())) &&
        mlir::isa<mlir::Float32Type>(mlir::getElementTypeOrSelf(dst_ty))) {
      rewriter.replaceOp(op, EmitBF16ToF32(dst_ty, src, builder));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Not bf16 -> f32");
  }
};

class RewriteCbrtPattern : public mlir::OpRewritePattern<mlir::math::CbrtOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::CbrtOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    mlir::arith::FastMathFlagsAttr fastmath = op.getFastmathAttr();

    mlir::Value input_abs =
        b.create<mlir::math::AbsFOp>(op.getOperand(), fastmath).getResult();

    mlir::TypedAttr third_attr =
        b.getFloatAttr(mlir::getElementTypeOrSelf(op.getType()), 1.0 / 3.0);
    mlir::Value third_value = GetConst(b, op.getType(), third_attr);
    mlir::Value cbrt_abs =
        b.create<mlir::math::PowFOp>(input_abs, third_value, fastmath);

    mlir::Value cbrt_signed =
        b.create<mlir::math::CopySignOp>(cbrt_abs, op.getOperand(), fastmath)
            .getResult();

    rewriter.replaceOp(op, cbrt_signed);
    return mlir::success();
  }
};

// Use a more numerically stable implementation of expm1(x).
// |x| > 0.5: exp(x) - 1
// |x| < 0.5: tanh(x/2) * (exp(x)+1)
class RewriteExpm1Pattern : public mlir::OpRewritePattern<mlir::math::ExpM1Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpM1Op op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    mlir::Type type = op.getType();
    mlir::Type element_type = mlir::getElementTypeOrSelf(type);
    mlir::Value one = GetConst(b, type, b.getFloatAttr(element_type, 1.0));
    mlir::Value half = GetConst(b, type, b.getFloatAttr(element_type, 0.5));
    mlir::Value zero = GetConst(b, type, b.getFloatAttr(element_type, 0.0));
    mlir::Value x = op.getOperand();

    mlir::arith::FastMathFlagsAttr fastmath = op.getFastmathAttr();

    mlir::Value exp_x = b.create<mlir::math::ExpOp>(x, fastmath);

    mlir::Value exp_x_minus_1 =
        b.create<mlir::arith::SubFOp>(exp_x, one, fastmath);

    mlir::Value half_x = b.create<mlir::arith::MulFOp>(x, half, fastmath);
    mlir::Value tanh_half_x = b.create<mlir::math::TanhOp>(half_x, fastmath);
    mlir::Value exp_x_plus_1 =
        b.create<mlir::arith::AddFOp>(exp_x, one, fastmath);
    mlir::Value small_result =
        b.create<mlir::arith::MulFOp>(tanh_half_x, exp_x_plus_1, fastmath);

    mlir::Value abs_x = b.create<mlir::math::AbsFOp>(x, fastmath);
    mlir::Value x_is_large = b.create<mlir::arith::CmpFOp>(
        mlir::arith::CmpFPredicate::OGT, abs_x, half);
    mlir::Value normal_result = b.create<mlir::arith::SelectOp>(
        x_is_large, exp_x_minus_1, small_result);

    // half_x can underflow resulting in zero.
    // TODO(willfroom): Do we actually need this check? tanh(0) == 0.
    mlir::Value half_x_is_zero = b.create<mlir::arith::CmpFOp>(
        mlir::arith::CmpFPredicate::OEQ, half_x, zero);
    mlir::Value result =
        b.create<mlir::arith::SelectOp>(half_x_is_zero, x, normal_result);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteExtFPattern, RewriteCbrtPattern, RewriteExpm1Pattern>(
        &getContext());

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass() {
  return std::make_unique<ExpandFloatOpsPass>();
}

}  // namespace xla::cpu
