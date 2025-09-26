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

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/implicit_arith_op_builder.h"
#include "xla/mlir/utils/type_util.h"

namespace xla::cpu {

#define GEN_PASS_DECL_EXPANDFLOATOPSPASS
#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;

mlir::func::FuncOp GetOrInsertDeclaration(mlir::PatternRewriter& rewriter,
                                          mlir::ModuleOp& module_op,
                                          absl::string_view name,
                                          mlir::FunctionType func_type) {
  // Check if the function already exists
  if (auto func = module_op.lookupSymbol<mlir::func::FuncOp>(name)) {
    // Ensure the existing function has the correct type
    if (func.getFunctionType() == func_type) {
      return func;
    }
  }

  // If not found or type mismatch, create the declaration
  mlir::PatternRewriter::InsertionGuard insertGuard(rewriter);
  rewriter.setInsertionPointToStart(module_op.getBody());

  auto func_decl =
      rewriter.create<mlir::func::FuncOp>(module_op.getLoc(), name, func_type);
  func_decl.setPrivate();
  return func_decl;
}

mlir::Value EmitBF16ToF32(mlir::Value in, mlir::ImplicitLocOpBuilder& b) {
  mlir::Value i16 = b.create<ma::BitcastOp>(b.getI16Type(), in);
  emitters::ImplicitArithOpBuilder i32(
      b.create<ma::ExtUIOp>(b.getI32Type(), i16), &b);
  return b.create<ma::BitcastOp>(b.getType<mlir::Float32Type>(), i32 << 16);
}

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (mlir::isa<mlir::BFloat16Type>(src.getType()) &&
        mlir::isa<mlir::Float32Type>(dst_ty)) {
      rewriter.replaceOp(op, EmitBF16ToF32(src, builder));
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

    mlir::Value input_abs =
        b.create<mlir::math::AbsFOp>(op.getOperand(), op.getFastmathAttr())
            .getResult();

    mlir::Value one_third = b.create<mlir::arith::ConstantOp>(
        b.getFloatAttr(op.getType(), 1.0 / 3.0));
    mlir::Value cbrt_abs = b.create<mlir::math::PowFOp>(input_abs, one_third,
                                                        op.getFastmathAttr());

    mlir::Value cbrt_signed =
        b.create<mlir::math::CopySignOp>(cbrt_abs, op.getOperand(),
                                         op.getFastmathAttr())
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
    mlir::Value one =
        b.create<mlir::arith::ConstantOp>(b.getFloatAttr(type, 1.0));
    mlir::Value half =
        b.create<mlir::arith::ConstantOp>(b.getFloatAttr(type, 0.5));
    mlir::Value zero =
        b.create<mlir::arith::ConstantOp>(b.getFloatAttr(type, 0.0));
    mlir::Value x = op.getOperand();

    mlir::Value exp_x = b.create<mlir::math::ExpOp>(x, op.getFastmathAttr());

    mlir::Value exp_x_minus_1 =
        b.create<mlir::arith::SubFOp>(exp_x, one, op.getFastmathAttr());

    mlir::Value half_x =
        b.create<mlir::arith::MulFOp>(x, half, op.getFastmathAttr());
    mlir::Value tanh_half_x =
        b.create<mlir::math::TanhOp>(half_x, op.getFastmathAttr());
    mlir::Value exp_x_plus_1 =
        b.create<mlir::arith::AddFOp>(exp_x, one, op.getFastmathAttr());
    mlir::Value small_result = b.create<mlir::arith::MulFOp>(
        tanh_half_x, exp_x_plus_1, op.getFastmathAttr());

    mlir::Value abs_x = b.create<mlir::math::AbsFOp>(x, op.getFastmathAttr());
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
