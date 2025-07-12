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
#include "xla/codegen/math/fptrunc.h"
#include "xla/codegen/math/log1p.h"
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

class RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
 public:
  RewriteTruncFPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    if (!mlir::isa<mlir::Float32Type>(src.getType()) ||
        !mlir::isa<mlir::BFloat16Type>(dst_ty)) {
      return rewriter.notifyMatchFailure(op, "Not f32 -> bf16");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto f32_to_bf16_decl = GetF32ToBF16Declaration(rewriter);
    auto call_op =
        b.create<mlir::func::CallOp>(f32_to_bf16_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::func::FuncOp GetF32ToBF16Declaration(
      mlir::PatternRewriter& rewriter) const {
    mlir::Type f32_type = rewriter.getF32Type();
    mlir::Type bf16_type = rewriter.getBF16Type();
    return GetOrInsertDeclaration(
        rewriter, module_op_,
        codegen::Intrinsic::FpTrunc::Name(codegen::Intrinsic::S(F32),
                                          codegen::Intrinsic::S(BF16)),
        rewriter.getFunctionType(f32_type, bf16_type));
  }

 private:
  mlir::ModuleOp& module_op_;
};

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

class RewriteErf64Pattern : public mlir::OpRewritePattern<mlir::math::ErfOp> {
 public:
  RewriteErf64Pattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ErfOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Type type = op.getType();

    if (!type.isF64()) {
      return rewriter.notifyMatchFailure(op, "not an 64 erf");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto erf_decl = GetErf64Declaration(rewriter);
    auto call_op = b.create<mlir::func::CallOp>(erf_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::func::FuncOp GetErf64Declaration(
      mlir::PatternRewriter& rewriter) const {
    mlir::Type f64_type = rewriter.getF64Type();
    return GetOrInsertDeclaration(rewriter, module_op_, "erf",
                                  rewriter.getFunctionType(f64_type, f64_type));
  }

 private:
  mlir::ModuleOp& module_op_;
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

class RewriteLog1pPattern : public mlir::OpRewritePattern<mlir::math::Log1pOp> {
 public:
  RewriteLog1pPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::Log1pOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Type type = op.getType();
    PrimitiveType primitive_type = ConvertMlirTypeToPrimitiveType(type);

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto log1p_decl = GetOrInsertDeclaration(
        rewriter, module_op_,
        codegen::math::Log1pFunctionName(1, primitive_type),
        rewriter.getFunctionType(type, type));
    auto call_op = b.create<mlir::func::CallOp>(log1p_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    mlir::ModuleOp module_op = getOperation();
    patterns.add<RewriteExtFPattern, RewriteCbrtPattern>(&getContext());
    patterns
        .add<RewriteTruncFPattern, RewriteErf64Pattern, RewriteLog1pPattern>(
            &getContext(), module_op);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateExpandFloatOpsPass() {
  return std::make_unique<ExpandFloatOpsPass>();
}

}  // namespace xla::cpu
