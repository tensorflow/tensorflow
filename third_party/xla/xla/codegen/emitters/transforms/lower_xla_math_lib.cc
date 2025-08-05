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

#include "absl/strings/string_view.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/math/erf.h"
#include "xla/codegen/math/exp.h"
#include "xla/codegen/math/fptrunc.h"
#include "xla/codegen/math/intrinsic.h"
#include "xla/codegen/math/log1p.h"
#include "xla/codegen/math/rsqrt.h"
#include "xla/codegen/math/tanh.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERXLAMATHLIBPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

using Type = ::xla::codegen::intrinsics::Type;
// TODO(talts): Add LowerMathOpPattern based on MathFunction instances.

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

class LowerExpOpPattern : public mlir::OpRewritePattern<mlir::math::ExpOp> {
 public:
  LowerExpOpPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpOp op, mlir::PatternRewriter& rewriter) const override {
    // Only convert F64 (or vectorized F64) exp operations
    auto op_type = op.getOperand().getType();
    bool op_is_f64 = mlir::isa<mlir::FloatType>(op_type) &&
                     mlir::dyn_cast<mlir::FloatType>(op_type).isF64();
    bool op_is_f64_vector =
        mlir::isa<mlir::VectorType>(op_type) &&
        mlir::dyn_cast<mlir::VectorType>(op_type).getElementType().isF64();
    if (!(op_is_f64 || op_is_f64_vector)) {
      return rewriter.notifyMatchFailure(op, "not an f64 exp operation");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    mlir::func::FuncOp xla_exp_func =
        codegen::intrinsics::Exp::GetOrInsertDeclaration(
            rewriter, module_op_, Type::TypeFromIrType(op_type));

    // Replace math.exp with call to xla.exp.f64
    auto call_op =
        builder.create<mlir::func::CallOp>(xla_exp_func, op.getOperand());

    rewriter.replaceOp(op, call_op);
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class LowerLog1pPattern : public mlir::OpRewritePattern<mlir::math::Log1pOp> {
 public:
  LowerLog1pPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::Log1pOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Type type = op.getType();

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto log1p_decl = codegen::intrinsics::Log1p::GetOrInsertDeclaration(
        rewriter, module_op_, Type::TypeFromIrType(type));
    auto call_op = b.create<mlir::func::CallOp>(log1p_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class LowerErfPattern : public mlir::OpRewritePattern<mlir::math::ErfOp> {
 public:
  LowerErfPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ErfOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Type type = op.getType();

    // Extend the argument to f32 and truncate the result back unconditionally
    // as these will be cleaned up later if they are already f32.
    if (type.isF16() || type.isF32()) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      mlir::Type f32_type = b.getF32Type();

      mlir::Value input_value =
          b.create<mlir::arith::ExtFOp>(f32_type, op.getOperand());

      auto erf_decl = codegen::intrinsics::Erf::GetOrInsertDeclaration(
          rewriter, module_op_, Type::TypeFromIrType(f32_type));
      auto call_op = b.create<mlir::func::CallOp>(erf_decl, input_value);

      mlir::Value f32_result = call_op.getResult(0);
      mlir::Value result = b.create<mlir::arith::TruncFOp>(type, f32_result);

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (type.isF64()) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      auto erf_decl = GetErf64Declaration(rewriter);
      auto call_op = b.create<mlir::func::CallOp>(erf_decl, op.getOperand());
      rewriter.replaceOp(op, call_op->getResults());
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Argument is not f32 or f64.");
  }

 private:
  mlir::func::FuncOp GetErf64Declaration(
      mlir::PatternRewriter& rewriter) const {
    mlir::Type f64_type = rewriter.getF64Type();
    return GetOrInsertDeclaration(rewriter, module_op_, "erf",
                                  rewriter.getFunctionType(f64_type, f64_type));
  }

  mlir::ModuleOp& module_op_;
};

class LowerTruncF32BF16FPattern
    : public mlir::OpRewritePattern<mlir::arith::TruncFOp> {
 public:
  LowerTruncF32BF16FPattern(mlir::MLIRContext* context,
                            mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::arith::TruncFOp op,
      mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    if (!mlir::isa<mlir::Float32Type>(src.getType()) ||
        !mlir::isa<mlir::BFloat16Type>(dst_ty)) {
      return rewriter.notifyMatchFailure(op, "Not f32 -> bf16");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    Type src_type = Type::S(F32);
    Type dst_type = Type::S(BF16);
    auto f32_to_bf16_decl =
        codegen::intrinsics::FpTrunc::GetOrInsertDeclaration(
            rewriter, module_op_, src_type, dst_type);
    auto call_op =
        b.create<mlir::func::CallOp>(f32_to_bf16_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class RsqrtPattern : public mlir::OpRewritePattern<mlir::math::RsqrtOp> {
 public:
  RsqrtPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::RsqrtOp op, mlir::PatternRewriter& rewriter) const override {
    // Don't change if not f32 or f64:
    auto src_type = op.getOperand().getType();
    if (!mlir::isa<mlir::Float32Type>(src_type) &&
        !mlir::isa<mlir::Float64Type>(src_type)) {
      return rewriter.notifyMatchFailure(op, "Not f32 or f64");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto rsqrt_decl = codegen::intrinsics::Rsqrt::GetOrInsertDeclaration(
        rewriter, module_op_, Type::TypeFromIrType(op.getOperand().getType()));
    auto call_op = b.create<mlir::func::CallOp>(rsqrt_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class LowerTanhOpPattern : public mlir::OpRewritePattern<mlir::math::TanhOp> {
 public:
  LowerTanhOpPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::TanhOp op, mlir::PatternRewriter& rewriter) const override {
    mlir::Type type = op.getType();

    if (!type.isF32() || !type.isF16()) {
      return rewriter.notifyMatchFailure(op, "unsupported type");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto tanh_decl = codegen::intrinsics::Tanh::GetOrInsertDeclaration(
        rewriter, module_op_, Type::TypeFromIrType(type));
    auto call_op = b.create<mlir::func::CallOp>(tanh_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class LowerXlaMathLibPass
    : public impl::LowerXlaMathLibPassBase<LowerXlaMathLibPass> {
 public:
  LowerXlaMathLibPass()
      : impl::LowerXlaMathLibPassBase<LowerXlaMathLibPass>() {}

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LowerExpOpPattern, LowerLog1pPattern, LowerErfPattern,
                 LowerTruncF32BF16FPattern, RsqrtPattern, LowerTanhOpPattern>(
        &getContext(), module_op);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerXlaMathLibPass() {
  return std::make_unique<LowerXlaMathLibPass>();
}

}  // namespace emitters
}  // namespace xla
