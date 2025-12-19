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

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/string_view.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/codegen/intrinsic/erf.h"
#include "xla/codegen/intrinsic/exp.h"
#include "xla/codegen/intrinsic/fptrunc.h"
#include "xla/codegen/intrinsic/log1p.h"
#include "xla/codegen/intrinsic/rsqrt.h"
#include "xla/codegen/intrinsic/tanh.h"
#include "xla/codegen/intrinsic/type.h"

namespace xla {
namespace emitters {

#define GEN_PASS_DEF_LOWERXLAINTRINSICLIBPASS
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
      mlir::func::FuncOp::create(rewriter, module_op.getLoc(), name, func_type);
  func_decl.setPrivate();
  return func_decl;
}

class LowerErfPattern : public mlir::OpRewritePattern<mlir::math::ErfOp> {
 public:
  LowerErfPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : OpRewritePattern(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ErfOp op, mlir::PatternRewriter& rewriter) const override {
    auto type = op.getType();
    mlir::Type element_type = mlir::getElementTypeOrSelf(op.getType());
    auto maybe_vector_type = mlir::dyn_cast<mlir::VectorType>(type);

    if (maybe_vector_type && maybe_vector_type.getRank() != 1) {
      return rewriter.notifyMatchFailure(op, "Vector rank is not 1.");
    }

    // Get the vectorized version of the given type if op has a vector type,
    // else just return the given type.
    auto get_vector_type = [&maybe_vector_type](mlir::Type type) -> mlir::Type {
      if (maybe_vector_type) {
        return maybe_vector_type.clone(type);
      }
      return type;
    };

    // Extend the argument to f32 and truncate the result back unconditionally
    // as these will be cleaned up later if they are already f32.
    if (element_type.isF16() || element_type.isF32()) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      mlir::Type f32_type = get_vector_type(b.getF32Type());

      mlir::Value input_value =
          mlir::arith::ExtFOp::create(b, f32_type, op.getOperand());

      auto erf_decl = codegen::intrinsics::Erf::GetOrInsertDeclaration(
          rewriter, module_op_, Type::TypeFromIrType(f32_type));
      auto call_op = mlir::func::CallOp::create(b, erf_decl, input_value);

      mlir::Value f32_result = call_op.getResult(0);
      mlir::Value result = mlir::arith::TruncFOp::create(b, type, f32_result);

      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    if (element_type.isF64()) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

      auto erf_decl = GetErf64Declaration(rewriter);

      if (!maybe_vector_type) {
        auto call_op = mlir::func::CallOp::create(b, erf_decl, op.getOperand());
        rewriter.replaceOp(op, call_op->getResults());
        return mlir::success();
      }

      llvm::SmallVector<mlir::Value> scalar_erf_results;
      for (int64_t idx = 0; idx < maybe_vector_type.getNumElements(); ++idx) {
        mlir::Value extracted = mlir::vector::ExtractOp::create(
            rewriter, op.getLoc(), op.getOperand(), idx);
        mlir::Value scalar_erf =
            mlir::func::CallOp::create(b, erf_decl, extracted).getResult(0);
        scalar_erf_results.push_back(scalar_erf);
      }
      rewriter.replaceOpWithNewOp<mlir::vector::FromElementsOp>(
          op, type, scalar_erf_results);
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
    auto dst_ty = op.getType();

    if (!mlir::isa<mlir::Float32Type>(
            mlir::getElementTypeOrSelf(src.getType())) ||
        !mlir::isa<mlir::BFloat16Type>(mlir::getElementTypeOrSelf(dst_ty))) {
      return rewriter.notifyMatchFailure(op, "Not f32 -> bf16");
    }

    if (auto vec_type = mlir::dyn_cast<mlir::VectorType>(src.getType());
        vec_type && vec_type.getRank() != 1) {
      // These will later be converted to loops of 1D vectors but will then miss
      // the XLA intrinsic lowering.
      op->emitWarning() << "Missed XLA intrinsic lowering as vector rank != 1.";
      return rewriter.notifyMatchFailure(op, "Vector rank is not 1.");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    auto src_type = Type::TypeFromIrType(src.getType());
    auto dst_type = Type::TypeFromIrType(dst_ty);
    auto f32_to_bf16_decl =
        codegen::intrinsics::FpTrunc::GetOrInsertDeclaration(
            rewriter, module_op_, src_type, dst_type);
    auto call_op =
        mlir::func::CallOp::create(b, f32_to_bf16_decl, op.getOperand());
    rewriter.replaceOp(op, call_op->getResults());
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

template <typename Intrinsic, typename Op>
class LowerIntrinsicPattern : public mlir::OpRewritePattern<Op> {
 public:
  LowerIntrinsicPattern(mlir::MLIRContext* context, mlir::ModuleOp& module_op)
      : mlir::OpRewritePattern<Op>(context), module_op_(module_op) {}

  mlir::LogicalResult matchAndRewrite(
      Op op, mlir::PatternRewriter& rewriter) const override {
    auto vec_type = mlir::dyn_cast<mlir::VectorType>(op.getType());
    if (vec_type && vec_type.getRank() != 1) {
      // These will later be converted to loops of 1D vectors but will then miss
      // the XLA intrinsic lowering.
      op->emitWarning() << "Missed XLA intrinsic lowering as vector rank != 1.";
      return rewriter.notifyMatchFailure(op, "Vector rank is not 1.");
    }
    Type type = Type::TypeFromIrType(op.getType());
    Type scalar_type =
        Type::TypeFromIrType(mlir::getElementTypeOrSelf(op.getType()));
    mlir::StringAttr features =
        module_op_->getAttrOfType<mlir::StringAttr>("mhlo.cpu_features");
    const std::string features_str = !features ? "" : features.getValue().str();
    bool is_supported = Intrinsic::IsSupported(features_str, type);
    bool scalar_supported = Intrinsic::IsSupported(features_str, scalar_type);
    if (!is_supported && !scalar_supported) {
      return rewriter.notifyMatchFailure(op, "unsupported type");
    }

    if (is_supported) {
      auto intrinsic_decl =
          Intrinsic::GetOrInsertDeclaration(rewriter, module_op_, type);
      rewriter.replaceOpWithNewOp<mlir::func::CallOp>(op, intrinsic_decl,
                                                      op.getOperand());
    } else {
      // If the element type is supported but not the vector type, then we
      // decompose the vector op into a sequence of scalar ops. This is not
      // optimal in that we could split into the largest possible supported
      // vectorized ops, but it works for now.
      auto intrinsic_decl =
          Intrinsic::GetOrInsertDeclaration(rewriter, module_op_, scalar_type);

      llvm::SmallVector<mlir::Value> scalar_results;
      scalar_results.reserve(vec_type.getNumElements());
      for (int64_t idx = 0; idx != vec_type.getNumElements(); ++idx) {
        mlir::Value scalar_value = mlir::vector::ExtractOp::create(
            rewriter, op.getLoc(), op.getOperand(), idx);
        mlir::Value scalar_result =
            mlir::func::CallOp::create(rewriter, op.getLoc(), intrinsic_decl,
                                       scalar_value)
                .getResult(0);
        scalar_results.push_back(scalar_result);
      }
      rewriter.replaceOpWithNewOp<mlir::vector::FromElementsOp>(op, vec_type,
                                                                scalar_results);
    }
    return mlir::success();
  }

 private:
  mlir::ModuleOp& module_op_;
};

class LowerXlaIntrinsicLibPass
    : public impl::LowerXlaIntrinsicLibPassBase<LowerXlaIntrinsicLibPass> {
 public:
  LowerXlaIntrinsicLibPass()
      : impl::LowerXlaIntrinsicLibPassBase<LowerXlaIntrinsicLibPass>() {}

  void runOnOperation() override {
    mlir::ModuleOp module_op = getOperation();
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<
        LowerIntrinsicPattern<codegen::intrinsics::Exp, mlir::math::ExpOp>,
        LowerIntrinsicPattern<codegen::intrinsics::Log1p, mlir::math::Log1pOp>,
        LowerIntrinsicPattern<codegen::intrinsics::Rsqrt, mlir::math::RsqrtOp>,
        LowerIntrinsicPattern<codegen::intrinsics::Tanh, mlir::math::TanhOp>,
        LowerErfPattern, LowerTruncF32BF16FPattern>(&getContext(), module_op);

    if (mlir::failed(
            mlir::applyPatternsGreedily(module_op, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<mlir::Pass> CreateLowerXlaIntrinsicLibPass() {
  return std::make_unique<LowerXlaIntrinsicLibPass>();
}
}  // namespace emitters
}  // namespace xla
