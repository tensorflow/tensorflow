/* Copyright 2024 The OpenXLA Authors.

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
#include <optional>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h"


namespace xla {
namespace gpu {

#define GEN_PASS_DEF_CONVERTFLOATNVIDIAPASS
#include "xla/backends/gpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;
namespace ml = ::mlir::LLVM;
using mlir::Value;

int GetSignificandBits(mlir::FloatType ty) {
  return llvm::APFloat::semanticsPrecision(ty.getFloatSemantics()) - 1;
}

int GetExponentBias(mlir::FloatType ty) {
  return 1 - llvm::APFloat::semanticsMinExponent(ty.getFloatSemantics());
}

struct RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (!llvm::isa<mlir::Float8E4M3FNType>(dst_ty) &&
        !llvm::isa<mlir::Float8E5M2Type>(dst_ty)) {
      return rewriter.notifyMatchFailure(op, "unsupported float conversion");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitTruncToF8Intrinsic(src, dst_ty, b));
    return mlir::success();
  }

  Value EmitTruncToF8Intrinsic(Value value, mlir::FloatType to_ty,
                               mlir::ImplicitLocOpBuilder& b) const {
    assert((llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(to_ty)));

    ml::CallIntrinsicOp cvtOp;
    if (value.getType() == b.getF16Type()) {
      // Fast path for truncating F16 type.
      Value vec =
          b.create<ml::UndefOp>(ml::getFixedVectorType(value.getType(), 2));
      vec = b.create<ml::InsertElementOp>(vec, value,
                                          b.create<ma::ConstantIntOp>(0, 8));
      auto cvtIntr = llvm::isa<mlir::Float8E4M3FNType>(to_ty)
                         ? "llvm.nvvm.f16x2.to.e4m3x2.rn"
                         : "llvm.nvvm.f16x2.to.e5m2x2.rn";
      cvtOp = b.create<ml::CallIntrinsicOp>(b.getIntegerType(16),
                                            b.getStringAttr(cvtIntr),
                                            mlir::ValueRange{vec});
    } else {
      // Other FP types get converted to F32 first.
      mlir::FloatType f32_ty = b.getF32Type();
      if (value.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
        value = b.create<ma::ExtFOp>(f32_ty, value);
      } else if (value.getType() != f32_ty) {
        value = b.create<ma::TruncFOp>(f32_ty, value);
      }
      auto cvtIntr = llvm::isa<mlir::Float8E4M3FNType>(to_ty)
                         ? "llvm.nvvm.ff.to.e4m3x2.rn"
                         : "llvm.nvvm.ff.to.e5m2x2.rn";
      cvtOp = b.create<ml::CallIntrinsicOp>(b.getIntegerType(16),
                                            b.getStringAttr(cvtIntr),
                                            mlir::ValueRange{value, value});
    }
    Value res = b.create<ml::TruncOp>(b.getIntegerType(8), cvtOp.getResults());

    // Downcasting to float8 saturates the value (uses "satfinite" modifier).
    // Handle infinity separately to mitigate the issue.
    mlir::Type src_int_ty =
        b.getIntegerType(value.getType().getIntOrFloatBitWidth());
    return FixInfinityConversionValue(
        b.create<ma::BitcastOp>(src_int_ty, value),
        mlir::cast<mlir::FloatType>(value.getType()), res, to_ty, b);
  }

  // If converting the input value would result in an infinity, return infinity
  // (with sign copied); otherwise return the conversion result.
  //
  // The input values have integer types (source is wider than the destination),
  // and actual floating point types are passed as extra arguments.
  static Value FixInfinityConversionValue(Value src, mlir::FloatType src_type,
                                          Value dst, mlir::FloatType dst_type,
                                          mlir::ImplicitLocOpBuilder& b) {
    // Extract and discard sign bit.
    auto make_const = [&](int64_t c) {
      return b.create<ma::ConstantIntOp>(c, src.getType());
    };
    int sign_pos = src.getType().getIntOrFloatBitWidth() - 1;
    Value sign_bit = b.create<ma::ShRUIOp>(src, make_const(sign_pos));
    Value input = b.create<ma::AndIOp>(src, make_const((1ull << sign_pos) - 1));

    // Values in the interval that contains all the values above the largest
    // representable in the destination type, as well as the infinity (source),
    // result in the infinity (destination).
    int64_t lower = GetOverflowInputValue(src_type, dst_type);
    int64_t upper = llvm::APFloat::getInf(src_type.getFloatSemantics())
                        .bitcastToAPInt()
                        .getZExtValue();
    Value is_inf = b.create<ma::AndIOp>(
        b.create<ma::CmpIOp>(ma::CmpIPredicate::ugt, input, make_const(lower)),
        b.create<ma::CmpIOp>(ma::CmpIPredicate::ule, input, make_const(upper)));

    // Build signed infinity result value.
    int64_t inf_val = llvm::APFloat::getInf(dst_type.getFloatSemantics())
                          .bitcastToAPInt()
                          .getZExtValue();
    Value sign_dst =
        b.create<ma::ShLIOp>(b.create<ml::TruncOp>(dst.getType(), sign_bit),
                             b.create<ma::ConstantIntOp>(7, dst.getType()));
    Value inf = b.create<ma::OrIOp>(
        b.create<ma::ConstantIntOp>(inf_val, dst.getType()), sign_dst);

    // Select result based on the predicate.
    Value res = b.create<ma::SelectOp>(is_inf, inf, dst);
    return b.create<ma::BitcastOp>(dst_type, res);
  }

  // Calculate the minimum raw value (represented as an integer) that would
  // overflow when converting from `src_type` to `dst_type` (floating point).
  static int64_t GetOverflowInputValue(mlir::FloatType src_type,
                                       mlir::FloatType dst_type) {
    // Get type data from floating point semantics.
    int src_mantissa = GetSignificandBits(src_type);
    int src_bias = GetExponentBias(src_type);
    int dst_mantissa = GetSignificandBits(dst_type);
    int dst_bias = GetExponentBias(dst_type);
    assert(src_mantissa > dst_mantissa);
    assert(src_bias >= dst_bias);

    // Get the largest value, shift to wider type and correct the exponent.
    int64_t largest = llvm::APFloat::getLargest(dst_type.getFloatSemantics())
                          .bitcastToAPInt()
                          .getZExtValue();
    int64_t threshold = largest << (src_mantissa - dst_mantissa);
    threshold += int64_t{src_bias - dst_bias} << src_mantissa;

    // Some values above the threshold could still be rounded down, so the
    // actual threshold that rounds to infinity is higher.
    threshold |= (1ull << (src_mantissa - dst_mantissa - 1)) - (largest & 1);
    return threshold;
  }
};

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (!llvm::isa<mlir::Float8E4M3FNType>(src.getType()) &&
        !llvm::isa<mlir::Float8E5M2Type>(src.getType())) {
      return rewriter.notifyMatchFailure(op, "unsupported float conversion");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitExtFromF8Intrinsic(src, dst_ty, b));
    return mlir::success();
  }

  Value EmitExtFromF8Intrinsic(Value value, mlir::FloatType to_ty,
                               mlir::ImplicitLocOpBuilder& b) const {
    assert((llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(
        value.getType())));

    // Extend the smaller type to the FP16 type using the intrinsic, and then
    // to the destination type. In the case of BF16 go through the intermediate
    // FP32 type (as there's no F2F op for f16->bf16).
    Value input = b.create<ml::ZExtOp>(
        b.getIntegerType(16),
        b.create<ma::BitcastOp>(b.getIntegerType(8), value));
    auto cvtIntr = llvm::isa<mlir::Float8E4M3FNType>(value.getType())
                       ? "llvm.nvvm.e4m3x2.to.f16x2.rn"
                       : "llvm.nvvm.e5m2x2.to.f16x2.rn";
    mlir::FloatType f16_ty = b.getF16Type();
    auto cvtOp = b.create<ml::CallIntrinsicOp>(
        ml::getFixedVectorType(f16_ty, 2), b.getStringAttr(cvtIntr),
        mlir::ValueRange{input});
    Value res = b.create<ml::ExtractElementOp>(
        cvtOp.getResults(), b.create<ma::ConstantIntOp>(0, 8));
    if (to_ty.getWidth() > f16_ty.getWidth()) {
      res = b.create<ma::ExtFOp>(to_ty, res);
    } else if (to_ty != f16_ty) {
      if (to_ty == b.getBF16Type()) {
        res = b.create<ma::ExtFOp>(b.getF32Type(), res);
      }
      res = b.create<ma::TruncFOp>(to_ty, res);
    }
    return res;
  }
};

class ConvertFloatNvidiaPass
    : public impl::ConvertFloatNvidiaPassBase<ConvertFloatNvidiaPass> {
 public:
  using ConvertFloatNvidiaPassBase::ConvertFloatNvidiaPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteTruncFPattern, RewriteExtFPattern>(&getContext());
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateConvertFloatNvidiaPass() {
  return std::make_unique<ConvertFloatNvidiaPass>();
}

}  // namespace gpu
}  // namespace xla
