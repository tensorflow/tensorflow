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
#include <string>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
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

Value ConvertToF32(Value v, mlir::ImplicitLocOpBuilder& b) {
  mlir::FloatType f32_ty = b.getF32Type();
  if (v.getType() == f32_ty) {
    return v;
  }
  if (v.getType().getIntOrFloatBitWidth() < f32_ty.getWidth()) {
    return ma::ExtFOp::create(b, f32_ty, v);
  }
  return ma::TruncFOp::create(b, f32_ty, v);
}

struct RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
  RewriteTruncFPattern(mlir::MLIRContext* context, bool enable_f8,
                       bool enable_f4)
      : OpRewritePattern(context),
        enable_f8_(enable_f8),
        enable_f4_(enable_f4) {}

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    const bool is_f8 =
        llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(dst_ty);
    const bool is_f4 = llvm::isa<mlir::Float4E2M1FNType>(dst_ty);

    if ((is_f8 && enable_f8_) || (is_f4 && enable_f4_)) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      rewriter.replaceOp(op, EmitTruncFIntrinsic(src, dst_ty, b));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported float conversion");
  }

 private:
  bool enable_f8_;
  bool enable_f4_;

  Value EmitTruncFIntrinsic(Value value, mlir::FloatType to_ty,
                            mlir::ImplicitLocOpBuilder& b) const {
    assert((llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type,
                      mlir::Float4E2M1FNType>(to_ty)));

    ml::CallIntrinsicOp cvtOp;
    if (llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(to_ty) &&
        value.getType() == b.getF16Type()) {
      // Fast path for truncating F16 type.
      Value vec =
          ml::UndefOp::create(b, mlir::VectorType::get(2, value.getType()));
      vec = ml::InsertElementOp::create(b, vec, value,
                                        ma::ConstantIntOp::create(b, 0, 8));
      const std::string cvtIntr = llvm::isa<mlir::Float8E4M3FNType>(to_ty)
                                      ? "llvm.nvvm.f16x2.to.e4m3x2.rn"
                                      : "llvm.nvvm.f16x2.to.e5m2x2.rn";
      cvtOp = ml::CallIntrinsicOp::create(b, b.getIntegerType(16),
                                          b.getStringAttr(cvtIntr),
                                          mlir::ValueRange{vec});
    } else {
      // Other FP types get converted to F32 first.
      value = ConvertToF32(value, b);
      const std::string cvtIntr = llvm::isa<mlir::Float4E2M1FNType>(to_ty)
                                      ? "llvm.nvvm.ff.to.e2m1x2.rn.satfinite"
                                  : llvm::isa<mlir::Float8E4M3FNType>(to_ty)
                                      ? "llvm.nvvm.ff.to.e4m3x2.rn"
                                      : "llvm.nvvm.ff.to.e5m2x2.rn";
      cvtOp = ml::CallIntrinsicOp::create(b, b.getIntegerType(16),
                                          b.getStringAttr(cvtIntr),
                                          mlir::ValueRange{value, value});
    }

    Value res = ml::TruncOp::create(
        b, b.getIntegerType(to_ty.getIntOrFloatBitWidth()), cvtOp.getResults());

    if (llvm::isa<mlir::Float4E2M1FNType>(to_ty)) {
      return ma::BitcastOp::create(b, to_ty, res);
    }

    // Downcasting to float8 saturates the value (uses "satfinite" modifier).
    // Handle infinity separately to mitigate the issue.
    mlir::Type src_int_ty =
        b.getIntegerType(value.getType().getIntOrFloatBitWidth());
    return FixInfinityConversionValue(
        ma::BitcastOp::create(b, src_int_ty, value),
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
      return ma::ConstantIntOp::create(b, src.getType(), c);
    };
    int sign_pos = src.getType().getIntOrFloatBitWidth() - 1;
    Value sign_bit = ma::ShRUIOp::create(b, src, make_const(sign_pos));
    Value input =
        ma::AndIOp::create(b, src, make_const((1ull << sign_pos) - 1));

    // Values in the interval that contains all the values above the largest
    // representable in the destination type, as well as the infinity (source),
    // result in the infinity (destination).
    int64_t lower = GetOverflowInputValue(src_type, dst_type);
    int64_t upper = llvm::APFloat::getInf(src_type.getFloatSemantics())
                        .bitcastToAPInt()
                        .getZExtValue();
    Value is_inf = ma::AndIOp::create(
        b,
        ma::CmpIOp::create(b, ma::CmpIPredicate::ugt, input, make_const(lower)),
        ma::CmpIOp::create(b, ma::CmpIPredicate::ule, input,
                           make_const(upper)));

    // Build signed infinity result value.
    int64_t inf_val = llvm::APFloat::getInf(dst_type.getFloatSemantics())
                          .bitcastToAPInt()
                          .getZExtValue();
    Value sign_dst =
        ma::ShLIOp::create(b, ml::TruncOp::create(b, dst.getType(), sign_bit),
                           ma::ConstantIntOp::create(b, dst.getType(), 7));
    Value inf = ma::OrIOp::create(
        b, ma::ConstantIntOp::create(b, dst.getType(), inf_val), sign_dst);

    // Select result based on the predicate.
    Value res = ma::SelectOp::create(b, is_inf, inf, dst);
    return ma::BitcastOp::create(b, dst_type, res);
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
  RewriteExtFPattern(mlir::MLIRContext* context, bool enable_f8, bool enable_f4)
      : OpRewritePattern(context),
        enable_f8_(enable_f8),
        enable_f4_(enable_f4) {}

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());

    const bool is_f8 =
        llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type>(src.getType());
    const bool is_f4 = llvm::isa<mlir::Float4E2M1FNType>(src.getType());

    if ((is_f8 && enable_f8_) || (is_f4 && enable_f4_)) {
      mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
      rewriter.replaceOp(op, EmitExtFIntrinsic(src, dst_ty, b));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "unsupported float conversion");
  }

 private:
  bool enable_f8_;
  bool enable_f4_;

  Value EmitExtFIntrinsic(Value value, mlir::FloatType to_ty,
                          mlir::ImplicitLocOpBuilder& b) const {
    assert((llvm::isa<mlir::Float8E4M3FNType, mlir::Float8E5M2Type,
                      mlir::Float4E2M1FNType>(value.getType())));

    // Extend the smaller type to the FP16 type using the intrinsic, and then
    // to the destination type. In the case of BF16 go through the intermediate
    // FP32 type (as there's no F2F op for f16->bf16).
    const std::string cvtIntr =
        llvm::isa<mlir::Float4E2M1FNType>(value.getType())
            ? "llvm.nvvm.e2m1x2.to.f16x2.rn"
        : llvm::isa<mlir::Float8E4M3FNType>(value.getType())
            ? "llvm.nvvm.e4m3x2.to.f16x2.rn"
            : "llvm.nvvm.e5m2x2.to.f16x2.rn";
    Value input = ml::ZExtOp::create(
        b, b.getIntegerType(16),
        ma::BitcastOp::create(
            b, b.getIntegerType(value.getType().getIntOrFloatBitWidth()),
            value));

    mlir::FloatType f16_ty = b.getF16Type();
    auto cvtOp = ml::CallIntrinsicOp::create(
        b, mlir::VectorType::get(2, f16_ty), b.getStringAttr(cvtIntr),
        mlir::ValueRange{input});
    Value res = ml::ExtractElementOp::create(
        b, cvtOp.getResults(), ma::ConstantIntOp::create(b, 0, 8));
    if (to_ty.getWidth() > f16_ty.getWidth()) {
      res = ma::ExtFOp::create(b, to_ty, res);
    } else if (to_ty != f16_ty) {
      if (to_ty == b.getBF16Type()) {
        res = ma::ExtFOp::create(b, b.getF32Type(), res);
      }
      res = ma::TruncFOp::create(b, to_ty, res);
    }
    return res;
  }
};

class ConvertFloatNvidiaPass
    : public impl::ConvertFloatNvidiaPassBase<ConvertFloatNvidiaPass> {
 public:
  using ConvertFloatNvidiaPassBase::ConvertFloatNvidiaPassBase;

  void runOnOperation() override {
    const int cc_version =
        compute_capability_major_ * 10 + compute_capability_minor_;
    const int ptx_version = ptx_version_major_ * 10 + ptx_version_minor_;
    const bool enable_f8 = (ptx_version >= 78 && cc_version >= 90) ||
                           (ptx_version >= 81 && cc_version >= 89);
    const bool enable_f4 = (ptx_version >= 86 && cc_version >= 100);

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteTruncFPattern>(&getContext(), enable_f8, enable_f4);
    patterns.add<RewriteExtFPattern>(&getContext(), enable_f8, enable_f4);
    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateConvertFloatNvidiaPass(
    int compute_capability_major, int compute_capability_minor,
    int ptx_version_major, int ptx_version_minor) {
  ConvertFloatNvidiaPassOptions options;
  options.compute_capability_major_ = compute_capability_major;
  options.compute_capability_minor_ = compute_capability_minor;
  options.ptx_version_major_ = ptx_version_major;
  options.ptx_version_minor_ = ptx_version_minor;
  return std::make_unique<ConvertFloatNvidiaPass>(options);
}

}  // namespace gpu
}  // namespace xla
