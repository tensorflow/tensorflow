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
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/log/check.h"
#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/emitters/implicit_arith_op_builder.h"
#include "xla/codegen/emitters/transforms/passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/mlir_hlo/mhlo/transforms/map_mhlo_to_scalar_op.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace emitters {

namespace ma = ::mlir::arith;

using ma::SelectOp;
using mlir::Value;

#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/codegen/emitters/transforms/passes.h.inc"

namespace {

using Val = ImplicitArithOpBuilder;

struct RewriteErf32Pattern : public mlir::OpRewritePattern<mlir::math::ErfOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ErfOp op, mlir::PatternRewriter& rewriter) const override {
    if (!op.getType().isF32()) {
      return rewriter.notifyMatchFailure(op, "not an f32 erf");
    }

    static const std::array<float, 5> kAlpha{
        0.00022905065861350646f, 0.0034082910107109506f, 0.050955695062380861f,
        0.18520832239976145f, 1.128379143519084f};

    static const std::array<float, 7> kBeta{-1.1791602954361697e-7,
                                            0.000023547966471313185f,
                                            0.0010179625278914885f,
                                            0.014070470171167667f,
                                            0.11098505178285362f,
                                            0.49746925110067538f,
                                            1.0f};

    // We clamp x to be within [-c;c] where c = erfinv(1-2^-23), outside of
    // which x should be +/-1.
    constexpr float kErfInvOneMinusHalfULP = 3.7439211627767994f;

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    auto c = [&](float v) -> Value {
      return b.create<ma::ConstantFloatOp>(llvm::APFloat(v),
                                           rewriter.getF32Type());
    };

    auto poly = [&](auto x, auto coefficients) -> Value {
      auto r = c(coefficients[0]);
      for (int i = 1; i < coefficients.size(); ++i) {
        r = b.create<mlir::math::FmaOp>(r, x, c(coefficients[i]));
      }
      return r;
    };

    Value x = op.getOperand();
    x = b.create<ma::MaximumFOp>(x, c(-kErfInvOneMinusHalfULP));
    x = b.create<ma::MinimumFOp>(x, c(kErfInvOneMinusHalfULP));
    Value x2 = b.create<ma::MulFOp>(x, x);

    rewriter.replaceOpWithNewOp<ma::DivFOp>(
        op, b.create<ma::MulFOp>(x, poly(x2, kAlpha)), poly(x2, kBeta));

    return mlir::success();
  }
};

int GetSignificandBits(mlir::FloatType ty) {
  return llvm::APFloat::semanticsPrecision(ty.getFloatSemantics()) - 1;
}

int GetExponentBias(mlir::FloatType ty) {
  return 1 - llvm::APFloat::semanticsMinExponent(ty.getFloatSemantics()) -
         llvm::isa<mlir::Float8E8M0FNUType>(ty);  // No zero exponent for E8M0.
}

bool IsFNUZ(mlir::FloatType ty) {
  return llvm::isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNUZType,
                   mlir::Float8E5M2FNUZType>(ty);
}

Value IsInf(Value value, mlir::ImplicitLocOpBuilder& b) {
  auto ty = mlir::cast<mlir::FloatType>(value.getType());
  if (mlir::LLVM::isCompatibleOuterType(ty)) {
    value = b.create<mlir::math::AbsFOp>(value);
    Value inf = b.create<ma::ConstantFloatOp>(
        llvm::APFloat::getInf(ty.getFloatSemantics()), ty);
    return b.create<ma::CmpFOp>(ma::CmpFPredicate::OEQ, value, inf);
  }

  assert(ty.getIntOrFloatBitWidth() <= 8);
  // F8E5M2, F8E4M3, F8E3M4 are the only 8 bit float with infinities.
  if (llvm::isa<mlir::Float8E5M2Type>(ty)) {
    Val bits{b.create<ma::BitcastOp>(b.getI8Type(), value), &b};
    return (bits & 0x7F) == 0x7C;
  } else if (llvm::isa<mlir::Float8E4M3Type>(ty)) {
    Val bits{b.create<ma::BitcastOp>(b.getI8Type(), value), &b};
    return (bits & 0x7F) == 0x78;
  } else if (llvm::isa<mlir::Float8E3M4Type>(ty)) {
    Val bits{b.create<ma::BitcastOp>(b.getI8Type(), value), &b};
    return (bits & 0x7F) == 0x70;
  } else {
    return b.create<ma::ConstantIntOp>(false, b.getI1Type());
  }
}

Value IsNaN(Value value, mlir::ImplicitLocOpBuilder& b) {
  auto ty = value.getType();
  if (mlir::LLVM::isCompatibleOuterType(ty)) {
    return b.create<ma::CmpFOp>(ma::CmpFPredicate::UNO, value, value);
  }
  if (llvm::isa<mlir::Float4E2M1FNType>(ty)) {
    return b.create<ma::ConstantIntOp>(false, b.getI1Type());
  }

  assert(ty.getIntOrFloatBitWidth() == 8);
  Val bits{b.create<ma::BitcastOp>(b.getI8Type(), value), &b};
  if (llvm::isa<mlir::Float8E5M2Type>(ty)) {
    return (bits & 0b0111'1111).cmp(ma::CmpIPredicate::ugt, 0b0111'1100);
  } else if (llvm::isa<mlir::Float8E4M3Type>(ty)) {
    return (bits & 0b0111'1111).cmp(ma::CmpIPredicate::ugt, 0b0111'1000);
  } else if (llvm::isa<mlir::Float8E4M3FNType>(ty)) {
    return (bits & 0b0111'1111) == 0b0111'1111;
  } else if (llvm::isa<mlir::Float8E3M4Type>(ty)) {
    return (bits & 0b0111'1111).cmp(ma::CmpIPredicate::ugt, 0b0111'0000);
  } else if (llvm::isa<mlir::Float8E8M0FNUType>(ty)) {
    return bits == 0xFF;
  }
  return bits == 0x80;
}

Value EmitReducePrecision(Value value, int exponent_bits, int mantissa_bits,
                          mlir::ImplicitLocOpBuilder& b) {
  mlir::mhlo::ReducePrecisionOp::Properties properties;
  properties.exponent_bits = b.getI32IntegerAttr(exponent_bits);
  properties.mantissa_bits = b.getI32IntegerAttr(mantissa_bits);
  return mlir::mhlo::MhloOpToStdScalarOp::mapOpOfType<
      mlir::mhlo::ReducePrecisionOp>(
      b.getLoc(), value.getType(), {value.getType()},
      mlir::mhlo::ReducePrecisionOp::Adaptor(value, nullptr, properties),
      /*attributes=*/std::nullopt, &b);
}

Value EmitF16ToF8e5m2(Value in, mlir::ImplicitLocOpBuilder& b) {
  Val in_bits{b.create<ma::BitcastOp>(b.getI16Type(), in), &b};
  // Use this method of checking for NaN because it's the same as what's used
  // in the reduce precision lowering.
  Value is_nan = (in_bits & 32767).cmp(ma::CmpIPredicate::ugt, 31744);

  Value value = EmitReducePrecision(in, 5, 2, b);
  value = b.create<ma::BitcastOp>(b.getI16Type(), value);
  value = b.create<ma::ShRUIOp>(value,
                                b.create<ma::ConstantIntOp>(8, b.getI16Type()));
  value = b.create<ma::TruncIOp>(b.getI8Type(), value);
  // When the input is NaN, just truncating can turn a NaN into an inf if the
  // mantissa becomes 0.
  value = b.create<ma::SelectOp>(
      is_nan, b.create<ma::ConstantIntOp>(0x7F, value.getType()), value);
  return b.create<ma::BitcastOp>(b.getType<mlir::Float8E5M2Type>(), value);
}

Value EmitFloatConversion(Value value, mlir::FloatType to_ty,
                          mlir::ImplicitLocOpBuilder& b) {
  using ma::CmpIPredicate;

  auto from_ty = mlir::cast<mlir::FloatType>(value.getType());
  if (to_ty == b.getType<mlir::Float8E5M2Type>() && from_ty == b.getF16Type()) {
    return EmitF16ToF8e5m2(value, b);
  }

  if (to_ty == b.getType<mlir::Float8E5M2Type>() &&
      from_ty == b.getBF16Type()) {
    // Going through f32 and f16 is significantly faster than the fallback code
    // below.
    return EmitF16ToF8e5m2(
        b.create<ma::TruncFOp>(b.getF16Type(),
                               b.create<ma::ExtFOp>(b.getF32Type(), value)),
        b);
  }

  // Fallback code. The generated code here is not good. If you end up here,
  // you might want to add a more specific conversion.
  // This is a port of ConvertImpl in
  // https://github.com/jax-ml/ml_dtypes/blob/main/ml_dtypes/include/float8.h

  int from_mantissa = GetSignificandBits(from_ty);
  int from_bias = GetExponentBias(from_ty);
  int from_min_exp =
      llvm::APFloat::semanticsMinExponent(from_ty.getFloatSemantics());
  int from_max_exp =
      llvm::APFloat::semanticsMaxExponent(from_ty.getFloatSemantics());
  auto from_int_ty = b.getIntegerType(from_ty.getIntOrFloatBitWidth());

  int to_mantissa = GetSignificandBits(to_ty);
  int to_bias = GetExponentBias(to_ty);
  int to_min_exp =
      llvm::APFloat::semanticsMinExponent(to_ty.getFloatSemantics());
  int to_max_exp =
      llvm::APFloat::semanticsMaxExponent(to_ty.getFloatSemantics());
  auto to_int_ty = b.getIntegerType(to_ty.getIntOrFloatBitWidth());

  mlir::IntegerType wide_int_ty;
  if (from_ty.getWidth() <= 8 && to_ty.getWidth() <= 8) {
    wide_int_ty = b.getI16Type();
  } else {
    wide_int_ty = b.getIntegerType(
        std::max(from_int_ty.getWidth(), to_int_ty.getWidth()));
    // Avoid overflow for bit shifts.
    auto may_overflow = [&](mlir::Type a, mlir::Type b) {
      return llvm::isa<mlir::Float8E8M0FNUType>(a) && b.isF16();
    };
    if (may_overflow(from_ty, to_ty) || may_overflow(to_ty, from_ty)) {
      wide_int_ty = b.getI32Type();
    }
  }
  auto convert_int = [&](mlir::Type ty, Value v) -> Val {
    if (v.getType() == ty) {
      return {v, &b};
    }
    if (ty.getIntOrFloatBitWidth() < v.getType().getIntOrFloatBitWidth()) {
      return {b.create<ma::TruncIOp>(ty, v), &b};
    }
    return {b.create<ma::ExtUIOp>(ty, v), &b};
  };

  int64_t exp_offset = to_bias - from_bias;
  int digit_shift = to_mantissa - from_mantissa;

  int from_width = value.getType().getIntOrFloatBitWidth();
  Val from_bits{b.create<ma::BitcastOp>(b.getIntegerType(from_width), value),
                &b};
  if (from_width < 8) {
    from_bits = convert_int(b.getIntegerType(8), from_bits);
  }

  auto cst = [&](mlir::Type ty, int64_t n) -> Val {
    return {b.create<ma::ConstantIntOp>(n, ty), &b};
  };

  // Shift bits to destination type, without sign bit.
  Val from_sign_bit;
  if (!llvm::isa<mlir::Float8E8M0FNUType>(from_ty)) {
    from_sign_bit = from_bits.shrui(from_width - 1) != 0;
    from_bits = from_bits & ((1ULL << (from_width - 1)) - 1);
  }

  auto cst_bits = [&](llvm::APFloat f) {
    return cst(b.getIntegerType(llvm::APFloat::getSizeInBits(f.getSemantics())),
               f.bitcastToAPInt().getZExtValue());
  };
  Value to_nan;
  Value to_inf;
  Val to_zero;

  // MX float types have neither infinities nor NaNs.
  if (llvm::isa<mlir::Float4E2M1FNType>(to_ty)) {
    to_zero = cst_bits(llvm::APFloat::getZero(to_ty.getFloatSemantics()));
    to_nan = to_zero | 0x8;
    to_inf = cst_bits(llvm::APFloat::getLargest(to_ty.getFloatSemantics()));
  } else if (llvm::isa<mlir::Float8E8M0FNUType>(to_ty)) {
    to_nan = cst_bits(llvm::APFloat::getNaN(to_ty.getFloatSemantics()));
    to_inf = to_nan;
    to_zero = Val{to_nan, &b};
  } else {
    to_inf = cst_bits(llvm::APFloat::getInf(to_ty.getFloatSemantics()));
    to_nan = cst_bits(llvm::APFloat::getNaN(to_ty.getFloatSemantics()));
    to_zero = cst_bits(llvm::APFloat::getZero(to_ty.getFloatSemantics()));
  }

  auto round_bits_to_nearest_even = [&](Val bits, Val roundoff,
                                        bool use_implicit_bit = false) {
    assert(bits.value().getType() == roundoff.value().getType());
    // Round to nearest even by adding a bias term.
    // Consider a bit pattern
    //   FFF...FLRTT...T,
    // where bits RTT...T need to be rounded-off.  We add a bias term to the
    // bit pattern s.t. a carry is introduced to round up only if
    // - L is 1, R is 1, OR
    // - L is 0, R is 1, any T is one.
    // We do this by adding L to a bit pattern consisting of all T = 1.
    Val bias = !use_implicit_bit
                   ? (bits.shrui(roundoff) & 1) +
                         (bits.MakeConstant(1).shl(roundoff - 1) - 1)
                   : bits.MakeConstant(1).shl(roundoff - 1);
    return bits + bias;
  };

  // Happy path: no subnormals, infinities or NaNs.
  Value result;
  {
    // Round the mantissa if it is shrinking.
    Val rounded_from_bits = convert_int(wide_int_ty, from_bits);
    if (digit_shift < 0) {
      rounded_from_bits =
          round_bits_to_nearest_even(
              rounded_from_bits, rounded_from_bits.MakeConstant(-digit_shift),
              /*use_implicit_bit=*/to_mantissa == 0) &
          ~((1ll << (-digit_shift)) - 1);
    }

    // Re-bias the exponent.
    rounded_from_bits = rounded_from_bits + (exp_offset << from_mantissa);

    // Check for overflows by aligning the significands. We always align the
    // narrower significand to the wider significand.
    int64_t to_highest = llvm::APFloat::getLargest(to_ty.getFloatSemantics())
                             .bitcastToAPInt()
                             .getZExtValue();
    int64_t aligned_highest = to_highest;
    if (digit_shift < 0) {
      aligned_highest <<= -digit_shift;
      // Shift down, all dropped bits should already be zero.
      result = rounded_from_bits.shrui(-digit_shift);
    } else {
      // Shift up, inserting zeros in the newly created digits.
      rounded_from_bits = rounded_from_bits.shl(digit_shift);
      result = rounded_from_bits;
    }
    result = convert_int(to_int_ty, result);

    // `From` supports larger values than `To`, we may overflow.
    if (std::make_pair(to_max_exp, to_mantissa) <
        std::make_pair(from_max_exp, from_mantissa)) {
      result = b.create<SelectOp>(
          rounded_from_bits.cmp(CmpIPredicate::ugt, aligned_highest), to_inf,
          result);
    }
  }

  auto i32_ty = b.getI32Type();
  Val biased_from_exp = convert_int(i32_ty, from_bits.shrui(from_mantissa));

  if (to_min_exp < from_min_exp) {
    // `To` supports more exponents near zero which means that some subnormal
    // values in `From` may become normal.

    // Subnormals.
    Val bits = convert_int(wide_int_ty, from_bits);

    // Determine exponent in target type.
    Value clz = convert_int(
        i32_ty, b.create<mlir::math::CountLeadingZerosOp>(from_bits));
    Value msb = cst(i32_ty, std::max(from_width, 8) - 1) - clz;
    Value normalization_factor = cst(i32_ty, from_mantissa) - msb;

    Val biased_exponent = cst(i32_ty, exp_offset + 1) - normalization_factor;
    // If the result is subnormal, adjust the subnormal bits to account for
    // the difference in exponent bias.
    Value subnormal_bits = bits;
    if (exp_offset < wide_int_ty.getWidth()) {
      subnormal_bits = bits.shl(exp_offset);
    }

    // Result is normal. Shift the mantissa to account for the number of
    // leading zero digits, and clear the hidden bit.
    // Insert the exponent bits.
    Value normal_bits =
        (bits.shl(convert_int(wide_int_ty, normalization_factor)) &
         ~(1 << from_mantissa)) |
        convert_int(wide_int_ty, biased_exponent).shl(from_mantissa);

    Value biased_exp_sle_zero = biased_exponent.cmp(CmpIPredicate::sle, 0);
    bits = Val(
        b.create<SelectOp>(biased_exp_sle_zero, subnormal_bits, normal_bits),
        &b);
    if (digit_shift >= 0) {
      bits = bits.shl(digit_shift);
    } else {
      bits = round_bits_to_nearest_even(
          bits, bits.MakeConstant(-digit_shift),
          /*use_implicit_bit=*/to_mantissa == 0 && exp_offset != 0);
      bits = bits.shrui(-digit_shift);
    }
    bits = convert_int(to_int_ty, bits);

    result = b.create<SelectOp>(biased_from_exp == 0, bits, result);
  } else if (to_min_exp > from_min_exp) {
    // `To` supports fewer exponents near zero which means that some values in
    // `From` may become subnormal.
    Val biased_to_exp = biased_from_exp + (to_bias - from_bias);
    // Subnormals and zero.
    // Round and shift mantissa down.
    Val from_has_leading_one = !llvm::isa<mlir::Float8E8M0FNUType>(from_ty)
                                   ? biased_from_exp != 0
                                   : cst(i32_ty, 1);
    Val from_has_leading_one_i32 = convert_int(i32_ty, from_has_leading_one);
    from_has_leading_one = convert_int(from_int_ty, from_has_leading_one);
    Val exponent_shift_i32 =
        (from_has_leading_one_i32 - biased_to_exp) - digit_shift;
    // Insert the implicit leading 1 bit on the mantissa for normalized
    // inputs.
    Val rounded_from_bits = (from_bits & ((1ll << from_mantissa) - 1)) |
                            from_has_leading_one.shl(from_mantissa);

    // NOTE: we need to round again from the original from_bits,
    // otherwise the lower precision bits may already be lost.  There is
    // an edge-case where rounding to a normalized value would normally
    // round down, but for a subnormal, we need to round up.
    Val exponent_shift_from_ty = convert_int(from_int_ty, exponent_shift_i32);
    Val exponent_shift_to_ty = convert_int(to_int_ty, exponent_shift_i32);
    Val positive_bits = convert_int(
        to_int_ty,
        round_bits_to_nearest_even(rounded_from_bits, exponent_shift_from_ty)
            .shrui(exponent_shift_from_ty));
    // To avoid UB, limit rounding and shifting to the full mantissa plus
    // leading 1.
    positive_bits =
        Val(b.create<SelectOp>(
                exponent_shift_i32.cmp(CmpIPredicate::sle, from_mantissa + 1),
                positive_bits, to_zero),
            &b);

    Val negative_bits = convert_int(to_int_ty, rounded_from_bits)
                            .shl(to_zero - exponent_shift_to_ty);
    Value bits =
        b.create<SelectOp>(exponent_shift_i32.cmp(CmpIPredicate::sgt, 0),
                           positive_bits, negative_bits);
    result = b.create<SelectOp>(biased_to_exp.cmp(CmpIPredicate::sle, 0), bits,
                                result);
  }

  Value result_is_inf = IsInf(value, b);
  Value input_is_nan = IsNaN(value, b);

  if (llvm::isa<mlir::Float8E8M0FNUType>(to_ty)) {
    // Converting a negative number to E8M0 results in NaN.
    input_is_nan = from_sign_bit | input_is_nan;
  } else if (IsFNUZ(to_ty)) {
    // Clear the sign bit if the result is zero (the output has no negative
    // zero). Handle the edge case when the input is zero and the result is not.
    Val result_is_non_zero =
        (digit_shift > 0 ? from_bits : Val{result, &b}) != 0;
    from_sign_bit = from_sign_bit & result_is_non_zero;
  } else if (IsFNUZ(from_ty)) {
    // Clear the sign bit if the input is NaN (it's positive but encoded as
    // negative 0).
    from_sign_bit = from_sign_bit ^ input_is_nan;
  }

  if (!llvm::isa<mlir::Float8E8M0FNUType>(from_ty)) {
    result = b.create<SelectOp>(from_bits == 0, to_zero, result);
  }
  result = b.create<SelectOp>(result_is_inf, to_inf, result);
  result = b.create<SelectOp>(input_is_nan, to_nan, result);

  // Insert sign bit.
  if (!llvm::isa<mlir::Float8E8M0FNUType>(from_ty)) {
    Value neg_result = Val{result, &b} | (1ll << (to_int_ty.getWidth() - 1));
    result = b.create<SelectOp>(from_sign_bit, neg_result, result);
  }
  result = b.create<ma::BitcastOp>(to_ty, result);
  return result;
}

struct RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (dst_ty.getWidth() > 8) {
      return rewriter.notifyMatchFailure(op, "not an 8 bit (or less) truncf");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitFloatConversion(src, dst_ty, b));
    return mlir::success();
  }
};

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    auto dst_ty = mlir::cast<mlir::FloatType>(op.getType());
    if (src.getType().getWidth() > 8) {
      return rewriter.notifyMatchFailure(op, "not an 8 bit (or less) extf");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    rewriter.replaceOp(op, EmitFloatConversion(src, dst_ty, b));
    return mlir::success();
  }
};

// Lowering for cmpf : f8 for float to pred conversions.
struct RewriteF8Cst : public mlir::OpRewritePattern<ma::CmpFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::CmpFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto lhs = mlir::cast<FloatValue>(op.getLhs());
    auto rhs = mlir::cast<FloatValue>(op.getRhs());

    if (lhs.getType().getWidth() > 8) {
      return rewriter.notifyMatchFailure(op, "not an 8 bit (or less) cmpf");
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    // Skip the f32 conversion if we're comparing UNE.cst.
    llvm::APFloat rhs_cst(rhs.getType().getFloatSemantics());
    if (op.getPredicate() == ma::CmpFPredicate::UNE &&
        mlir::matchPattern(rhs, mlir::m_ConstantFloat(&rhs_cst))) {
      mlir::Type int_ty = rewriter.getIntegerType(lhs.getType().getWidth());
      Val int_value{b.create<ma::BitcastOp>(int_ty, lhs), &b};
      int64_t constant = rhs_cst.bitcastToAPInt().getZExtValue();
      // If we're comparing to +-0, compare the absolute values.
      if (rhs_cst.isZero() && !IsFNUZ(lhs.getType())) {
        int64_t mask = (1 << (lhs.getType().getWidth() - 1)) - 1;
        int_value = int_value & mask;
        constant &= mask;
      }
      auto cst = b.create<ma::ConstantIntOp>(constant, int_ty);
      rewriter.replaceOpWithNewOp<ma::CmpIOp>(op, ma::CmpIPredicate::ne,
                                              int_value, cst);
      return mlir::success();
    }

    auto lhs_ext = b.create<ma::ExtFOp>(b.getF32Type(), lhs);
    auto rhs_ext = b.create<ma::ExtFOp>(b.getF32Type(), rhs);
    rewriter.replaceOpWithNewOp<ma::CmpFOp>(op, op->getResultTypes(),
                                            mlir::ValueRange{lhs_ext, rhs_ext},
                                            op->getAttrs());
    return mlir::success();
  }
};

struct RewriteAbsFPattern : public mlir::OpRewritePattern<mlir::math::AbsFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::AbsFOp op, mlir::PatternRewriter& rewriter) const override {
    using FloatValue = mlir::TypedValue<mlir::FloatType>;
    auto src = mlir::cast<FloatValue>(op.getOperand());
    // LowerGpuOpsToNVVMOps has a lowering for abs that doesn't work with bf16.
    // Once that's removed, remove the code for BF16 here.
    if (src.getType().getWidth() > 8 && !src.getType().isBF16()) {
      return rewriter.notifyMatchFailure(op,
                                         "not an f8 (or less) or bf16 absf");
    }

    // If type is unsigned (E8M0), the operation is no-op.
    if (!llvm::APFloat::semanticsHasSignedRepr(
            src.getType().getFloatSemantics())) {
      rewriter.replaceAllOpUsesWith(op, op.getOperand());
      return mlir::success();
    }

    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    mlir::Type i_ty = rewriter.getIntegerType(src.getType().getWidth());
    Val value{b.create<ma::BitcastOp>(i_ty, src), &b};
    int64_t mask = (1ull << (src.getType().getWidth() - 1)) - 1;
    value = value & mask;
    rewriter.replaceOpWithNewOp<ma::BitcastOp>(op, src.getType(), value);
    return mlir::success();
  }
};

template <typename Op>
struct RewriteIToFpPattern : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      Op op, mlir::PatternRewriter& rewriter) const override {
    if (op.getType().getIntOrFloatBitWidth() > 8) {
      return rewriter.notifyMatchFailure(op, "not an f8 (or less) itofp");
    }
    Value to_float =
        rewriter.create<Op>(op.getLoc(), rewriter.getF32Type(), op.getIn());
    rewriter.replaceOpWithNewOp<ma::TruncFOp>(op, op.getType(), to_float);
    return mlir::success();
  }
};

template <typename Op>
struct RewriteFpToIPattern : public mlir::OpRewritePattern<Op> {
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      Op op, mlir::PatternRewriter& rewriter) const override {
    if (op.getIn().getType().getIntOrFloatBitWidth() > 8) {
      return rewriter.notifyMatchFailure(op, "not an f8 (or less) fptoi");
    }
    Value to_f32 = rewriter.create<ma::ExtFOp>(
        op.getLoc(), rewriter.getF32Type(), op.getIn());
    rewriter.replaceOpWithNewOp<Op>(op, op.getType(), to_f32);
    return mlir::success();
  }
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;
  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteTruncFPattern, RewriteExtFPattern, RewriteAbsFPattern,
                 RewriteF8Cst, RewriteIToFpPattern<ma::SIToFPOp>,
                 RewriteIToFpPattern<ma::UIToFPOp>,
                 RewriteFpToIPattern<ma::FPToSIOp>,
                 RewriteFpToIPattern<ma::FPToUIOp>>(&getContext());
    mlir::populatePolynomialApproximateTanhPattern(patterns);
    patterns.add<RewriteErf32Pattern>(&getContext());
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

}  // namespace emitters
}  // namespace xla
