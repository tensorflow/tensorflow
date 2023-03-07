/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include <optional>
#include <string>
#include <utility>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/xla/mlir/math/transforms/passes.h"

namespace xla {
namespace {

#define GEN_PASS_DEF_MATHAPPROXIMATIONPASS
#include "tensorflow/compiler/xla/mlir/math/transforms/passes.h.inc"

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

using ::mlir::ImplicitLocOpBuilder;
using ::mlir::LogicalResult;
using ::mlir::OperationPass;
using ::mlir::OpRewritePattern;
using ::mlir::PatternRewriter;
using ::mlir::RewritePatternSet;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::VectorType;

namespace arith = ::mlir::arith;
namespace func = ::mlir::func;
namespace math = ::mlir::math;
namespace vector = ::mlir::vector;

using TypePredicate = ::llvm::function_ref<bool(Type)>;

#define LN2_VALUE \
  0.693147180559945309417232121458176568075500134360255254120680009493393621L
#define LOG2E_VALUE \
  1.442695040888963407359924681001892137426645954152985934135449406931109219L

// Returns vector shape if the element type is matching the predicate (scalars
// that do match the predicate have shape equal to `{1}`).
std::optional<SmallVector<int64_t, 2>> vectorShape(Type type,
                                                   TypePredicate pred) {
  // If the type matches the predicate then its shape is `{1}`.
  if (pred(type)) return SmallVector<int64_t, 2>{1};

  // Otherwise check if the type is a vector type.
  auto vectorType = type.dyn_cast<VectorType>();
  if (vectorType && pred(vectorType.getElementType())) {
    return llvm::to_vector<2>(vectorType.getShape());
  }

  return std::nullopt;
}

bool isF32(Type type) { return type.isF32(); }
bool isI32(Type type) { return type.isInteger(32); }

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Returns true if shape != {1}.
bool isNonScalarShape(ArrayRef<int64_t> shape) {
  return shape.size() > 1 || shape[0] > 1;
}

// Broadcasts scalar type into vector type (iff shape is non-scalar).
Type broadcast(Type type, ArrayRef<int64_t> shape) {
  assert(!type.isa<VectorType>() && "must be scalar type");
  return isNonScalarShape(shape) ? VectorType::get(shape, type) : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
Value broadcast(ImplicitLocOpBuilder &builder, Value value,
                ArrayRef<int64_t> shape) {
  assert(!value.getType().isa<VectorType>() && "must be scalar value");
  auto type = broadcast(value.getType(), shape);
  return isNonScalarShape(shape)
             ? builder.create<vector::BroadcastOp>(type, value)
             : value;
}

//----------------------------------------------------------------------------//
// Helper functions to create constants.
//----------------------------------------------------------------------------//

Value f32Cst(ImplicitLocOpBuilder &builder, float value) {
  return builder.create<arith::ConstantOp>(builder.getF32FloatAttr(value));
}

Value i32Cst(ImplicitLocOpBuilder &builder, int32_t value) {
  return builder.create<arith::ConstantOp>(builder.getI32IntegerAttr(value));
}

Value f32FromBits(ImplicitLocOpBuilder &builder, uint32_t bits) {
  Value i32v = i32Cst(builder, static_cast<int32_t>(bits));
  return builder.create<arith::BitcastOp>(builder.getF32Type(), i32v);
}

//----------------------------------------------------------------------------//
// Helper functions to build math functions approximations.
//----------------------------------------------------------------------------//

// Return the clamped value or NaN if value is NaN.
// Note: the bounds must be normal, not NaN's.
Value ClampWithNormals(ImplicitLocOpBuilder &builder,
                       const llvm::SmallVector<int64_t, 2> &shape, Value value,
                       float lower_bound, float upper_bound) {
  assert(!std::isnan(lower_bound));
  assert(!std::isnan(upper_bound));

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };

  auto select_cmp = [&builder](auto pred, Value value, Value bound) {
    return builder.create<arith::SelectOp>(
        builder.create<arith::CmpFOp>(pred, value, bound), value, bound);
  };

  // Note: prefer UGE/ULE vs. UGT/ULT, since they generate vmaxps/vminps vs.
  // vcmpleps+vmovaps on x86_64. The latter outcome is also obtained with
  // arith::{Max,Min}FOp.
  value = select_cmp(arith::CmpFPredicate::UGE, value,
                     bcast(f32Cst(builder, lower_bound)));
  value = select_cmp(arith::CmpFPredicate::ULE, value,
                     bcast(f32Cst(builder, upper_bound)));
  return value;
}

// Return the maximum of the two values or NaN if value is NaN
Value Max(ImplicitLocOpBuilder &builder, Value value, Value bound) {
  return builder.create<arith::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::UGE, value, bound),
      value, bound);
}

// Computes exp2 for an i32 argument.
Value Exp2I32(ImplicitLocOpBuilder &builder, Value arg) {
  auto shape = vectorShape(arg.getType(), isI32);
  assert(shape.has_value() && "arg must be of i32 type");

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  auto f32_vec = broadcast(builder.getF32Type(), *shape);
  // The exponent of f32 located at 23-bit.
  Value cst_exponent_bit = bcast(i32Cst(builder, 23));
  // Set the exponent bias to zero.
  Value cst_bias = bcast(i32Cst(builder, 127));

  Value biased_arg = builder.create<arith::AddIOp>(arg, cst_bias);
  Value exp2_i32 = builder.create<arith::ShLIOp>(biased_arg, cst_exponent_bit);
  Value exp2_f32 = builder.create<arith::BitcastOp>(f32_vec, exp2_i32);

  return exp2_f32;
}

// Decomposes given floating point value `arg` into a normalized fraction and
// an integral power of two (see std::frexp). Returned values have float type.
std::pair<Value, Value> Frexp(ImplicitLocOpBuilder &builder, Value arg,
                              bool isPositive = false) {
  auto shape = vectorShape(arg.getType(), isF32);
  assert(shape.has_value() && "arg must be of f32 type");

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  auto i32 = builder.getIntegerType(32);
  auto i32_vec = broadcast(builder.getI32Type(), *shape);
  auto f32_vec = broadcast(builder.getF32Type(), *shape);

  Value cst126f = f32Cst(builder, 126.0f);
  Value cst_half = f32Cst(builder, 0.5f);
  Value cst_inv_mant_mask = f32FromBits(builder, ~0x7f800000u);

  // Bitcast to i32 for bitwise operations.
  Value i32_half = builder.create<arith::BitcastOp>(i32, cst_half);
  Value i32_inv_mant_mask =
      builder.create<arith::BitcastOp>(i32, cst_inv_mant_mask);
  Value i32_arg = builder.create<arith::BitcastOp>(i32_vec, arg);

  // Compute normalized fraction.
  Value tmp0 = builder.create<arith::AndIOp>(i32_arg, bcast(i32_inv_mant_mask));
  Value tmp1 = builder.create<arith::OrIOp>(tmp0, bcast(i32_half));
  Value normalized_fraction = builder.create<arith::BitcastOp>(f32_vec, tmp1);

  // Compute exponent.
  Value arg0 = isPositive ? arg : builder.create<math::AbsFOp>(arg);
  Value biased_exponent_bits = builder.create<arith::ShRUIOp>(
      builder.create<arith::BitcastOp>(i32_vec, arg0),
      bcast(i32Cst(builder, 23)));
  Value biased_exponent =
      builder.create<arith::SIToFPOp>(f32_vec, biased_exponent_bits);
  Value exponent =
      builder.create<arith::SubFOp>(biased_exponent, bcast(cst126f));

  return {normalized_fraction, exponent};
}

struct ExpM1Approximation : public OpRewritePattern<math::ExpM1Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpM1Op op,
                                PatternRewriter &rewriter) const final;
};

// This approximation comes from XLA Classic.
LogicalResult ExpM1Approximation::matchAndRewrite(
    math::ExpM1Op op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  Value cst_zero = bcast(f32Cst(builder, 0.0f));
  Value cst_half = bcast(f32Cst(builder, 0.5f));
  Value cst_one = bcast(f32Cst(builder, 1.0f));

  // expm1(x) == tanh(x/2)*(exp(x)+1)
  // x/2 can underflow, if it does we approximate expm1 with x.
  Value x = op.getOperand();
  Value x_over_two = builder.create<arith::MulFOp>(x, cst_half);
  Value x_over_two_is_zero = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OEQ, x_over_two, cst_zero);
  Value abs_x = builder.create<math::AbsFOp>(x);

  Value abs_x_is_large =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, abs_x, cst_half);
  Value tanh_of_x_over_two = builder.create<math::TanhOp>(x_over_two);
  Value exp_of_x = builder.create<math::ExpOp>(x);
  Value exp_of_x_plus_one = builder.create<arith::AddFOp>(exp_of_x, cst_one);
  Value exp_of_x_minus_one = builder.create<arith::SubFOp>(exp_of_x, cst_one);

  Value expm1_of_x =
      builder.create<arith::MulFOp>(tanh_of_x_over_two, exp_of_x_plus_one);
  expm1_of_x = builder.create<arith::SelectOp>(abs_x_is_large,
                                               exp_of_x_minus_one, expm1_of_x);
  expm1_of_x =
      builder.create<arith::SelectOp>(x_over_two_is_zero, x, expm1_of_x);

  rewriter.replaceOp(op, expm1_of_x);
  return mlir::success();
}

struct ExpApproximation : public OpRewritePattern<math::ExpOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult ExpApproximation::matchAndRewrite(
    math::ExpOp op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value()) {
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  }

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  auto add = [&](Value a, Value b) -> Value {
    return builder.create<arith::AddFOp>(a, b);
  };
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };
  auto floor = [&](Value a) { return builder.create<math::FloorOp>(a); };
  auto fmla = [&](Value a, Value b, Value c) {
    return builder.create<math::FmaOp>(a, b, c);
  };
  auto mul = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };

  // Polynomial approximation. Originally from Cephes, but then modified for
  // XLA Classic.
  //
  // To compute e^x, we re-express it as
  //
  //   e^x = e^(a + b)
  //       = e^(a + n log(2))
  //       = e^a * 2^n.
  //
  // We choose n = round(x / log(2)), restricting the value of `a` to
  // (-log(2)/2, log(2)/2).  We then use a polynomial to compute e^a. The
  // relative error between our approximation and the true value of e^a is less
  // than 2^-22.5 for all values of `a` within this range.

  // Restrict input to a small range, including some values that evaluate to
  // +/- inf.  Note that for our lower bound, we choose log(2^-126) instead of
  // log(F32_EPSILON). We do so because this routine always flushes denormal
  // floating points to 0. Therefore, we only need to worry about exponentiating
  // up to the smallest representable non-denormal floating point, which is
  // 2^-126.

  // Constants.
  Value cst_half = bcast(f32Cst(builder, 0.5f));
  Value cst_one = bcast(f32Cst(builder, 1.0f));

  // 1/log(2)
  Value cst_log2ef = bcast(f32Cst(builder, 1.44269504088896341f));

  Value cst_exp_c1 = bcast(f32Cst(builder, -0.693359375f));
  Value cst_exp_c2 = bcast(f32Cst(builder, 2.12194440e-4f));
  Value cst_exp_p0 = bcast(f32Cst(builder, 1.9875691500E-4f));
  Value cst_exp_p1 = bcast(f32Cst(builder, 1.3981999507E-3f));
  Value cst_exp_p2 = bcast(f32Cst(builder, 8.3334519073E-3f));
  Value cst_exp_p3 = bcast(f32Cst(builder, 4.1665795894E-2f));
  Value cst_exp_p4 = bcast(f32Cst(builder, 1.6666665459E-1f));
  Value cst_exp_p5 = bcast(f32Cst(builder, 5.0000001201E-1f));

  // Our computations below aren't particularly sensitive to the exact choices
  // here, so we choose values a bit larger/smaller than
  //
  //   log(F32_MAX) = 88.723...
  //   log(2^-126) = -87.337...
  Value x = op.getOperand();
  x = ClampWithNormals(builder, *shape, x, -87.8f, 88.8f);
  Value n = floor(fmla(x, cst_log2ef, cst_half));

  // When we eventually do the multiplication in e^a * 2^n, we need to handle
  // the case when n > 127, the max fp32 exponent (so 2^n == inf) but e^a < 1
  // (so e^a * 2^n != inf).  There's a similar problem for n < -126, the
  // smallest fp32 exponent.
  //
  // A straightforward solution would be to detect n out of range and split it
  // up, doing
  //
  //   e^a * 2^n = e^a * 2^(n1 + n2)
  //             = (2^n1 * e^a) * 2^n2.
  //
  // But it turns out this approach is quite slow, probably because it
  // manipulates subnormal values.
  //
  // The approach we use instead is to clamp n to [-127, 127]. Let n' be the
  // value of n clamped to [-127, 127]. In the case where n' = 127, `a` can grow
  // up to as large as 88.8 - 127 * log(2) which is about 0.7703. Even though
  // this value of `a` is outside our previously specified range, e^a will still
  // only have a relative error of approximately 2^-16 at worse. In practice
  // this seems to work well enough; it passes our exhaustive tests, breaking
  // only one result, and by one ulp (we return exp(88.7228394) = max-float but
  // we should return inf).
  //
  // In the case where n' = -127, the original input value of x is so small that
  // e^x, our final answer, is less than 2^-126. Since 2^-126 is the smallest
  // normal floating point, and since we flush denormals, we simply return 0. We
  // do this in a branchless way by observing that our code for constructing 2^n
  // produces 0 if n = -127.
  //
  // The proof that n' = -127 implies e^x < 2^-126 is as follows:
  //
  //    n' = -127 implies n <= -127
  //              implies round(x / log(2)) <= -127
  //              implies x/log(2) < -126.5
  //              implies x < -126.5 * log(2)
  //              implies e^x < e^(-126.5 * log(2))
  //              implies e^x < 2^-126.5 < 2^-126
  //
  //    This proves that n' = -127 implies e^x < 2^-126.
  n = ClampWithNormals(builder, *shape, n, -127.0f, 127.0f);

  // Computes x = x - n' * log(2), the value for `a`
  x = fmla(cst_exp_c1, n, x);
  x = fmla(cst_exp_c2, n, x);

  // Polynomial to compute z = e^a, accurate for a in (-0.5, 0.5).
  Value z = fmla(x, cst_exp_p0, cst_exp_p1);
  z = fmla(z, x, cst_exp_p2);
  z = fmla(z, x, cst_exp_p3);
  z = fmla(z, x, cst_exp_p4);
  z = fmla(z, x, cst_exp_p5);
  z = fmla(z, mul(x, x), x);
  z = add(cst_one, z);

  // Convert n' to an i32.  This is safe because we clamped it above.
  auto i32_vec = broadcast(builder.getI32Type(), *shape);
  Value n_i32 = builder.create<arith::FPToSIOp>(i32_vec, n);

  // Creates the value 2^n' if -126 <= n' <= 127 and 0 if n' = -127.
  Value pow2 = Exp2I32(builder, n_i32);

  // Return z * 2^n' if -126 <= n' <= 127 and 0 if n = -127.
  Value ret = mul(z, pow2);

  rewriter.replaceOp(op, ret);
  return mlir::success();
}

template <typename Op>
struct LogApproximationBase : public OpRewritePattern<Op> {
  using OpRewritePattern<Op>::OpRewritePattern;

  /// Base 2 if 'base2' is set; natural logarithm (base e) otherwise.
  LogicalResult logMatchAndRewrite(Op op, PatternRewriter &rewriter,
                                   bool base2) const;
};

// This approximation comes from Julien Pommier's SSE math library.
// Link: http://gruntthepeon.free.fr/ssemath
template <typename Op>
LogicalResult LogApproximationBase<Op>::logMatchAndRewrite(
    Op op, PatternRewriter &rewriter, bool base2) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value()) {
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  }

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  Value cst_zero = bcast(f32Cst(builder, 0.0f));
  Value cst_one = bcast(f32Cst(builder, 1.0f));
  Value cst_neg_half = bcast(f32Cst(builder, -0.5f));

  // The smallest non denormalized float number.
  Value cst_min_norm_pos = bcast(f32FromBits(builder, 0x00800000u));
  Value cst_minus_inf = bcast(f32FromBits(builder, 0xff800000u));
  Value cst_pos_inf = bcast(f32FromBits(builder, 0x7f800000u));
  Value cst_nan = bcast(f32FromBits(builder, 0x7fc00000));

  // Polynomial coefficients.
  Value cst_cephes_sqrthf = bcast(f32Cst(builder, 0.707106781186547524f));
  Value cst_cephes_log_p0 = bcast(f32Cst(builder, 7.0376836292E-2f));
  Value cst_cephes_log_p1 = bcast(f32Cst(builder, -1.1514610310E-1f));
  Value cst_cephes_log_p2 = bcast(f32Cst(builder, 1.1676998740E-1f));
  Value cst_cephes_log_p3 = bcast(f32Cst(builder, -1.2420140846E-1f));
  Value cst_cephes_log_p4 = bcast(f32Cst(builder, +1.4249322787E-1f));
  Value cst_cephes_log_p5 = bcast(f32Cst(builder, -1.6668057665E-1f));
  Value cst_cephes_log_p6 = bcast(f32Cst(builder, +2.0000714765E-1f));
  Value cst_cephes_log_p7 = bcast(f32Cst(builder, -2.4999993993E-1f));
  Value cst_cephes_log_p8 = bcast(f32Cst(builder, +3.3333331174E-1f));

  Value x = op.getOperand();

  // Truncate input values to the minimum positive normal.
  x = Max(builder, x, cst_min_norm_pos);

  // Extract significant in the range [0.5,1) and exponent.
  std::pair<Value, Value> pair = Frexp(builder, x, /*isPositive=*/true);
  x = pair.first;
  Value e = pair.second;

  // Shift the inputs from the range [0.5,1) to [sqrt(1/2), sqrt(2)) and shift
  // by -1.0. The values are then centered around 0, which improves the
  // stability of the polynomial evaluation:
  //
  //   if( x < SQRTHF ) {
  //     e -= 1;
  //     x = x + x - 1.0;
  //   } else { x = x - 1.0; }
  Value mask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, x,
                                             cst_cephes_sqrthf);
  Value tmp = builder.create<arith::SelectOp>(mask, x, cst_zero);

  x = builder.create<arith::SubFOp>(x, cst_one);
  e = builder.create<arith::SubFOp>(
      e, builder.create<arith::SelectOp>(mask, cst_one, cst_zero));
  x = builder.create<arith::AddFOp>(x, tmp);

  Value x2 = builder.create<arith::MulFOp>(x, x);
  Value x3 = builder.create<arith::MulFOp>(x2, x);

  // Evaluate the polynomial approximant of degree 8 in three parts.
  Value y0, y1, y2;
  y0 = builder.create<math::FmaOp>(cst_cephes_log_p0, x, cst_cephes_log_p1);
  y1 = builder.create<math::FmaOp>(cst_cephes_log_p3, x, cst_cephes_log_p4);
  y2 = builder.create<math::FmaOp>(cst_cephes_log_p6, x, cst_cephes_log_p7);
  y0 = builder.create<math::FmaOp>(y0, x, cst_cephes_log_p2);
  y1 = builder.create<math::FmaOp>(y1, x, cst_cephes_log_p5);
  y2 = builder.create<math::FmaOp>(y2, x, cst_cephes_log_p8);
  y0 = builder.create<math::FmaOp>(y0, x3, y1);
  y0 = builder.create<math::FmaOp>(y0, x3, y2);
  y0 = builder.create<arith::MulFOp>(y0, x3);

  y0 = builder.create<math::FmaOp>(cst_neg_half, x2, y0);
  x = builder.create<arith::AddFOp>(x, y0);

  if (base2) {
    Value cst_log2e = bcast(f32Cst(builder, static_cast<float>(LOG2E_VALUE)));
    x = builder.create<math::FmaOp>(x, cst_log2e, e);
  } else {
    Value cst_ln2 = bcast(f32Cst(builder, static_cast<float>(LN2_VALUE)));
    x = builder.create<math::FmaOp>(e, cst_ln2, x);
  }

  Value invalid_mask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::ULT,
                                                     op.getOperand(), cst_zero);
  Value zero_mask = builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ,
                                                  op.getOperand(), cst_zero);
  Value pos_inf_mask = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OEQ, op.getOperand(), cst_pos_inf);

  // Filter out invalid values:
  //  • x == 0     -> -INF
  //  • x < 0      ->  NAN
  //  • x == +INF  -> +INF
  Value aproximation = builder.create<arith::SelectOp>(
      zero_mask, cst_minus_inf,
      builder.create<arith::SelectOp>(
          invalid_mask, cst_nan,
          builder.create<arith::SelectOp>(pos_inf_mask, cst_pos_inf, x)));

  rewriter.replaceOp(op, aproximation);

  return mlir::success();
}

struct Log2Approximation : public LogApproximationBase<math::Log2Op> {
  using LogApproximationBase::LogApproximationBase;

  LogicalResult matchAndRewrite(math::Log2Op op,
                                PatternRewriter &rewriter) const final {
    return logMatchAndRewrite(op, rewriter, /*base2=*/true);
  }
};

struct LogApproximation : public LogApproximationBase<math::LogOp> {
  using LogApproximationBase::LogApproximationBase;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const final {
    return logMatchAndRewrite(op, rewriter, /*base2=*/false);
  }
};

struct Log1pApproximation : public OpRewritePattern<math::Log1pOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::Log1pOp op,
                                PatternRewriter &rewriter) const final;
};

// Approximate log(1+x).
LogicalResult Log1pApproximation::matchAndRewrite(
    math::Log1pOp op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value()) {
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  }

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  // Approximate log(1+x) using the following, due to W. Kahan:
  //   u = x + 1.0;
  //   if (u == 1.0 || u == inf) return x;
  //   return x * log(u) / (u - 1.0);
  //          ^^^^^^^^^^^^^^^^^^^^^^
  //             "log_large" below.
  Value cst_one = bcast(f32Cst(builder, 1.0f));
  Value x = op.getOperand();
  Value u = builder.create<arith::AddFOp>(x, cst_one);
  Value u_small =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, u, cst_one);
  Value log_u = builder.create<math::LogOp>(u);
  Value u_inf =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, u, log_u);
  Value log_large = builder.create<arith::MulFOp>(
      x, builder.create<arith::DivFOp>(
             log_u, builder.create<arith::SubFOp>(u, cst_one)));
  Value approximation = builder.create<arith::SelectOp>(
      builder.create<arith::OrIOp>(u_small, u_inf), x, log_large);
  rewriter.replaceOp(op, approximation);
  return mlir::success();
}

struct TanhApproximation : public OpRewritePattern<math::TanhOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::TanhOp op,
                                PatternRewriter &rewriter) const final;
};

// This approximation comes from Eigen::generic_fast_tanh function.
LogicalResult TanhApproximation::matchAndRewrite(
    math::TanhOp op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value()) {
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  }

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  Value x = ClampWithNormals(builder, *shape, op.getOperand(),
                             -7.99881172180175781f, 7.99881172180175781f);

  // Mask for tiny values that are approximated with `operand`.
  Value tiny = bcast(f32Cst(builder, 0.0004f));
  Value tiny_mask = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OLT, builder.create<math::AbsFOp>(op.getOperand()),
      tiny);

  // The monomial coefficients of the numerator polynomial (odd).
  Value alpha1 = bcast(f32Cst(builder, 4.89352455891786e-03f));
  Value alpha3 = bcast(f32Cst(builder, 6.37261928875436e-04f));
  Value alpha5 = bcast(f32Cst(builder, 1.48572235717979e-05f));
  Value alpha7 = bcast(f32Cst(builder, 5.12229709037114e-08f));
  Value alpha9 = bcast(f32Cst(builder, -8.60467152213735e-11f));
  Value alpha11 = bcast(f32Cst(builder, 2.00018790482477e-13f));
  Value alpha13 = bcast(f32Cst(builder, -2.76076847742355e-16f));

  // The monomial coefficients of the denominator polynomial (even).
  Value beta0 = bcast(f32Cst(builder, 4.89352518554385e-03f));
  Value beta2 = bcast(f32Cst(builder, 2.26843463243900e-03f));
  Value beta4 = bcast(f32Cst(builder, 1.18534705686654e-04f));
  Value beta6 = bcast(f32Cst(builder, 1.19825839466702e-06f));

  // Since the polynomials are odd/even, we need x^2.
  Value x2 = builder.create<arith::MulFOp>(x, x);

  // Evaluate the numerator polynomial p.
  Value p = builder.create<math::FmaOp>(x2, alpha13, alpha11);
  p = builder.create<math::FmaOp>(x2, p, alpha9);
  p = builder.create<math::FmaOp>(x2, p, alpha7);
  p = builder.create<math::FmaOp>(x2, p, alpha5);
  p = builder.create<math::FmaOp>(x2, p, alpha3);
  p = builder.create<math::FmaOp>(x2, p, alpha1);
  p = builder.create<arith::MulFOp>(x, p);

  // Evaluate the denominator polynomial q.
  Value q = builder.create<math::FmaOp>(x2, beta6, beta4);
  q = builder.create<math::FmaOp>(x2, q, beta2);
  q = builder.create<math::FmaOp>(x2, q, beta0);

  // Divide the numerator by the denominator.
  Value res = builder.create<arith::SelectOp>(
      tiny_mask, x, builder.create<arith::DivFOp>(p, q));

  rewriter.replaceOp(op, res);

  return mlir::success();
}

void populateMathApproximationPatterns(RewritePatternSet &patterns,
                                       ArrayRef<std::string> oplist) {
  for (const std::string &op : oplist) {
    if (op == "all") {
      patterns.add<ExpApproximation, ExpM1Approximation, LogApproximation,
                   Log1pApproximation, Log2Approximation, TanhApproximation>(
          patterns.getContext());
    } else if (op == "exp") {
      patterns.add<ExpApproximation>(patterns.getContext());
    } else if (op == "expm1") {
      patterns.add<ExpM1Approximation>(patterns.getContext());
    } else if (op == "log") {
      patterns.add<LogApproximation>(patterns.getContext());
    } else if (op == "log1p") {
      patterns.add<Log1pApproximation>(patterns.getContext());
    } else if (op == "log2") {
      patterns.add<Log2Approximation>(patterns.getContext());
    } else if (op == "tanh") {
      patterns.add<TanhApproximation>(patterns.getContext());
    }
  }
}

struct MathApproximationPass
    : public impl::MathApproximationPassBase<MathApproximationPass> {
  explicit MathApproximationPass(ArrayRef<std::string> approx_oplist) {
    this->oplist = approx_oplist;
  }

  void runOnOperation() override;
};

void MathApproximationPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateMathApproximationPatterns(patterns, oplist);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateMathApproximationPass(
    ArrayRef<std::string> oplist) {
  return std::make_unique<MathApproximationPass>(oplist);
}

}  // namespace xla
