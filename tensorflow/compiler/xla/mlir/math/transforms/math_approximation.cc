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

// Returns vector shape if the element type is matching the predicate (scalars
// that do match the predicate have shape equal to `{1}`).
llvm::Optional<SmallVector<int64_t, 2>> vectorShape(Type type,
                                                    TypePredicate pred) {
  // If the type matches the predicate then its shape is `{1}`.
  if (pred(type)) return SmallVector<int64_t, 2>{1};

  // Otherwise check if the type is a vector type.
  auto vectorType = type.dyn_cast<VectorType>();
  if (vectorType && pred(vectorType.getElementType())) {
    return llvm::to_vector<2>(vectorType.getShape());
  }

  return llvm::None;
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
  assert(!isnan(lower_bound));
  assert(!isnan(upper_bound));

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

struct EigenExpM1Approximation : public OpRewritePattern<math::ExpM1Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpM1Op op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult EigenExpM1Approximation::matchAndRewrite(
    math::ExpM1Op op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  // expm1(x) = exp(x) - 1 = u - 1.
  // We have to handle it carefully when x is near 0, i.e. u ~= 1,
  // and when the input is ~= -inf, i.e. u - 1 ~= -1.
  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value cstNegOne = bcast(f32Cst(builder, -1.0f));
  Value x = op.getOperand();
  Value u = builder.create<math::ExpOp>(x);
  Value uEqOneOrNaN =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::UEQ, u, cstOne);
  Value uMinusOne = builder.create<arith::SubFOp>(u, cstOne);
  Value uMinusOneEqNegOne = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OEQ, uMinusOne, cstNegOne);
  // logU = log(u) ~= x
  Value logU = builder.create<math::LogOp>(u);

  // Detect exp(x) = +inf; written this way to avoid having to form +inf.
  Value isInf =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OEQ, logU, u);

  // (u - 1) * (x / ~x)
  Value expm1 = builder.create<arith::MulFOp>(
      uMinusOne, builder.create<arith::DivFOp>(x, logU));
  expm1 = builder.create<arith::SelectOp>(isInf, u, expm1);
  Value approximation = builder.create<arith::SelectOp>(
      uEqOneOrNaN, x,
      builder.create<arith::SelectOp>(uMinusOneEqNegOne, cstNegOne, expm1));
  rewriter.replaceOp(op, approximation);

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

struct LogApproximation : public OpRewritePattern<math::LogOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::LogOp op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult LogApproximation::matchAndRewrite(
    math::LogOp op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.has_value()) {
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  }

  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };

  Value cst_min_norm_pos = bcast(f32FromBits(builder, 0x00800000u));
  Value cst_zero = bcast(f32Cst(builder, 0.0f));

  Value x = op.getOperand();

  // Flush positive denormals to zero.
  Value less_than_zero =
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, x, cst_zero);
  Value less_than_min_norm_pos = builder.create<arith::CmpFOp>(
      arith::CmpFPredicate::OLT, x, cst_min_norm_pos);
  x = builder.create<arith::SelectOp>(
      less_than_min_norm_pos,
      builder.create<arith::SelectOp>(less_than_zero, x, cst_zero), x);

  // Emit Log2Op instead of LogOp to avoid an infinite match-and-rewrite loop.
  Value log2 = builder.create<math::Log2Op>(x);
  Value cst = bcast(f32Cst(builder, 6.93147181e-1f));
  Value res = builder.create<arith::MulFOp>(cst, log2);
  rewriter.replaceOp(op, res);
  return mlir::success();
}

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

void populateMathApproximationPatterns(RewritePatternSet &patterns,
                                       ArrayRef<std::string> oplist) {
  for (const std::string &op : oplist) {
    if (op == "all") {
      patterns.add<ExpApproximation, EigenExpM1Approximation, LogApproximation,
                   Log1pApproximation>(patterns.getContext());
    } else if (op == "exp") {
      patterns.add<ExpApproximation>(patterns.getContext());
    } else if (op == "expm1") {
      patterns.add<EigenExpM1Approximation>(patterns.getContext());
    } else if (op == "log") {
      patterns.add<LogApproximation>(patterns.getContext());
    } else if (op == "log1p") {
      patterns.add<Log1pApproximation>(patterns.getContext());
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
