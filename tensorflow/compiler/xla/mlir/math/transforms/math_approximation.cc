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
      patterns
          .add<EigenExpM1Approximation, LogApproximation, Log1pApproximation>(
              patterns.getContext());
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
