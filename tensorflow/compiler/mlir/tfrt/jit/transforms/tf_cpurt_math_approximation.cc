/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/ArrayRef.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

using ::llvm::ArrayRef;
using ::llvm::SmallVector;

using ::mlir::ImplicitLocOpBuilder;
using ::mlir::LogicalResult;
using ::mlir::OpRewritePattern;
using ::mlir::PatternRewriter;
using ::mlir::RewritePatternSet;
using ::mlir::Type;
using ::mlir::Value;
using ::mlir::VectorType;

namespace arith = ::mlir::arith;
namespace math = ::mlir::math;
namespace vector = ::mlir::vector;

using TypePredicate = ::llvm::function_ref<bool(Type)>;

// Returns vector shape if the element type is matching the predicate (scalars
// that do match the predicate have shape equal to `{1}`).
static llvm::Optional<SmallVector<int64_t, 2>> vectorShape(Type type,
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

// Returns vector shape of the type. If the type is a scalar returns `1`.
static SmallVector<int64_t, 2> vectorShape(Type type) {
  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? llvm::to_vector<2>(vectorType.getShape())
                    : SmallVector<int64_t, 2>{1};
}

// Returns vector element type. If the type is a scalar returns the argument.
static Type elementType(Type type) {
  auto vectorType = type.dyn_cast<VectorType>();
  return vectorType ? vectorType.getElementType() : type;
}

static bool isF32(Type type) { return type.isF32(); }

//----------------------------------------------------------------------------//
// Broadcast scalar types and values into vector types and values.
//----------------------------------------------------------------------------//

// Returns true if shape != {1}.
static bool isNonScalarShape(ArrayRef<int64_t> shape) {
  return shape.size() > 1 || shape[0] > 1;
}

// Broadcasts scalar type into vector type (iff shape is non-scalar).
static Type broadcast(Type type, ArrayRef<int64_t> shape) {
  assert(!type.isa<VectorType>() && "must be scalar type");
  return isNonScalarShape(shape) ? VectorType::get(shape, type) : type;
}

// Broadcasts scalar value into vector (iff shape is non-scalar).
static Value broadcast(ImplicitLocOpBuilder &builder, Value value,
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

static Value f32Cst(ImplicitLocOpBuilder &builder, float value) {
  return builder.create<arith::ConstantOp>(builder.getF32FloatAttr(value));
}

static Value i32Cst(ImplicitLocOpBuilder &builder, int32_t value) {
  return builder.create<arith::ConstantOp>(builder.getI32IntegerAttr(value));
}

//----------------------------------------------------------------------------//
// Helper functions to build math function approximations.
//----------------------------------------------------------------------------//

static Value min(ImplicitLocOpBuilder &builder, Value a, Value b) {
  return builder.create<mlir::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OLT, a, b), a, b);
}

static Value max(ImplicitLocOpBuilder &builder, Value a, Value b) {
  return builder.create<mlir::SelectOp>(
      builder.create<arith::CmpFOp>(arith::CmpFPredicate::OGT, a, b), a, b);
}

static Value clamp(ImplicitLocOpBuilder &builder, Value value, Value lowerBound,
                   Value upperBound) {
  return max(builder, min(builder, value, upperBound), lowerBound);
}

// Eigen's implementation of ldexp.
// ldexp(x, exp) = x * 2^exp
// Set e = min(max(exp, -278), 278)
//     b = floor(e/4)
// Then out = ((((x * 2^(b)) * 2^(b)) * 2^(b)) * 2^(e-3*b))
static Value ldexp(ImplicitLocOpBuilder &builder, Value x, Value exp) {
  assert(isF32(elementType(x.getType())) && "argument x must be f32 type");
  assert(isF32(elementType(exp.getType())) && "argument exp must be f32 type");

  auto shape = vectorShape(x.getType());
  auto exp_shape = vectorShape(exp.getType());
  assert(shape == exp_shape && "x and exp must be of equal shape");
  auto f32Vec = broadcast(builder.getF32Type(), shape);
  auto i32Vec = broadcast(builder.getI32Type(), shape);

  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, shape);
  };
  auto mulf = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };
  auto subi = [&](Value a, Value b) -> Value {
    return builder.create<arith::SubIOp>(a, b);
  };
  auto shli = [&](Value a, Value pos) -> Value {
    return builder.create<arith::ShLIOp>(a, pos);
  };

  Value cstMantBitsI = bcast(i32Cst(builder, 23));
  Value cstMaxExponent = bcast(f32Cst(builder, 278.0f));
  Value cstMinExponent = bcast(f32Cst(builder, -278.0f));
  Value cstBiasI = bcast(i32Cst(builder, 127));
  Value cst2I = bcast(i32Cst(builder, 2));

  Value e = clamp(builder, exp, cstMinExponent, cstMaxExponent);
  Value eI = builder.create<arith::FPToSIOp>(i32Vec, e);
  Value bI = builder.create<arith::ShRSIOp>(eI, cst2I);
  Value biasedBI = builder.create<arith::AddIOp>(bI, cstBiasI);
  Value c = builder.create<arith::BitcastOp>(
      f32Vec, shli(biasedBI, cstMantBitsI));               // 2^b
  Value out = mulf(mulf(mulf(x, c), c), c);                // x * 2^(3b)
  bI = subi(subi(subi(eI, bI), bI), bI);                   // e - 3b
  biasedBI = builder.create<arith::AddIOp>(bI, cstBiasI);  // 2^(e - 3b)
  c = builder.create<arith::BitcastOp>(f32Vec, shli(biasedBI, cstMantBitsI));
  out = mulf(out, c);
  return out;
}

struct EigenExpApproximation : public OpRewritePattern<math::ExpOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(math::ExpOp op,
                                PatternRewriter &rewriter) const final;
};

LogicalResult EigenExpApproximation::matchAndRewrite(
    math::ExpOp op, PatternRewriter &rewriter) const {
  auto shape = vectorShape(op.getOperand().getType(), isF32);
  if (!shape.hasValue())
    return rewriter.notifyMatchFailure(op, "unsupported operand type");
  ImplicitLocOpBuilder builder(op->getLoc(), rewriter);

  auto addf = [&](Value a, Value b) -> Value {
    return builder.create<arith::AddFOp>(a, b);
  };
  auto bcast = [&](Value value) -> Value {
    return broadcast(builder, value, *shape);
  };
  auto floor = [&](Value a) { return builder.create<math::FloorOp>(a); };
  auto fma = [&](Value a, Value b, Value c) {
    return builder.create<math::FmaOp>(a, b, c);
  };
  auto mulf = [&](Value a, Value b) -> Value {
    return builder.create<arith::MulFOp>(a, b);
  };

  Value cstOne = bcast(f32Cst(builder, 1.0f));
  Value cstHalf = bcast(f32Cst(builder, 0.5f));
  Value cstExpHi = bcast(f32Cst(builder, 88.723f));
  Value cstExpLo = bcast(f32Cst(builder, -88.723f));

  Value cstCephesLog2E = bcast(f32Cst(builder, 1.44269504088896341f));
  Value cstCephesExpP0 = bcast(f32Cst(builder, 1.9875691500E-4f));
  Value cstCephesExpP1 = bcast(f32Cst(builder, 1.3981999507E-3f));
  Value cstCephesExpP2 = bcast(f32Cst(builder, 8.3334519073E-3f));
  Value cstCephesExpP3 = bcast(f32Cst(builder, 4.1665795894E-2f));
  Value cstCephesExpP4 = bcast(f32Cst(builder, 1.6666665459E-1f));
  Value cstCephesExpP5 = bcast(f32Cst(builder, 5.0000001201E-1f));

  Value x = clamp(builder, op.getOperand(), cstExpLo, cstExpHi);
  Value m = floor(fma(x, cstCephesLog2E, cstHalf));

  Value cstCephesExpC1 = bcast(f32Cst(builder, -0.693359375f));
  Value cstCephesExpC2 = bcast(f32Cst(builder, 2.12194440e-4f));
  Value r = fma(m, cstCephesExpC1, x);
  r = fma(m, cstCephesExpC2, r);

  Value r2 = mulf(r, r);
  Value r3 = mulf(r2, r);

  Value y = fma(cstCephesExpP0, r, cstCephesExpP1);
  Value y1 = fma(cstCephesExpP3, r, cstCephesExpP4);
  Value y2 = addf(r, cstOne);
  y = fma(y, r, cstCephesExpP2);
  y1 = fma(y1, r, cstCephesExpP5);
  y = fma(y, r3, y1);
  y = fma(y, r2, y2);
  Value ret = max(builder, ldexp(builder, y, m), op.getOperand());
  rewriter.replaceOp(op, ret);
  return mlir::success();
}

static void populateMathApproximationPatterns(RewritePatternSet &patterns,
                                              ArrayRef<std::string> oplist) {
  for (const std::string &op : oplist) {
    if (op == "exp" || op == "all")
      patterns.add<EigenExpApproximation>(patterns.getContext());
  }
}

struct MathApproximationPass
    : public MathApproximationBase<MathApproximationPass> {
  explicit MathApproximationPass(ArrayRef<std::string> approx_oplist) {
    this->oplist = approx_oplist;
  }

  void runOnFunction() override;
};

void MathApproximationPass::runOnFunction() {
  mlir::OwningRewritePatternList patterns(&getContext());
  populateMathApproximationPatterns(patterns, oplist);
  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateMathApproximationPass(
    ArrayRef<std::string> oplist) {
  return std::make_unique<MathApproximationPass>(oplist);
}

}  // namespace tensorflow
