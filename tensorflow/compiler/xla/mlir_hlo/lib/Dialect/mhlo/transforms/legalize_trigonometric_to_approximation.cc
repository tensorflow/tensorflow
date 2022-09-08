/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

// This file implements the lowering for trigonometric standard ops to
// approximations.

#include <utility>

#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZETANHTOAPPROXIMATIONPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

namespace {

template <typename OpTy>
class ApproximateOnExtendedF32Lowering : public OpRewritePattern<OpTy> {
 public:
  explicit ApproximateOnExtendedF32Lowering(MLIRContext *ctx)
      : OpRewritePattern<OpTy>(ctx, /*benefit=*/100) {}

  virtual Value emitApproximation(ValueRange, Location,
                                  PatternRewriter &) const = 0;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto rawArgs = op.getOperation()->getOperands();

    // Supports only f16 and f32 for now.
    if (!op.getType().isF16() && !op.getType().isF32()) return failure();

    // Extend operands to f32 if needed and possible.
    SmallVector<Value, 2> f32Args;
    f32Args.reserve(rawArgs.size());
    for (Value arg : rawArgs) {
      // Similar to XLA, do not rewrite f64 as precision might matter.
      Type argTy = arg.getType();
      if (argTy.isF64()) return failure();

      if (argTy.isF16())
        arg = rewriter.create<arith::ExtFOp>(loc, rewriter.getF32Type(), arg);

      // If we still do not have f32, fail.
      if (!arg.getType().isF32()) return failure();

      f32Args.push_back(arg);
    }

    Value result = emitApproximation(f32Args, loc, rewriter);
    assert(result.getType().isF32() && "Expect f32 intermediate result.");

    // Truncate back if needed.
    if (op.getType().isF16())
      result =
          rewriter.create<arith::TruncFOp>(loc, rewriter.getF16Type(), result);

    rewriter.replaceOp(op, {result});
    return success();
  }
};

// This approximation resembles Eigen and realizes a constant approximation for
// the +/-1 limits on top.
// https://gitlab.com/libeigen/eigen/-/blob/master/Eigen/src/Core/MathFunctionsImpl.h
class ApproximateTanhLowering
    : public ApproximateOnExtendedF32Lowering<math::TanhOp> {
 public:
  explicit ApproximateTanhLowering(MLIRContext *ctx)
      : ApproximateOnExtendedF32Lowering<math::TanhOp>(ctx) {}

  // Emits the fast tanh approximation that is also used by XLA.
  Value emitApproximation(ValueRange args, Location loc,
                          PatternRewriter &rewriter) const override {
    Value input = args.front();
    assert(input.getType().isF32());
    static constexpr std::array<float, 7> numeratorCoeffs{
        -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
        5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
        4.89352455891786e-03f};
    static constexpr std::array<float, 4> denominatorCoeffs{
        1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
        4.89352518554385e-03f};

    // Materialize polynomial approximation.
    Value inputSquared = rewriter.create<arith::MulFOp>(loc, input, input);
    Value numerator = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(numeratorCoeffs[0]));
    for (int64_t i = 1; i < static_cast<int64_t>(numeratorCoeffs.size()); i++) {
      numerator = rewriter.create<arith::AddFOp>(
          loc, rewriter.create<arith::MulFOp>(loc, inputSquared, numerator),
          rewriter.create<arith::ConstantOp>(
              loc, rewriter.getF32FloatAttr(numeratorCoeffs[i])));
    }
    numerator = rewriter.create<arith::MulFOp>(loc, input, numerator);
    Value denominator = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF32FloatAttr(denominatorCoeffs[0]));
    for (int64_t i = 1; i < static_cast<int64_t>(denominatorCoeffs.size());
         i++) {
      denominator = rewriter.create<arith::AddFOp>(
          loc, rewriter.create<arith::MulFOp>(loc, inputSquared, denominator),
          rewriter.create<arith::ConstantOp>(
              loc, rewriter.getF32FloatAttr(denominatorCoeffs[i])));
    }
    Value approx = rewriter.create<arith::DivFOp>(loc, numerator, denominator);

    // For small values of |x|, we can approximate tanh(x) = x. For extremely
    // small values of x (|x| < 1e-37), the other approximation would evaluate
    // tanh(x) = 0.
    constexpr float kUseIdentityApprox = 0.0004;
    Value absInput = rewriter.create<math::AbsFOp>(loc, input);
    Value useIdentityApprox = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OLT, absInput,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(kUseIdentityApprox)));
    approx =
        rewriter.create<arith::SelectOp>(loc, useIdentityApprox, input, approx);

    // For very small/large values, use a constant approximation -1/1.
    Value tooLargeInput = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UGT, input,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(7.90531110763549805f)));
    Value tooSmallInput = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::ULT, input,
        rewriter.create<arith::ConstantOp>(
            loc, rewriter.getF32FloatAttr(-7.90531110763549805f)));
    Value inputIsNan = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::UNE, input, input);
    approx = rewriter.create<arith::SelectOp>(
        loc, tooLargeInput,
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(1.0)),
        approx);
    approx = rewriter.create<arith::SelectOp>(
        loc, tooSmallInput,
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(-1.0)),
        approx);
    approx = rewriter.create<arith::SelectOp>(loc, inputIsNan, input, approx);

    return approx;
  }
};

struct LegalizeTrigonometricToApproximationPass
    : public impl::LegalizeTanhToApproximationPassBase<
          LegalizeTrigonometricToApproximationPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateTrigonometricToApproximationPatterns(&getContext(), &patterns);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createLegalizeTrigonometricToApproximationPass() {
  return std::make_unique<LegalizeTrigonometricToApproximationPass>();
}

void populateTrigonometricToApproximationPatterns(mlir::MLIRContext *context,
                                                  RewritePatternSet *patterns) {
  // clang-format off
  patterns->add<ApproximateTanhLowering>(context);
  // clang-format on
}

}  // namespace mhlo
}  // namespace mlir
