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

#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
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
    auto raw_args = op.getOperation()->getOperands();

    // Supports only f16 and f32 for now.
    if (!op.getType().isF16() && !op.getType().isF32()) return failure();

    // Extend operands to f32 if needed and possible.
    SmallVector<Value, 2> f32_args;
    f32_args.reserve(raw_args.size());
    for (Value arg : raw_args) {
      // Similar to XLA, do not rewrite f64 as precision might matter.
      Type arg_ty = arg.getType();
      if (arg_ty.isF64()) return failure();

      if (arg_ty.isF16())
        arg = rewriter.create<FPExtOp>(loc, arg, rewriter.getF32Type());

      // If we still do not have f32, fail.
      if (!arg.getType().isF32()) return failure();

      f32_args.push_back(arg);
    }

    Value result = emitApproximation(f32_args, loc, rewriter);
    assert(result.getType().isF32() && "Expect f32 intermediate result.");

    // Truncate back if needed.
    if (op.getType().isF16())
      result = rewriter.create<FPTruncOp>(loc, result, rewriter.getF16Type());

    rewriter.replaceOp(op, {result});
    return success();
  }
};

class ApproximateTanhLowering
    : public ApproximateOnExtendedF32Lowering<math::TanhOp> {
 public:
  explicit ApproximateTanhLowering(MLIRContext *ctx)
      : ApproximateOnExtendedF32Lowering<math::TanhOp>(ctx) {}

  // Emits the fast tanh approximation that is also used by XLA.
  Value emitApproximation(ValueRange args, Location loc,
                          PatternRewriter &rewriter) const override {
    // For small values of x, we can approximate tanh(x) = x.  For extremely
    // small values of x (|x| < 1e-37), the other approximation would evaluate
    // tanh(x) = 0.
    Value input = args.front();
    assert(input.getType().isF32());
    constexpr float kCanUseApprox = 0.0004;
    Value abs_value = rewriter.create<AbsFOp>(loc, input);
    Value can_use_approx = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(kCanUseApprox));
    Value return_input = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT,
                                                 abs_value, can_use_approx);
    // Clamp the input to [-c, c].
    Value max_clamp = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(7.90531110763549805f));
    Value smaller_than_max =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ULE, input, max_clamp);
    Value clamped_half =
        rewriter.create<SelectOp>(loc, smaller_than_max, input, max_clamp);
    Value min_clamp = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(-7.90531110763549805f));
    Value larger_than_min = rewriter.create<CmpFOp>(loc, CmpFPredicate::UGE,
                                                    clamped_half, min_clamp);
    Value input_clamped = rewriter.create<SelectOp>(loc, larger_than_min,
                                                    clamped_half, min_clamp);

    static constexpr std::array<float, 7> numerator_coeffs{
        -2.76076847742355e-16f, 2.00018790482477e-13f, -8.60467152213735e-11f,
        5.12229709037114e-08f,  1.48572235717979e-05f, 6.37261928875436e-04f,
        4.89352455891786e-03f};

    static constexpr std::array<float, 4> denominator_coeffs{
        1.19825839466702e-06f, 1.18534705686654e-04f, 2.26843463243900e-03f,
        4.89352518554385e-03f};

    Value input_squared =
        rewriter.create<MulFOp>(loc, input_clamped, input_clamped);
    Value numerator = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(numerator_coeffs[0]));
    for (int i = 1; i < numerator_coeffs.size(); i++) {
      numerator = rewriter.create<AddFOp>(
          loc, rewriter.create<MulFOp>(loc, input_squared, numerator),
          rewriter.create<ConstantOp>(
              loc, rewriter.getF32FloatAttr(numerator_coeffs[i])));
    }

    numerator = rewriter.create<MulFOp>(loc, input_clamped, numerator);

    Value denominator = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(denominator_coeffs[0]));
    for (int i = 1; i < denominator_coeffs.size(); i++) {
      denominator = rewriter.create<AddFOp>(
          loc, rewriter.create<MulFOp>(loc, input_squared, denominator),
          rewriter.create<ConstantOp>(
              loc, rewriter.getF32FloatAttr(denominator_coeffs[i])));
    }

    Value approx = rewriter.create<DivFOp>(loc, numerator, denominator);

    return rewriter.create<SelectOp>(loc, return_input, input, approx);
  }
};

struct LegalizeTrigonometricToApproximationPass
    : public PassWrapper<LegalizeTrigonometricToApproximationPass,
                         FunctionPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    PopulateTrigonometricToApproximationPatterns(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(getFunction(), std::move(patterns));
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createLegalizeTrigonometricToApproximationPass() {
  return std::make_unique<LegalizeTrigonometricToApproximationPass>();
}

void PopulateTrigonometricToApproximationPatterns(
    mlir::MLIRContext *context, OwningRewritePatternList *patterns) {
  // clang-format off
  patterns->insert<ApproximateTanhLowering>(context);
  // clang-format on
}

}  // namespace mhlo
}  // namespace mlir
