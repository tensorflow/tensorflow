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
    : public ApproximateOnExtendedF32Lowering<TanhOp> {
 public:
  explicit ApproximateTanhLowering(MLIRContext *ctx)
      : ApproximateOnExtendedF32Lowering<TanhOp>(ctx) {}

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

class ApproximateAtan2Lowering
    : public ApproximateOnExtendedF32Lowering<Atan2Op> {
 public:
  explicit ApproximateAtan2Lowering(MLIRContext *ctx)
      : ApproximateOnExtendedF32Lowering<Atan2Op>(ctx) {}

  // Reduces atan2 to atan in the same way XLA does it.
  Value emitApproximation(ValueRange args, Location loc,
                          PatternRewriter &rewriter) const override {
    Value y = args[0];
    Value x = args[1];
    assert(x.getType().isF32() && y.getType().isF32() &&
           "expect f32 arguments");
    Value ax = rewriter.create<AbsFOp>(loc, x);
    Value ay = rewriter.create<AbsFOp>(loc, y);
    Value le_ax_ay = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLE, ax, ay);
    Value min_ax_ay = rewriter.create<mlir::SelectOp>(loc, le_ax_ay, ax, ay);
    Value max_ax_ay = rewriter.create<mlir::SelectOp>(loc, le_ax_ay, ay, ax);
    Value zero_to_one = rewriter.create<DivFOp>(loc, min_ax_ay, max_ax_ay);
    Value a = emitAtanCoreApproximation(zero_to_one, loc, rewriter);

    Value pi_over_2 =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(1.57079637f));
    a = rewriter.create<mlir::SelectOp>(
        loc, le_ax_ay, rewriter.create<SubFOp>(loc, pi_over_2, a), a);

    Value zero = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(0));
    Value lt_x_0 = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT, x, zero);
    Value pi =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(3.14159274f));
    a = rewriter.create<mlir::SelectOp>(loc, lt_x_0,
                                        rewriter.create<SubFOp>(loc, pi, a), a);

    Value t = rewriter.create<mlir::SelectOp>(loc, lt_x_0, pi, zero);
    Value eq_y_0 = rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, y, zero);
    a = rewriter.create<mlir::SelectOp>(loc, eq_y_0, t, a);

    // Propagate nan.
    Value is_nan = rewriter.create<CmpFOp>(loc, CmpFPredicate::UNO, y, x);
    Value nan = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(std::numeric_limits<float>::quiet_NaN()));
    a = rewriter.create<mlir::SelectOp>(loc, is_nan, nan, a);

    // x and y are +- inf.
    Value three_pi_over_4 =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(2.3561945f));
    Value pi_over_4 = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(0.785398185f));
    t = rewriter.create<mlir::SelectOp>(loc, lt_x_0, three_pi_over_4,
                                        pi_over_4);
    Value inf = rewriter.create<ConstantOp>(
        loc, rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity()));
    Value eq_x_inf = rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, x, inf);
    Value eq_y_inf = rewriter.create<CmpFOp>(loc, CmpFPredicate::OEQ, y, inf);
    Value all_inf = rewriter.create<mlir::AndOp>(loc, eq_x_inf, eq_y_inf);
    a = rewriter.create<mlir::SelectOp>(loc, all_inf, t, a);

    return rewriter.create<CopySignOp>(loc, a, y);
  }

 private:
  // The core atan reduction derives from the heuristic described in
  // https://arxiv.org/abs/1508.03211 and has a < 0.95 ulp error in the [-1, 1]
  // range (though that assumed FMA was available, and it is not here).  This is
  // the same approximation that is also used by XLA.
  Value emitAtanCoreApproximation(Value x, Location loc,
                                  PatternRewriter &rewriter) const {
    auto constant = [&](float c) {
      return rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(c));
    };

    // Computes ab + c.
    auto mul_add = [&](Value a, Value b, Value c) {
      Value prod = rewriter.create<MulFOp>(loc, a, b);
      return rewriter.create<AddFOp>(loc, prod, c);
    };

    Value s = rewriter.create<MulFOp>(loc, x, x);
    Value r = constant(0.0027856871f);
    r = mul_add(r, s, constant(-0.0158660002f));
    r = mul_add(r, s, constant(0.042472221f));
    r = mul_add(r, s, constant(-0.0749753043f));
    r = mul_add(r, s, constant(0.106448799f));
    r = mul_add(r, s, constant(-0.142070308f));
    r = mul_add(r, s, constant(0.199934542f));
    r = mul_add(r, s, constant(-0.333331466f));
    r = rewriter.create<MulFOp>(loc, r, s);
    return mul_add(r, x, x);
  }
};

class ApproximateAtanLowering
    : public ApproximateOnExtendedF32Lowering<AtanOp> {
 public:
  explicit ApproximateAtanLowering(MLIRContext *ctx)
      : ApproximateOnExtendedF32Lowering<AtanOp>(ctx) {}

  // Reduce atan(x) to atan2(x, 1) to subsequently rely on an atan approximation
  // for the argument range [-1, 1].
  Value emitApproximation(ValueRange args, Location loc,
                          PatternRewriter &rewriter) const override {
    Value x = args.front();
    assert(x.getType().isF32());
    Value one = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(1));
    return rewriter.create<Atan2Op>(loc, x, one);
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
  patterns->insert<
      ApproximateAtanLowering,
      ApproximateAtan2Lowering,
      ApproximateTanhLowering>(context);
  // clang-format on
}

}  // namespace mhlo
}  // namespace mlir
