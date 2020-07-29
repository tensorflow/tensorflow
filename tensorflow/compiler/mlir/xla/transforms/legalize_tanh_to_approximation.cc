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

// This file implements logic for lowering the tanh standard ops to an
// approximation.

#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"
#include "tensorflow/compiler/mlir/xla/transforms/rewriters.h"

namespace mlir {
namespace xla {
namespace {

/// Emits the fast tanh approximation that is also used by XLA.
static Value EmitTanhApproximation(Value input, Value abs_value, Location loc,
                                   PatternRewriter &rewriter) {
  // For small values of x, we can approximate tanh(x)=x. For extremely small
  // values of x (|x| < 1e-37), the other approximation would evaluate
  // tanh(x) = 0.
  constexpr float kCanUseApprox = 0.0004;
  Value can_use_approx =
      rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(kCanUseApprox));
  Value return_input = rewriter.create<CmpFOp>(loc, CmpFPredicate::OLT,
                                               abs_value, can_use_approx);

  // Clamp the input to [-9, 9].
  Value plus_nine =
      rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(9.0));
  Value smaller_than_nine =
      rewriter.create<CmpFOp>(loc, CmpFPredicate::ULE, abs_value, plus_nine);
  Value input_clamped = rewriter.create<SelectOp>(
      loc, smaller_than_nine, input,
      rewriter.create<CopySignOp>(loc, plus_nine, input));

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

class ApproximateTanhLowering : public OpRewritePattern<TanhOp> {
 public:
  explicit ApproximateTanhLowering(MLIRContext *ctx)
      : OpRewritePattern<TanhOp>(ctx, 100) {}

  LogicalResult matchAndRewrite(TanhOp tanhOp,
                                PatternRewriter &rewriter) const override {
    Type operand_type = tanhOp.getType();

    if (operand_type.isF64()) {
      // Similar to XLA, do not rewrite f64 as precision might matter.
      return failure();
    }

    Location loc = tanhOp.getLoc();
    Value input = tanhOp.operand();
    if (operand_type.isF16()) {
      input = rewriter.create<FPExtOp>(loc, input, rewriter.getF32Type());
    }

    // If we still do not have f32, fail.
    if (!input.getType().isF32()) {
      return failure();
    }

    // For |operand| > 20.0, we just return -1/1.
    constexpr double kMaxValue = 20.0;
    Value max_value =
        rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(kMaxValue));
    Value abs_value = rewriter.create<AbsFOp>(loc, input);

    Value one = rewriter.create<ConstantOp>(loc, rewriter.getF32FloatAttr(1.0));
    Value one_with_sign = rewriter.create<CopySignOp>(loc, one, input);

    Value smaller_than_twenty =
        rewriter.create<CmpFOp>(loc, CmpFPredicate::ULT, abs_value, max_value);

    // Otherwise, we use the approximation.
    Value approx = EmitTanhApproximation(input, abs_value, loc, rewriter);

    Value result = rewriter.create<SelectOp>(loc, smaller_than_twenty, approx,
                                             one_with_sign);

    // Truncate back if needed.
    if (operand_type.isF16()) {
      result = rewriter.create<FPTruncOp>(loc, result, rewriter.getF16Type());
    }

    rewriter.replaceOp(tanhOp, {result});
    return success();
  }
};

struct LegalizeTanhToApproximation
    : public PassWrapper<LegalizeTanhToApproximation, FunctionPass> {
  /// Perform the lowering of standard dialect operations to approximations.
  void runOnFunction() override {
    OwningRewritePatternList patterns;
    PopulateTanhToApproximationPatterns(&getContext(), &patterns);
    applyPatternsAndFoldGreedily(getFunction(), patterns);
  }
};

}  // anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
createLegalizeTanhToApproximationPass() {
  return std::make_unique<LegalizeTanhToApproximation>();
}

void PopulateTanhToApproximationPatterns(mlir::MLIRContext *context,
                                         OwningRewritePatternList *patterns) {
  patterns->insert<ApproximateTanhLowering>(context);
}

static PassRegistration<LegalizeTanhToApproximation> legalize_pass(
    "xla-legalize-tanh-to-approximation",
    "Legalize tanh from standard dialect to an approximation");

}  // namespace xla
}  // namespace mlir
