/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/case.h"

#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"

namespace mlir::odml {
namespace {

// Legalizes stablehlo.case op to tfl.if op.
// This pattern only supports stablehlo.case ops with exactly two branches.
class LegalizeCaseOp : public OpConversionPattern<stablehlo::CaseOp> {
 public:
  using OpConversionPattern<stablehlo::CaseOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      stablehlo::CaseOp case_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // stablehlo.case can have N branches, but tfl.if only supports two.
    if (case_op.getBranches().size() != 2) {
      return rewriter.notifyMatchFailure(
          case_op, "can only convert stablehlo.case with 2 branches");
    }

    // `stablehlo.case` takes an index, `tfl.if` takes a boolean predicate.
    // For a 2-branch `stablehlo.case` (branch 0 and branch 1), we need to map
    // the index to a boolean.
    // According to the stablehlo.case spec, an out-of-bounds index defaults to
    // the index of the last branch, which is 1 in this case. So, index 0 maps
    // to branch 0, and any other index (1, or out of bounds) maps to branch 1.
    // This can be expressed as a predicate `index != 0` for branch 1.

    auto loc = case_op->getLoc();
    auto index = case_op.getIndex();
    auto index_type = mlir::cast<ShapedType>(index.getType());

    // Create a constant tensor of the same shape as the index, filled with
    // zeros.
    auto const_zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(index_type));

    // Create the predicate `index != 0`.
    auto pred_type = index_type.clone(rewriter.getI1Type());
    auto pred = stablehlo::CompareOp::create(
        rewriter, loc, pred_type, index, const_zero,
        stablehlo::ComparisonDirectionAttr::get(
            rewriter.getContext(), stablehlo::ComparisonDirection::NE),
        stablehlo::ComparisonTypeAttr{});  // Default comparison type is fine
                                           // for integers.

    // Create the tfl.if op.
    auto tfl_if =
        TFL::IfOp::create(rewriter, loc, case_op.getResultTypes(), pred);

    // Branch 1 of stablehlo.case becomes the `then_region` of tfl.if.
    tfl_if.getThenRegion().takeBody(case_op.getBranches()[1]);
    ReplaceTerminatorWithYield(tfl_if.getThenRegion(), rewriter);

    // Branch 0 of stablehlo.case becomes the `else_region` of tfl.if.
    tfl_if.getElseRegion().takeBody(case_op.getBranches()[0]);
    ReplaceTerminatorWithYield(tfl_if.getElseRegion(), rewriter);

    rewriter.replaceOp(case_op, tfl_if.getResults());
    return success();
  }
};

}  // namespace

void PopulateCasePatterns(MLIRContext* context, RewritePatternSet& patterns,
                          ConversionTarget& target) {
  patterns.add<LegalizeCaseOp>(context);
  // Mark stablehlo.case as dynamically legal: it's legal if it does NOT have
  // exactly 2 branches, as those are the ones we want to convert.
  target.addDynamicallyLegalOp<stablehlo::CaseOp>(
      [](stablehlo::CaseOp op) { return op.getBranches().size() != 2; });
}

}  // namespace mlir::odml
