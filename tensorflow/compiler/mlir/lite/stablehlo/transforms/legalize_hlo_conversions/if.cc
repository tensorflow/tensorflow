/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/if.h"

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/util.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

class LegalizeIfOp : public OpConversionPattern<mhlo::IfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IfOp if_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    auto new_op = rewriter.create<TFL::IfOp>(
        if_op.getLoc(), if_op.getResultTypes(), if_op.getPred());

    new_op.getThenRegion().takeBody(if_op.getTrueBranch());
    new_op.getElseRegion().takeBody(if_op.getFalseBranch());

    ReplaceTerminatorWithYield(new_op.getThenRegion(), rewriter);
    ReplaceTerminatorWithYield(new_op.getElseRegion(), rewriter);

    rewriter.replaceOp(if_op, new_op.getResults());
    return success();
  }
};

}  // namespace

void PopulateIfPatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                        ConversionTarget& target) {
  patterns.add<LegalizeIfOp>(ctx);
  target.addIllegalOp<mhlo::IfOp>();
}

}  // namespace mlir::odml
