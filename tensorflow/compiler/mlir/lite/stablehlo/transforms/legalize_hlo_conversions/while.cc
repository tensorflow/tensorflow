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

#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/legalize_hlo_conversions/while.h"

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Region.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir::odml {
namespace {

// Replaces `region`'s terminator to TFL::Yield.
void TFLReplaceReturnOp(Region& region, PatternRewriter& rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);

  for (auto& block : region.getBlocks()) {
    Operation* terminator = block.getTerminator();
    rewriter.setInsertionPoint(terminator);
    rewriter.replaceOpWithNewOp<TFL::YieldOp>(terminator,
                                              terminator->getOperands());
  }
}

class LeagalizeWhileOp : public OpConversionPattern<mhlo::WhileOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::WhileOp while_op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const final {
    // Creates a TFL::WhileOp to replace the mhlo::WhileOp. HLO WhileOp
    // currently doesn't support stateless, so this
    // parameters are set to the default values.
    auto is_stateless = rewriter.getBoolAttr(false);
    auto new_while = rewriter.create<TFL::WhileOp>(
        while_op.getLoc(), while_op->getResultTypes(), while_op->getOperands(),
        /*is_stateless=*/is_stateless);
    new_while.getCond().takeBody(while_op.getCond());
    new_while.getBody().takeBody(while_op.getBody());
    TFLReplaceReturnOp(new_while.getCond(), rewriter);
    TFLReplaceReturnOp(new_while.getBody(), rewriter);
    rewriter.replaceOp(while_op, new_while.getResults());
    return success();
  }
};

bool IsWhileLegal(mhlo::WhileOp while_op) {
  for (auto type : while_op->getOperandTypes()) {
    if (mlir::isa<TupleType>(type)) return true;
  }
  return false;
}

}  // namespace

void PopulateWhilePatterns(MLIRContext* ctx, RewritePatternSet& patterns,
                           ConversionTarget& target) {
  target.addDynamicallyLegalOp<mhlo::WhileOp>(IsWhileLegal);
  patterns.add<LeagalizeWhileOp>(ctx);
}

}  // namespace mlir::odml
