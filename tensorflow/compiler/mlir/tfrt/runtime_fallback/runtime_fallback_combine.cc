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

//===----------------------------------------------------------------------===//
//
// This file implements a set of simple combiners for optimizing operations in
// the Runtime Fallback dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_ops.h"

// This optimizes the following scenario:
// %tft0, %c2 = "tfd.move_dht_to_tft"(%dht0, %c1)
//     : (!dht.host_tensor, !tfrt.chain) -> (!tfd.tf_tensor, !tfrt.chain)
// %dht1, %c3 = "tfd.convert_tft_to_dht"(%tft0, %c2)
//     : (!tfd.tf_tensor, !tfrt.chain) -> (!dht.host_tensor, !tfrt.chain)
// some_op %dht1, %c3
//
// becomes
// some_op %dht0, %c1

struct SimplifyDoubleConversion
    : public mlir::OpRewritePattern<mlir::tfd::ConvertTftToDhtOp> {
  // We register this pattern to match every tfd.move_dht_to_tft op.
  // The "benefit" is used by the framework to order the patterns and process
  // them in order of profitability.
  explicit SimplifyDoubleConversion(mlir::MLIRContext* context)
      : mlir::OpRewritePattern<mlir::tfd::ConvertTftToDhtOp>(context,
                                                             /*benefit=*/1) {}

  // This method attempts to match a pattern and rewrite it. The rewriter
  // argument is the orchestrator of the sequence of rewrites. The pattern is
  // expected to interact with it to perform any changes to the IR from here.
  mlir::LogicalResult matchAndRewrite(
      mlir::tfd::ConvertTftToDhtOp op,
      mlir::PatternRewriter& rewriter) const override {
    // Look through the inputs of the ConvertTftToDhtOp.
    mlir::Value convert_op_input_0 = op.getOperand(0);
    mlir::Value convert_op_input_1 = op.getOperand(1);
    mlir::tfd::MoveDhtToTftOp move_input_op_0 =
        llvm::dyn_cast_or_null<mlir::tfd::MoveDhtToTftOp>(
            convert_op_input_0.getDefiningOp());
    mlir::tfd::MoveDhtToTftOp move_input_op_1 =
        llvm::dyn_cast_or_null<mlir::tfd::MoveDhtToTftOp>(
            convert_op_input_1.getDefiningOp());

    // The inputs should be MoveDhtToTftOp.
    if (!move_input_op_0 || !move_input_op_1) return mlir::failure();
    // Both inputs are the same MoveDhtToTftOp.
    if (move_input_op_0 != move_input_op_1) return mlir::failure();

    // Use the rewriter to replace the ConvertTftToDhtOp's users with the
    // operands of MoveDhtToTftOp.
    rewriter.replaceOp(
        op, {move_input_op_0.getOperand(0), move_input_op_0.getOperand(1)});
    return mlir::success();
  }
};

// Register rewrite pattern as "canonicalization" patterns on the MoveDhtToTftOp
// so that they can be picked up by the Canonicalization framework.
void mlir::tfd::ConvertTftToDhtOp::getCanonicalizationPatterns(
    RewritePatternSet& results, MLIRContext* context) {
  results.add<SimplifyDoubleConversion>(context);
}
