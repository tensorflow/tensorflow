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

// This file implements the logic to lower some specific ops to external library
// calls.
//
// Here the external function is model by a `disc_ral.dispatch` op. We use
// `disc_ral.dispatch` to serve as a unified entrance of disc external
// calls due to following reasons.
// - `disc_ral.dispatch` ensures that the first argument is always the
//   `disc_ral.context`
// - `disc_ral.dispatch` simplifies the logic to handle different instantiations
//   of one op for different devices and different element types. For example,
//   we may have GEMM ops with different element types.

#include "mlir-hlo/Dialect/mhlo/IR/disc_ral_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace disc_ral {

namespace {

// Converting:
//   %output = disc_ral.recv_input(ctx, input_idx)
//     to
//   %output = disc_ral.dispatch(ctx, input_idx) {call_target_name =
//   "ral_recv_input", backend_config = "cpu"}
struct RecvInputOpConvertor : public OpRewritePattern<RecvInputOp> {
  using OpRewritePattern<RecvInputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(RecvInputOp op,
                                PatternRewriter& rewriter) const override {
    auto operands = op.getOperands();
    rewriter.replaceOpWithNewOp<DispatchOp>(op, op.getType(), operands.front(),
                                            operands.drop_front(),
                                            "ral_recv_input", false, "cpu");
    return success();
  }
};

// Converting:
//   disc_ral.send_output(ctx, output_idx, output)
//     to
//   disc_ral.dispatch(ctx, output_idx, output) {call_target_name =
//   "ral_send_output", backend_config = "cpu"}
struct SendOutputOpConvertor : public OpRewritePattern<SendOutputOp> {
  using OpRewritePattern<SendOutputOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SendOutputOp op,
                                PatternRewriter& rewriter) const override {
    auto operands = op.getOperands();
    rewriter.replaceOpWithNewOp<DispatchOp>(op, llvm::None, operands.front(),
                                            operands.drop_front(),
                                            "ral_send_output", false, "cpu");
    return success();
  }
};

struct RalLowerToLibraryCallPass
    : public RalLowerToLibraryCallPassBase<RalLowerToLibraryCallPass> {
  using RalLowerToLibraryCallPassBase<
      RalLowerToLibraryCallPass>::RalLowerToLibraryCallPassBase;

  void runOnFunction() override {
    FuncOp func = getFunction();
    MLIRContext* context = &getContext();
    OwningRewritePatternList patterns(context);
    patterns.insert<RecvInputOpConvertor, SendOutputOpConvertor>(context);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      func.emitError("applyPatternsAndFoldGreedily does not converge");
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> createRalLowerToLibraryCallPass() {
  return std::make_unique<RalLowerToLibraryCallPass>();
}

}  // namespace disc_ral
}  // namespace mlir
