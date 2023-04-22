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

#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_traits.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

// Dequantize ops will produce 3x larger tensors, so we want to move it after
// some passthrought ops to reduce the memory consumption.
struct PushDownDequantize : public OpRewritePattern<DequantizeOp> {
  explicit PushDownDequantize(MLIRContext* context)
      : OpRewritePattern<DequantizeOp>(context) {}

  LogicalResult matchAndRewrite(DequantizeOp op,
                                PatternRewriter& rewriter) const override {
    if (!op->hasOneUse()) return failure();

    auto use = op->use_begin();
    Operation* user = use->getOwner();
    unsigned operand_index = use->getOperandNumber();
    if (user->hasTrait<OpTrait::IsTerminator>()) return failure();

    auto get_num_elements = [](Value value) {
      return value.getType().cast<TensorType>().getNumElements();
    };

    // If the op is the pass-through op with (3x) smaller output, the dequantize
    // op can be pushed down to the single result of this op.
    if (!llvm::dyn_cast<mlir::SameScalesOpInterface>(user) ||
        user->getNumResults() != 1 ||
        get_num_elements(user->getOperand(operand_index)) <=
            get_num_elements(user->getResult(0))) {
      return failure();
    }

    // Set the output type of the dequantize op and push it down.
    Type result_type = user->getResult(0).getType();
    op.output().setType(result_type);
    user->replaceAllUsesWith(op);

    // Set the input type of the pass through op and pull it up.
    Type user_new_type =
        QuantizedType::getQuantizedElementType(op.input().getType())
            .castFromExpressedType(result_type);
    user->getResult(0).setType(user_new_type);
    user->setOperand(operand_index, op.input());

    // Set the input of the dequantize to the result of the pass throught op.
    // And switch the order of the ops.
    op->setOperand(0, user->getResult(0));
    op->moveAfter(user);
    return success();
  }
};

// This transformation pass optimizes the op execution order of the ops in the
// model.
struct OptimizeOpOrderPass
    : public PassWrapper<OptimizeOpOrderPass, FunctionPass> {
  void runOnFunction() override;

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-optimize-op-order";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Optimize the execution order of the ops.";
  }
};

void OptimizeOpOrderPass::runOnFunction() {
  OwningRewritePatternList patterns(&getContext());
  auto func = getFunction();
  auto* ctx = func.getContext();
  patterns.insert<PushDownDequantize>(ctx);
  if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

// Creates an instance of the TensorFlow Lite optimize op order pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizeOpOrderPass() {
  return std::make_unique<OptimizeOpOrderPass>();
}

static PassRegistration<OptimizeOpOrderPass> pass;

}  // namespace TFL
}  // namespace mlir
