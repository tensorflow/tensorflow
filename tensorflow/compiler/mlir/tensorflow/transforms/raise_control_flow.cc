/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for raising from the "TensorFlow control flow"
// dialect of MLIR to the standard TensorFlow dialect.  The TensorFlow control
// flow dialect represents control flow with Switch/Merge and a few related
// control flow nodes, along with control dependencies.
//
// This pass rebuilds them code in terms of MLIR branches and blocks,
// eliminating control dependencies, and results in the code being in the
// canonical TensorFlow dialect.

#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/control_flow_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace TFControlFlow {

namespace {
struct RaiseTFControlFlow : public FunctionPass<RaiseTFControlFlow> {
  void runOnFunction() {
    // First start by recognizing loops and reconstructing a loop tree.
    buildLoopNests();

    // Next, transform Switch/Merge and other control flow ops into proper
    // conditional control flow.
    buildConditionals();

    // Now that we have proper conditional control flow ops, the control edges
    // can be dropped, and the underscores removed from operation names.
    rewriteOps();
  }

  void buildLoopNests();
  void buildConditionals();
  void rewriteOps();
};

//===----------------------------------------------------------------------===//
// Loop nest reconstruction
//===----------------------------------------------------------------------===//

void RaiseTFControlFlow::buildLoopNests() {
  // TODO(clattner)
}

//===----------------------------------------------------------------------===//
// Conditional Reconstruction
//===----------------------------------------------------------------------===//

void RaiseTFControlFlow::buildConditionals() {
  // TODO.
}

//===----------------------------------------------------------------------===//
// Final rewrite from TF Control Flow form to canonical TensorFlow form
//===----------------------------------------------------------------------===//

static bool isUnderscoredTFOp(Operation &op) {
  return op.getName().getStringRef().startswith("_tf.");
}

// Drop control edges, and remove underscores from operation names.
void RaiseTFControlFlow::rewriteOps() {
  auto function = getFunction();
  OpBuilder builder(function.getBody());

  // On the first pass, create replacement operations for every one we are going
  // to replace, updating anything that uses the normal results with the newly
  // created operation.
  for (auto &bb : function) {
    for (auto &op : bb) {
      // Ignore any operations that we aren't looking for.
      if (!isUnderscoredTFOp(op)) continue;

      // We always insert the replacement operation next to the operation it
      // is replacing.
      builder.setInsertionPoint(&op);

      // Drop the leading _ off the name.
      OperationState result(op.getLoc(),
                            op.getName().getStringRef().drop_front());

      // Add an operand for each non-control input we find.  Control values
      // aren't necessary any more since the order within a block encodes the
      // same information.
      for (auto &operand : op.getOpOperands()) {
        if (!operand.get()->getType().isa<TFControlType>())
          result.operands.push_back(operand.get());

        // Drop all operands from the old operation, eliminating any
        // inter-dependencies after this pass.
        operand.drop();
      }

      // Add a result type for each non-control result we find.
      bool sawControlResult = false;
      for (auto opResult : op.getResults()) {
        if (opResult->getType().isa<TFControlType>()) {
          sawControlResult = true;
        } else {
          // We assume all control inputs are at the end of the result list.
          assert(!sawControlResult && "all control results must be last");
          (void)sawControlResult;
          result.types.push_back(opResult->getType());
        }
      }

      result.attributes.append(op.getAttrs().begin(), op.getAttrs().end());

      // Create the replacement operation.
      auto *replacement = builder.createOperation(result);

      // We know that all the control results are last, so we can just rewrite
      // the first results.
      for (unsigned i = 0, e = result.types.size(); i != e; ++i)
        op.getResult(i)->replaceAllUsesWith(replacement->getResult(i));
    }
  }

  // In the second pass, we can safely remove all of the old operations, because
  // we know that all inter-dependencies are dropped.
  for (auto &bb : function) {
    // Advance the iterator so we don't invalidate it when we remove an
    // operation later in the loop.
    for (auto &op : llvm::make_early_inc_range(bb))
      if (isUnderscoredTFOp(op)) op.erase();
  }
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateRaiseTFControlFlowPass() {
  return std::make_unique<RaiseTFControlFlow>();
}

static PassRegistration<RaiseTFControlFlow> pass(
    "tf-raise-control-flow",
    "Raise from the TensorFlow Control Flow "
    "dialect to the standard TensorFlow dialect");

}  // namespace TFControlFlow
}  // namespace mlir
