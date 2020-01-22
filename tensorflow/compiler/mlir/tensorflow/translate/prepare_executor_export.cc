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

#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Function.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// This pass is used in preparation for Graph export.
//
// For empty islands in the tf_executor dialect, a NoOp or Identity/IdentityN
// is inserted depending on if there are any data results. This allows the Graph
// exporter to assume all islands have 1 op, when mapping to a TensorFlow Node.

namespace mlir {

namespace {

struct PrepareExecutorExportPass
    : public FunctionPass<PrepareExecutorExportPass> {
  void runOnFunction() override;
};

// Finds empty IslandOps and populate them with a NoOp or Identity/IdentityN
// depending on if there are any data results.
void PopulateEmptyIslands(OpBuilder builder, tf_executor::GraphOp graph) {
  auto body = graph.GetBody().without_terminator();
  for (Operation& op : body) {
    auto island = llvm::dyn_cast<tf_executor::IslandOp>(op);
    if (!island || !island.GetBody().without_terminator().empty()) continue;

    builder.setInsertionPointToStart(&island.GetBody());
    tf_executor::YieldOp yield = island.GetYield();
    if (yield.getNumOperands() == 0) {
      builder.create<TF::NoOp>(island.getLoc(), llvm::ArrayRef<mlir::Type>{},
                               llvm::ArrayRef<mlir::Value>{},
                               llvm::ArrayRef<mlir::NamedAttribute>{});
    } else if (yield.getNumOperands() == 1) {
      Value operand = yield.getOperand(0);
      auto identity = builder.create<TF::IdentityOp>(
          island.getLoc(), operand.getType(), operand);
      yield.setOperand(0, identity.output());
    } else {
      auto types = llvm::to_vector<4>(yield.getOperandTypes());
      auto identity_n = builder.create<TF::IdentityNOp>(island.getLoc(), types,
                                                        yield.getOperands());
      for (auto it : llvm::enumerate(identity_n.getResults()))
        yield.setOperand(it.index(), it.value());
    }
  }
}

void PrepareExecutorExportPass::runOnFunction() {
  OpBuilder builder(getFunction().getContext());
  getFunction().walk([&](tf_executor::GraphOp graph) {
    PopulateEmptyIslands(builder, graph);
  });
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreatePrepareExecutorExportPass() {
  return std::make_unique<PrepareExecutorExportPass>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::PrepareExecutorExportPass> pass(
    "tf-executor-prepare-export",
    "Transforms TF executor dialect to a more friendly form for exporting.");
