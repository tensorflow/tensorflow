
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

#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

// This pass is used in preparation for Graph export.
// The GraphDef exporter expects each op to be in its own island.
// This pass puts the IR in that form.
//
// We do this as an IR->IR transform to keep the Graph exporter as simple as
// possible.

namespace mlir {

namespace {

struct BreakUpIslands : OperationPass<BreakUpIslands, FuncOp> {
  void runOnOperation() final;

  void BreakUpIsland(tf_executor::IslandOp op);
};

}  // end anonymous namespace

void BreakUpIslands::runOnOperation() {
  auto graph_op_range = getOperation().getBody().front().without_terminator();
  tf_executor::GraphOp graph_op;
  if (graph_op_range.begin() != graph_op_range.end() &&
      std::next(graph_op_range.begin()) == graph_op_range.end()) {
    graph_op = dyn_cast<tf_executor::GraphOp>(
        getOperation().getBody().front().front());
  }
  if (!graph_op) {
    getOperation().emitError("Expected function to contain only a graph_op");
    signalPassFailure();
    return;
  }

  for (auto& item : llvm::make_early_inc_range(graph_op.GetBody())) {
    if (auto island = dyn_cast<tf_executor::IslandOp>(&item)) {
      BreakUpIsland(island);
    }
  }
}

// Converts a single island into multiple islands (one for each op). The islands
// are chained together by control flow values.
void BreakUpIslands::BreakUpIsland(tf_executor::IslandOp op) {
  OpBuilder builder(op);
  OpBuilder island_builder(op);
  auto control_type = tf_executor::ControlType::get(&getContext());
  Value* previous_island = nullptr;
  auto tmp_control_inputs = llvm::to_vector<4>(op.controlInputs());
  ArrayRef<Value*> previous_control = tmp_control_inputs;
  // For each operator in the island,
  for (Operation& sub_op :
       llvm::make_early_inc_range(op.GetBody().without_terminator())) {
    auto loc = sub_op.getLoc();
    auto island = builder.create<tf_executor::IslandOp>(
        loc, llvm::to_vector<4>(sub_op.getResultTypes()), control_type,
        previous_control);
    island.body().push_back(new Block);
    Block* block = &island.body().back();
    sub_op.replaceAllUsesWith(island.outputs());
    block->getOperations().splice(block->begin(), op.GetBody().getOperations(),
                                  sub_op);
    island_builder.setInsertionPointToEnd(block);
    island_builder.create<tf_executor::YieldOp>(
        loc, llvm::to_vector<4>(sub_op.getResults()));
    previous_island = island.control();
    previous_control = previous_island;
  }
  // TODO(parkers): Potential problem where:
  //   island {
  //     ... %result = ops ...; print(%something);
  //     return %result
  //   }
  // could strand print() if there is no existing control dependency on the
  // island. This should be such that all uses are found to be inside other
  // islands or "tf_executor.fetch" ops and control dependencies on the last
  // island are added for each of these as well as replacing all usages of
  // op.control().
  if (previous_island || op.control()->use_empty()) {
    for (auto item : llvm::zip(op.outputs(), op.GetYield().fetches())) {
      std::get<0>(item)->replaceAllUsesWith(std::get<1>(item));
    }
    if (previous_island) {
      op.control()->replaceAllUsesWith(previous_island);
    }
    op.erase();
  }
}

}  // namespace mlir

static mlir::PassRegistration<mlir::BreakUpIslands> pass(
    "tf-executor-break-up-islands",
    "Transform from TF control dialect to TF executor dialect.");
