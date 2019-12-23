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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/IR/Value.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace tf_executor {

// Prunes unreachable operations of a tf_executor.graph operation.
void PruneGraph(GraphOp graph) {
  // A graph has a single block which forms a DAG: operations that aren't
  // reachable from the `fetch` operands can be eliminated.

  llvm::SmallPtrSet<Operation*, 8> reachable_ops;
  llvm::SmallVector<Operation*, 8> ops_to_visit;

  // Visit an op's operands if it is output of an Operation in same graph.
  auto visit_op = [&](Operation* op) {
    for (Value operand : op->getOperands()) {
      Operation* def = operand->getDefiningOp();
      if (def && def->getParentOp() == graph &&
          reachable_ops.insert(def).second) {
        // Op has not been visited, add to queue to visit later.
        ops_to_visit.push_back(def);
      }
    }
  };

  // Visit `fetch` operands.
  visit_op(graph.GetFetch());

  while (!ops_to_visit.empty()) {
    Operation* op = ops_to_visit.pop_back_val();
    if (auto island_op = llvm::dyn_cast<IslandOp>(op)) {
      // Visit island and island inner ops operands.
      op->walk([&](Operation* inner_op) { visit_op(inner_op); });
      continue;
    } else {
      // Op is not an island, only visit its operands.
      visit_op(op);
    }

    // If op is a `tf_executor.NextIteration.Source`, visit its associated
    // `tf_executor.NextIteration.Sink` op.
    if (auto source_op = llvm::dyn_cast<NextIterationSourceOp>(op)) {
      Operation* sink_op = source_op.GetSink().getOperation();
      if (reachable_ops.insert(sink_op).second) {
        ops_to_visit.push_back(sink_op);
      }
    }
  }

  // Erase unreachable ops in reverse order.
  for (Operation& op : llvm::make_early_inc_range(
           llvm::drop_begin(llvm::reverse(graph.GetBody()), 1))) {
    if (reachable_ops.find(&op) == reachable_ops.end()) {
      op.erase();
    }
  }
}

namespace {

// This transformation pass prunes a TF graph eliminating dead-nodes.
struct GraphPruning : public FunctionPass<GraphPruning> {
  void runOnFunction() override {
    getFunction().walk([](tf_executor::GraphOp graph) { PruneGraph(graph); });
  }
};

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateTFExecutorGraphPruningPass() {
  return std::make_unique<GraphPruning>();
}

static PassRegistration<GraphPruning> pass(
    "tf-executor-graph-pruning",
    "Prune unreachable nodes in a TensorFlow Graph.");

}  // namespace tf_executor
}  // namespace mlir
