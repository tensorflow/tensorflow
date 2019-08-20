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
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace tf_executor {

// Prunes a TF graph eliminating dead nodes.
void prune_graph(GraphOp graph) {
  // A graph has a single block which forms a DAG: nodes that aren't reachable
  // from the `fetch` operands can be eliminated.

  // Delete unreachable node from the graph. We traverse it in reverse order so
  // that we just have to check that a node does not have any users to delete
  // it.
  for (Operation &op : llvm::make_early_inc_range(
           llvm::drop_begin(llvm::reverse(graph.GetBody()), 1))) {
    // NextIteration.Sink operation are handled specially: they are live if the
    // source is live, and removed when the source is processed.
    if (auto sinkOp = dyn_cast<NextIterationSinkOp>(op)) continue;

    // For NextIteration.Source, we just check that the source does not have any
    // other user than the sink.
    if (auto sourceOp = dyn_cast<NextIterationSourceOp>(op)) {
      Operation *sink = sourceOp.GetSink().getOperation();
      if (llvm::any_of(sourceOp.getResults(), [sink](Value *result) {
            return llvm::any_of(result->getUsers(), [sink](Operation *user) {
              return user != sink;
            });
          }))
        continue;

      // No other users than the sink, erase the pair!
      sink->erase();
      sourceOp.erase();
      continue;
    }

    // General case.
    if (op.use_empty()) op.erase();
  }
}

namespace {

// This transformation pass prunes a TF graph eliminating dead-nodes.
struct GraphPruning : public FunctionPass<GraphPruning> {
  void runOnFunction() override {
    getFunction().walk<tf_executor::GraphOp>(
        [](tf_executor::GraphOp graph) { prune_graph(graph); });
  }
};

}  // namespace

FunctionPassBase *CreateTFExecutorGraphPruningPass() {
  return new GraphPruning();
}

static PassRegistration<GraphPruning> pass(
    "tf-executor-graph-pruning", "Prune a TensorFlow Graph from dead nodes.");

}  // namespace tf_executor
}  // namespace mlir
