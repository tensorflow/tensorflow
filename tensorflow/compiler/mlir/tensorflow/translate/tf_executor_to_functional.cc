/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Function.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

namespace mlir {

namespace {
// This pass lifts tf_executor.island inner ops from a tf_executor.graph that
// contains only tf_executor.island ops.
//
// e.g.
//   func @my_fn(%arg0, %arg1) -> (...) {
//     %graph_results:2 = tf_executor.graph {
//       %island_0_result, %island_0_control = tf_executor.island {
//         %a = tf.opA(%arg0)
//         tf_executor.yield %a
//       }
//       %island_1_result, %island_1_control = tf_executor.island {
//         %b = tf.opB(%arg1, %island_0_result)
//         tf_executor.yield %b
//       }
//       tf_executor.fetch %island_0_result, %island_1_result
//     }
//     return %graph_results#0, %graph_results#1
//   }
//
// will be transformed into:
//   func @my_fn(%arg0, %arg1) -> (...) {
//     %a = tf.opA(%arg0)
//     %b = tf.opB(%arg1, %a)
//     return %a, %b
//   }

struct ExecutorDialectToFunctionalConversion
    : public PassWrapper<ExecutorDialectToFunctionalConversion, FunctionPass> {
  void runOnFunction() override;
};

// Extracts inner ops of tf_executor.island ops in a tf_executor.graph, in the
// order of ops in tf_executor.graph.
LogicalResult LiftIslandOpInnerOpsFromGraph(tf_executor::GraphOp graph) {
  auto graph_position = graph.getOperation()->getIterator();
  Block* parent_block = graph.getOperation()->getBlock();
  for (Operation& op : graph.GetBody().without_terminator()) {
    auto island_op = llvm::dyn_cast<tf_executor::IslandOp>(op);
    if (!island_op)
      return op.emitOpError()
             << "is not supported for lifting out of tf_executor.graph, "
                "expected tf_executor.island";

    // Move inner ops in island to before the outer graph.
    auto& island_body = island_op.GetBody().getOperations();
    parent_block->getOperations().splice(graph_position, island_body,
                                         island_body.begin(),
                                         std::prev(island_body.end()));
    // Forward island fetches (tf_executor.yield operands) to island op result
    // uses.
    for (auto result :
         llvm::zip(island_op.outputs(), island_op.GetYield().fetches()))
      std::get<0>(result).replaceAllUsesWith(std::get<1>(result));
  }

  // Forward graph fetches (tf_executor.fetch operands) to graph op result uses.
  for (auto result : llvm::zip(graph.results(), graph.GetFetch().fetches()))
    std::get<0>(result).replaceAllUsesWith(std::get<1>(result));

  graph.erase();
  return success();
}

void ExecutorDialectToFunctionalConversion::runOnFunction() {
  auto result = getFunction().walk([](tf_executor::GraphOp graph) {
    if (failed(LiftIslandOpInnerOpsFromGraph(graph)))
      return WalkResult::interrupt();

    return WalkResult::advance();
  });
  if (result.wasInterrupted()) signalPassFailure();
}
}  // end anonymous namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateExecutorDialectToFunctionalConversionPass() {
  return std::make_unique<ExecutorDialectToFunctionalConversion>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::ExecutorDialectToFunctionalConversion> pass(
    "tf-executor-to-functional-conversion",
    "Transform from the TF executor dialect (tf_executor.graph containing only "
    "tf_executor.island ops) to func op.");
