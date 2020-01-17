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

#include "llvm/Support/Debug.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"

#define DEBUG_TYPE "tf-functional-to-executor"

namespace mlir {

namespace {
// This pass converts mlir functions consisting of mlir ops into a tf_executor
// dialect as a single island.
// Result like so:
//   func @my_fn(%argi...) -> (result_t) {
//     %results:[[n_args]] = tf_executor.graph {
//        %island_results:[[nargs + 1]] = tf_executor.island {
//          ... original ops ...
//          tf_executor.yield %results...
//        }
//        tf_executor.fetch %island_results#...
//      }
//      return %graph_results#...
//    }
struct FunctionalToExecutorDialectConversion
    : public FunctionPass<FunctionalToExecutorDialectConversion> {
  void runOnFunction() override;
};
}  // end anonymous namespace

void FunctionalToExecutorDialectConversion::runOnFunction() {
  if (getFunction().getBlocks().size() != 1) {
    LLVM_DEBUG(llvm::dbgs() << "Expect single block function, skip conversion "
                               "to tf_executor dialect\n");
    return;
  }
  auto loc = getFunction().getLoc();
  mlir::Block& body = getFunction().getBody().front();
  // Find region of interest and ReturnOp.
  auto copy_range = body.without_terminator();
  if (copy_range.begin() != copy_range.end() &&
      std::next(copy_range.begin()) == copy_range.end() &&
      isa<tf_executor::GraphOp>(*copy_range.begin())) {
    // Already a graph.
    return;
  }

  auto return_op = dyn_cast<ReturnOp>(body.getTerminator());
  if (!return_op) {
    LLVM_DEBUG(llvm::dbgs() << "Expect function to end with return\n");
    return;
  }
  // Build GraphOp.
  OpBuilder builder(&body, body.begin());
  auto graph_op = builder.create<tf_executor::GraphOp>(
      loc, getFunction().getType().getResults());
  graph_op.body().push_back(new Block);
  builder.setInsertionPointToEnd(&graph_op.GetBody());
  auto island = builder.create<tf_executor::IslandOp>(
      loc, getFunction().getType().getResults(),
      tf_executor::ControlType::get(&getContext()), ArrayRef<Value>());
  // Create Fetch.
  ValueRange to_fetch = island.getResults();
  if (to_fetch.size() != 1) {
    // Drop control result for fetch.
    to_fetch = to_fetch.drop_back();
  }
  builder.create<tf_executor::FetchOp>(loc, to_fetch);
  // Build Island.
  island.body().push_back(new Block);
  island.body().front().getOperations().splice(
      island.body().front().begin(), body.getOperations(), copy_range.begin(),
      copy_range.end());
  builder.setInsertionPointToEnd(&island.body().front());
  builder.create<tf_executor::YieldOp>(loc, return_op.getOperands());
  for (auto item : llvm::enumerate(graph_op.getResults())) {
    return_op.setOperand(item.index(), item.value());
  }
}

std::unique_ptr<OpPassBase<FuncOp>>
CreateFunctionalToExecutorDialectConversionPass() {
  return std::make_unique<FunctionalToExecutorDialectConversion>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::FunctionalToExecutorDialectConversion> pass(
    "tf-functional-to-executor-conversion",
    "Transform from func op to TF executor dialect.");
