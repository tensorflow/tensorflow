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

#include <cstdint>
#include <memory>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// This pass is used in preparation for Graph export.
// The GraphDef exporter expects each op to be in its own island.
// This pass puts the IR in that form.
//
// We do this as an IR->IR transform to keep the Graph exporter as simple as
// possible.

namespace mlir {
namespace TF {

namespace {

#define GEN_PASS_DEF_SPLITINTOISLANDPEROPPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class SplitIntoIslandPerOpPass
    : public impl::SplitIntoIslandPerOpPassBase<SplitIntoIslandPerOpPass> {
 public:
  void runOnOperation() override;

 private:
  void SplitIsland(tf_executor::IslandOp island_op,
                   tf_executor::GraphOp graph_op);
};

void SplitIntoIslandPerOpPass::runOnOperation() {
  func::FuncOp func = getOperation();

  if (func.isExternal()) {
    // Just ignore the op if this is an external func with no body.
    return;
  }

  tf_executor::GraphOp graph_op;

  if (llvm::hasSingleElement(func.front().without_terminator())) {
    graph_op = dyn_cast<tf_executor::GraphOp>(func.front().front());
  }

  if (!graph_op) {
    func.emitError("expected function to contain only a graph_op");
    signalPassFailure();
    return;
  }

  if (!(llvm::hasSingleElement(graph_op.GetBody().without_terminator()) &&
        llvm::isa<tf_executor::IslandOp>(graph_op.GetBody().front()))) {
    graph_op.emitError(
        "expected graph op to contain only a single island_op and a single "
        "fetch_op");
    signalPassFailure();
    return;
  }

  tf_executor::IslandOp island_op =
      dyn_cast<tf_executor::IslandOp>(graph_op.GetBody().front());

  // We don't need to honor *any* control deps that are already placed on
  // islands. Drop them now in this pass - a following pass will use side
  // effect analysis to completely explain and apply correct control deps.
  island_op.getControl().dropAllUses();

  // Break up all islands by simply creating a new island wrapping each
  // individual sub op. Do not create any control dependencies between the
  // newly created islands.
  SplitIsland(island_op, graph_op);

  // None of the originally given control deps are necessary.
  tf_executor::FetchOp fetch_op = graph_op.GetFetch();
  int num_control_fetches =
      fetch_op.getNumOperands() - graph_op.getNumResults();
  if (num_control_fetches > 0) {
    fetch_op.getFetchesMutable().erase(graph_op.getNumResults(),
                                       num_control_fetches);
  }
}

// Populates an empty IslandOp and with a NoOp or Identity/IdentityN depending
// on if there are any data results.
void PopulateEmptyIsland(tf_executor::IslandOp island) {
  OpBuilder builder(&island.GetBody(), island.GetBody().begin());
  tf_executor::YieldOp yield = island.GetYield();
  if (yield.getNumOperands() == 0) {
    builder.create<TF::NoOp>(island.getLoc(), TypeRange{}, ValueRange{});
  } else if (yield.getNumOperands() == 1) {
    Value operand = yield.getOperand(0);
    auto identity = builder.create<TF::IdentityOp>(island.getLoc(),
                                                   operand.getType(), operand);
    yield.setOperand(0, identity.getOutput());
  } else {
    auto identity_n = builder.create<TF::IdentityNOp>(
        island.getLoc(), yield.getOperandTypes(), yield.getOperands());
    for (const auto& it : llvm::enumerate(identity_n.getResults()))
      yield.setOperand(it.index(), it.value());
  }
}

// Helper that creates an island that `sub_op` will be moved to.
tf_executor::IslandOp CreateIsland(TypeRange result_types,
                                   const tf_executor::ControlType& control_type,
                                   const Location& loc, Operation& sub_op,
                                   tf_executor::IslandOp original_island) {
  OpBuilder builder(original_island);
  auto island = builder.create<tf_executor::IslandOp>(
      loc, result_types, control_type, mlir::ValueRange{});
  island.getBody().push_back(new Block);
  Block* block = &island.getBody().back();
  OpBuilder island_builder(original_island);
  island_builder.setInsertionPointToEnd(block);
  sub_op.replaceAllUsesWith(island.getOutputs());
  sub_op.moveBefore(block, block->begin());
  island_builder.create<tf_executor::YieldOp>(loc, sub_op.getResults());
  return island;
}

// Converts a single island into multiple islands (one for each op).
void SplitIntoIslandPerOpPass::SplitIsland(tf_executor::IslandOp island_op,
                                           tf_executor::GraphOp graph_op) {
  auto island_body = island_op.GetBody().without_terminator();
  // Populate islands that are empty (only yield).
  if (island_body.empty()) {
    PopulateEmptyIsland(island_op);
    return;
  }

  // Skip islands that are already only a single op.
  if (island_op.WrapsSingleOp()) return;

  auto control_type = tf_executor::ControlType::get(&getContext());

  // For each operation in the island, construct a new island to wrap the op,
  // yield all the results, and replace all the usages with the results of the
  // new island.
  for (auto& sub_op : llvm::make_early_inc_range(island_body)) {
    CreateIsland(sub_op.getResultTypes(), control_type, sub_op.getLoc(), sub_op,
                 island_op);
  }

  // Ensure that consumers of the outputs of the original island now depend
  // directly on the new island wrapping the original output op instead of
  // depending on the original island's output since we're about to erase the
  // original island op.
  for (auto item :
       llvm::zip(island_op.getOutputs(), island_op.GetYield().getFetches()))
    std::get<0>(item).replaceAllUsesWith(std::get<1>(item));
  island_op.erase();
}

}  // namespace
}  // namespace TF

std::unique_ptr<OperationPass<func::FuncOp>> CreateSplitIntoIslandPerOpPass() {
  return std::make_unique<TF::SplitIntoIslandPerOpPass>();
}

}  // namespace mlir
