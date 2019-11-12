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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/STLExtras.h"  // TF:local_config_mlir
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

  void BreakUpIsland(tf_executor::IslandOp op,
                     llvm::DenseMap<Operation*, llvm::SmallVector<Value*, 4>>*
                         new_control_edges);
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

  // Map from the users of the existing islands to the list of control
  // edges that need to be added.
  llvm::DenseMap<Operation*, llvm::SmallVector<Value*, 4>> new_control_edges;
  // Iterate in reverse order to avoid invalidating Operation* stored in
  // new_control_edges.
  for (auto& item :
       llvm::make_early_inc_range(llvm::reverse(graph_op.GetBody()))) {
    if (auto island = dyn_cast<tf_executor::IslandOp>(&item)) {
      BreakUpIsland(island, &new_control_edges);
    }
  }
  OpBuilder builder(getOperation());

  // Apply edge additions in reverse order so that the ops don't get
  // invalidated.
  llvm::SmallVector<Value*, 8> edges;
  llvm::SmallPtrSet<Operation*, 4> dups;
  llvm::SmallVector<Type, 4> types;
  for (auto& item :
       llvm::make_early_inc_range(llvm::reverse(graph_op.GetBody()))) {
    auto it = new_control_edges.find(&item);
    if (it == new_control_edges.end()) continue;
    auto& edge = *it;
    builder.setInsertionPoint(&item);
    OperationState state(item.getLoc(), item.getName());
    types.assign(item.result_type_begin(), item.result_type_end());
    state.addTypes(types);
    for (Region& region : item.getRegions()) {
      state.addRegion()->takeBody(region);
    }
    edges.assign(item.operand_begin(), item.operand_end());
    dups.clear();

    for (Value* input : edges) {
      dups.insert(input->getDefiningOp());
    }
    // Insert new control edges removing duplicates.
    for (Value* value : llvm::reverse(edge.second)) {
      if (dups.insert(value->getDefiningOp()).second) edges.push_back(value);
    }
    state.addOperands(edges);
    Operation* new_op = builder.createOperation(state);
    item.replaceAllUsesWith(new_op);
    new_op->setAttrs(item.getAttrList());
    item.erase();
  }
}

// Converts a single island into multiple islands (one for each op). The islands
// are chained together by control flow values.
void BreakUpIslands::BreakUpIsland(
    tf_executor::IslandOp op,
    llvm::DenseMap<Operation*, llvm::SmallVector<Value*, 4>>*
        new_control_edges) {
  auto island_body = op.GetBody().without_terminator();
  // Skip islands that are already only a single op.
  // Skip islands that are empty (only yield).
  if (island_body.empty() || has_single_element(island_body)) return;
  OpBuilder builder(op);
  OpBuilder island_builder(op);
  auto control_type = tf_executor::ControlType::get(&getContext());
  Value* previous_island = nullptr;
  auto tmp_control_inputs = llvm::to_vector<4>(op.controlInputs());
  // Add control dependencies for yields of values defined by other islands to
  // the island that defines that fetched value.
  for (auto* fetch : op.GetYield().fetches()) {
    // Ok, because there is no op to add control to (eg: function args).
    if (!fetch->getDefiningOp()) continue;
    if (fetch->getDefiningOp()->getParentOp() == op) {
      // OK, because it is the same island.
    } else if (auto island_op = llvm::dyn_cast<tf_executor::IslandOp>(
                   fetch->getDefiningOp())) {
      tmp_control_inputs.push_back(island_op.control());
    } else {
      // TODO(parkers): Any defining op that has a control output can be handled
      // just like an island.
      fetch->getDefiningOp()->emitError("Fetching non-island as dependency.");
      return signalPassFailure();
    }
  }
  ArrayRef<Value*> previous_control = tmp_control_inputs;
  // For each operation in the island, construct a new island to wrap the op,
  // yield all the results, and replace all the usages with the results of the
  // new island.
  for (Operation& sub_op : llvm::make_early_inc_range(island_body)) {
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
  op.control()->replaceAllUsesWith(previous_island);
  // All existing outputs need to add a control flow edge to the
  // previous_island.
  for (Value* out : op.outputs()) {
    for (auto& use : out->getUses()) {
      Operation* owner = use.getOwner();
      if (auto island_op =
              llvm::dyn_cast<tf_executor::IslandOp>(owner->getParentOp())) {
        (*new_control_edges)[island_op].push_back(previous_island);
      } else if (llvm::isa<tf_executor::FetchOp>(owner) ||
                 llvm::isa<tf_executor::MergeOp>(owner) ||
                 llvm::isa<tf_executor::SwitchOp>(owner)) {
        (*new_control_edges)[owner].push_back(previous_island);
      } else {
        use.getOwner()->emitError("Adding control dependency not supported");
        return signalPassFailure();
      }
    }
  }
  for (auto item : llvm::zip(op.outputs(), op.GetYield().fetches()))
    std::get<0>(item)->replaceAllUsesWith(std::get<1>(item));
  op.erase();
}

std::unique_ptr<OpPassBase<FuncOp>> CreateBreakUpIslandsPass() {
  return std::make_unique<BreakUpIslands>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::BreakUpIslands> pass(
    "tf-executor-break-up-islands",
    "Transform from TF control dialect to TF executor dialect.");
