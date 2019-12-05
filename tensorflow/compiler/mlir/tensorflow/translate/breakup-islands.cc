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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "mlir/Support/STLExtras.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
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
                     const TF::SideEffectAnalysis& side_effect_analysis,
                     llvm::DenseMap<Operation*, llvm::SmallVector<Value*, 4>>*
                         new_control_edges);
};

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
  auto& side_effect_analysis = getAnalysis<TF::SideEffectAnalysis>();
  // Iterate in reverse order to avoid invalidating Operation* stored in
  // new_control_edges.
  for (auto& item :
       llvm::make_early_inc_range(llvm::reverse(graph_op.GetBody()))) {
    if (auto island = dyn_cast<tf_executor::IslandOp>(&item)) {
      BreakUpIsland(island, side_effect_analysis, &new_control_edges);
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

// Helper that creates an island. If `sub_op` is not nullptr, it will be moved
// to the island.
tf_executor::IslandOp CreateIsland(ArrayRef<Type> result_types,
                                   ArrayRef<Value*> control_inputs,
                                   const tf_executor::ControlType& control_type,
                                   const Location& loc, Operation* sub_op,
                                   tf_executor::IslandOp original_island) {
  OpBuilder builder(original_island);
  auto island = builder.create<tf_executor::IslandOp>(
      loc, result_types, control_type, control_inputs);
  island.body().push_back(new Block);
  Block* block = &island.body().back();
  if (sub_op) {
    sub_op->replaceAllUsesWith(island.outputs());
    sub_op->moveBefore(block, block->begin());
  }
  OpBuilder island_builder(original_island);
  island_builder.setInsertionPointToEnd(block);
  if (sub_op) {
    island_builder.create<tf_executor::YieldOp>(
        loc, llvm::to_vector<4>(sub_op->getResults()));
  } else {
    island_builder.create<tf_executor::YieldOp>(loc, ArrayRef<Value*>{});
  }
  return island;
}

// A struct contains the operations in an island that do not have incoming or
// outgoing dependencies.
struct IslandSourcesAndSinks {
  // Sub-ops that do not depend on other ops in the island.
  llvm::SmallPtrSet<Operation*, 4> sources;
  // Sub-ops that do not have other sub-ops island depending on them (excluding
  // yield).
  llvm::SmallPtrSet<Operation*, 4> sinks;
};

// Finds IslandSourcesAndSinks for an unmodified island.
IslandSourcesAndSinks FindSourcesAndSinksInIsland(
    tf_executor::IslandOp island,
    const TF::SideEffectAnalysis& side_effect_analysis) {
  IslandSourcesAndSinks result;
  auto island_body = island.GetBody().without_terminator();
  for (Operation& sub_op : island_body) {
    auto predecessors = side_effect_analysis.DirectControlPredecessors(&sub_op);
    result.sinks.insert(&sub_op);
    // Remove predecessor from sinks.
    for (auto predecessor : predecessors) result.sinks.erase(predecessor);
    bool has_in_island_operands = false;
    for (auto operand : sub_op.getOperands()) {
      auto defining_op = operand->getDefiningOp();
      if (!defining_op || defining_op->getParentOp() != island) continue;
      // Remove operands from sinks.
      result.sinks.erase(defining_op);
      has_in_island_operands = true;
    }
    if (predecessors.empty() && !has_in_island_operands) {
      result.sources.insert(&sub_op);
    }
  }
  return result;
}

// Converts a single island into multiple islands (one for each op). The islands
// are chained together by control flow values.
void BreakUpIslands::BreakUpIsland(
    tf_executor::IslandOp op,
    const TF::SideEffectAnalysis& side_effect_analysis,
    llvm::DenseMap<Operation*, llvm::SmallVector<Value*, 4>>*
        new_control_edges) {
  auto island_body = op.GetBody().without_terminator();
  // Skip islands that are already only a single op.
  // Skip islands that are empty (only yield).
  if (island_body.empty() || has_single_element(island_body)) return;
  auto control_type = tf_executor::ControlType::get(&getContext());
  auto island_control_inputs = llvm::to_vector<4>(op.controlInputs());
  // Add control dependencies for yields of values defined by other islands to
  // the island that defines that fetched value.
  for (auto* fetch : op.GetYield().fetches()) {
    // Ok, because there is no op to add control to (eg: function args).
    if (!fetch->getDefiningOp()) continue;
    if (fetch->getDefiningOp()->getParentOp() == op) {
      // OK, because it is the same island.
    } else if (auto island_op = llvm::dyn_cast<tf_executor::IslandOp>(
                   fetch->getDefiningOp())) {
      island_control_inputs.push_back(island_op.control());
    } else {
      // TODO(parkers): Any defining op that has a control output can be handled
      // just like an island.
      fetch->getDefiningOp()->emitError("Fetching non-island as dependency.");
      return signalPassFailure();
    }
  }
  // If there are multiple control inputs, create an empty island to group them.
  if (island_control_inputs.size() > 1) {
    auto island = CreateIsland({}, island_control_inputs, control_type,
                               op.getLoc(), nullptr, op);
    island_control_inputs.clear();
    island_control_inputs.push_back(island.control());
  }
  // Find sources and sinks inside the original island.
  auto sources_and_sinks =
      FindSourcesAndSinksInIsland(op, side_effect_analysis);
  // The corresponding control output of the new island created for each sub-op.
  llvm::SmallDenseMap<Operation*, Value*, 8> new_control_for_sub_ops;
  // Control outputs of newly created islands that are sinks.
  llvm::SmallVector<Value*, 8> sink_island_controls;
  // For each operation in the island, construct a new island to wrap the op,
  // yield all the results, and replace all the usages with the results of the
  // new island.
  for (auto& sub_op : llvm::make_early_inc_range(island_body)) {
    const auto predecessors =
        side_effect_analysis.DirectControlPredecessors(&sub_op);
    // Get the controls from the predecessors.
    llvm::SmallVector<Value*, 4> predecessors_control;
    predecessors_control.reserve(predecessors.size());
    for (auto predecessor : predecessors) {
      predecessors_control.push_back(new_control_for_sub_ops[predecessor]);
    }
    // If sub_op is a source, use island_control_inputs, because that's required
    // by inter-islands dependencies; otherwise, we do not need to include
    // island_control_inputs, since they must have been tracked by the (direct
    // or indirect) control predecessors or operands.
    ArrayRef<Value*> control = sources_and_sinks.sources.count(&sub_op) > 0
                                   ? island_control_inputs
                                   : predecessors_control;
    auto island =
        CreateIsland(llvm::to_vector<4>(sub_op.getResultTypes()), control,
                     control_type, sub_op.getLoc(), &sub_op, op);
    new_control_for_sub_ops[&sub_op] = island.control();
    if (sources_and_sinks.sinks.count(&sub_op)) {
      sink_island_controls.push_back(island.control());
    }
  }
  // Create output controls for the sinks.
  assert(!sink_island_controls.empty());
  // If there are multiple output controls, create an empty island to group
  // them.
  if (sink_island_controls.size() > 1) {
    auto island = CreateIsland({}, sink_island_controls, control_type,
                               op.getLoc(), nullptr, op);
    sink_island_controls.clear();
    sink_island_controls.push_back(island.control());
  }
  assert(sink_island_controls.size() == 1);
  op.control()->replaceAllUsesWith(sink_island_controls[0]);
  // All existing outputs need to add a control flow edge from
  // sink_island_controls[0].
  for (Value* out : op.outputs()) {
    for (auto& use : out->getUses()) {
      Operation* owner = use.getOwner();
      if (auto island_op =
              llvm::dyn_cast<tf_executor::IslandOp>(owner->getParentOp())) {
        (*new_control_edges)[island_op].push_back(sink_island_controls[0]);
      } else if (llvm::isa<tf_executor::FetchOp>(owner) ||
                 llvm::isa<tf_executor::MergeOp>(owner) ||
                 llvm::isa<tf_executor::SwitchOp>(owner)) {
        (*new_control_edges)[owner].push_back(sink_island_controls[0]);
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

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateBreakUpIslandsPass() {
  return std::make_unique<BreakUpIslands>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::BreakUpIslands> pass(
    "tf-executor-break-up-islands",
    "Transform from TF control dialect to TF executor dialect.");
