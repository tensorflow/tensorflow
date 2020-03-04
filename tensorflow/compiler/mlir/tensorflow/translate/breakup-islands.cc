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
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Attributes.h"  // TF:llvm-project
#include "mlir/IR/Builders.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/IR/Value.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "mlir/Pass/PassRegistry.h"  // TF:llvm-project
#include "mlir/Support/STLExtras.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

// This pass is used in preparation for Graph export.
// The GraphDef exporter expects each op to be in its own island.
// This pass puts the IR in that form.
//
// We do this as an IR->IR transform to keep the Graph exporter as simple as
// possible.

namespace mlir {

namespace {

struct BreakUpIslands : FunctionPass<BreakUpIslands> {
  void runOnFunction() final;

  void BreakUpIsland(tf_executor::IslandOp island_op,
                     const TF::SideEffectAnalysis& side_effect_analysis,
                     llvm::DenseMap<Operation*, llvm::SmallVector<Value, 4>>*
                         new_control_inputs);
};

void BreakUpIslands::runOnFunction() {
  auto graph_op_range = getFunction().getBody().front().without_terminator();
  tf_executor::GraphOp graph_op;
  if (graph_op_range.begin() != graph_op_range.end() &&
      std::next(graph_op_range.begin()) == graph_op_range.end()) {
    graph_op = dyn_cast<tf_executor::GraphOp>(
        getOperation().getBody().front().front());
  }
  if (!graph_op) {
    getOperation().emitError("expected function to contain only a graph_op");
    signalPassFailure();
    return;
  }

  // New control inputs to be added. For an operation x, new_control_inputs[x]
  // contains all control inputs that need to be added to x as operands.
  llvm::DenseMap<Operation*, llvm::SmallVector<Value, 4>> new_control_inputs;
  auto& side_effect_analysis = getAnalysis<TF::SideEffectAnalysis>();
  // Iterate in reverse order to avoid invalidating Operation* stored in
  // new_control_inputs.
  for (auto& item :
       llvm::make_early_inc_range(llvm::reverse(graph_op.GetBody()))) {
    if (auto island = dyn_cast<tf_executor::IslandOp>(&item)) {
      BreakUpIsland(island, side_effect_analysis, &new_control_inputs);
    }
  }
  OpBuilder builder(getOperation());

  // For every op, add new control inputs in reverse order so that the ops don't
  // get invalidated.
  llvm::SmallVector<Value, 8> operands;
  llvm::SmallPtrSet<Operation*, 4> defining_ops;
  llvm::SmallVector<Type, 4> types;
  for (auto& item :
       llvm::make_early_inc_range(llvm::reverse(graph_op.GetBody()))) {
    auto it = new_control_inputs.find(&item);
    if (it == new_control_inputs.end()) continue;
    auto& new_control_inputs_for_item = it->second;
    builder.setInsertionPoint(&item);
    OperationState state(item.getLoc(), item.getName());
    types.assign(item.result_type_begin(), item.result_type_end());
    state.addTypes(types);
    for (Region& region : item.getRegions()) {
      state.addRegion()->takeBody(region);
    }
    // Assign existing operands for item.
    operands.assign(item.operand_begin(), item.operand_end());

    // Collect defining ops for existing operands.
    defining_ops.clear();
    for (Value operand : operands) {
      defining_ops.insert(operand.getDefiningOp());
    }
    for (Value new_control_input : llvm::reverse(new_control_inputs_for_item)) {
      // Add new control input if its defining op is not already a defining
      // op for some other operand. Update defining_ops.
      if (defining_ops.insert(new_control_input.getDefiningOp()).second) {
        operands.push_back(new_control_input);
      }
    }
    state.addOperands(operands);
    Operation* new_op = builder.createOperation(state);
    item.replaceAllUsesWith(new_op);
    new_op->setAttrs(item.getAttrList());
    item.erase();
  }
}

// Populates an empty IslandOp and with a NoOp or Identity/IdentityN depending
// on if there are any data results.
void PopulateEmptyIsland(tf_executor::IslandOp island) {
  OpBuilder builder(&island.GetBody(), island.GetBody().begin());
  tf_executor::YieldOp yield = island.GetYield();
  if (yield.getNumOperands() == 0) {
    builder.create<TF::NoOp>(island.getLoc(), llvm::ArrayRef<mlir::Type>{},
                             llvm::ArrayRef<mlir::Value>{},
                             llvm::ArrayRef<mlir::NamedAttribute>{});
  } else if (yield.getNumOperands() == 1) {
    Value operand = yield.getOperand(0);
    auto identity = builder.create<TF::IdentityOp>(island.getLoc(),
                                                   operand.getType(), operand);
    yield.setOperand(0, identity.output());
  } else {
    auto types = llvm::to_vector<4>(yield.getOperandTypes());
    auto identity_n = builder.create<TF::IdentityNOp>(island.getLoc(), types,
                                                      yield.getOperands());
    for (auto it : llvm::enumerate(identity_n.getResults()))
      yield.setOperand(it.index(), it.value());
  }
}

// Helper that creates an island. If `sub_op` is not nullptr, it will be moved
// to the island. Otherwise a NoOp will be added to the island.
tf_executor::IslandOp CreateIsland(ArrayRef<Type> result_types,
                                   ArrayRef<Value> control_inputs,
                                   const tf_executor::ControlType& control_type,
                                   const Location& loc, Operation* sub_op,
                                   tf_executor::IslandOp original_island) {
  OpBuilder builder(original_island);
  auto island = builder.create<tf_executor::IslandOp>(
      loc, result_types, control_type, control_inputs);
  island.body().push_back(new Block);
  Block* block = &island.body().back();
  OpBuilder island_builder(original_island);
  island_builder.setInsertionPointToEnd(block);
  if (sub_op) {
    sub_op->replaceAllUsesWith(island.outputs());
    sub_op->moveBefore(block, block->begin());
    island_builder.create<tf_executor::YieldOp>(loc, sub_op->getResults());
  } else {
    island_builder.create<TF::NoOp>(
        island.getLoc(), llvm::ArrayRef<mlir::Type>{},
        llvm::ArrayRef<mlir::Value>{}, llvm::ArrayRef<mlir::NamedAttribute>{});
    island_builder.create<tf_executor::YieldOp>(loc, ArrayRef<Value>{});
  }
  return island;
}

// A struct contains the operations in an island that do not have incoming or
// outgoing dependencies.
struct IslandSourcesAndSinks {
  // Sub-ops that do not depend on other sub-ops in the island.
  llvm::SmallPtrSet<Operation*, 4> sources;
  // Sub-ops that do not have other sub-ops in the island depending on them
  // (excluding yield).
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
      auto defining_op = operand.getDefiningOp();
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
    tf_executor::IslandOp island_op,
    const TF::SideEffectAnalysis& side_effect_analysis,
    llvm::DenseMap<Operation*, llvm::SmallVector<Value, 4>>*
        new_control_inputs) {
  auto island_body = island_op.GetBody().without_terminator();
  // Populate islands that are empty (only yield).
  if (island_body.empty()) {
    PopulateEmptyIsland(island_op);
    return;
  }

  // Skip islands that are already only a single op.
  if (has_single_element(island_body)) return;

  auto control_type = tf_executor::ControlType::get(&getContext());
  auto island_control_inputs = llvm::to_vector<4>(island_op.controlInputs());
  // Add control dependencies for yields of values defined by other islands to
  // the island that defines that fetched value.
  for (auto fetch : island_op.GetYield().fetches()) {
    if (!fetch.getDefiningOp()) {
      // Skip, because there is no op to add control to (eg: function args).
      continue;
    } else if (fetch.getDefiningOp()->getParentOp() == island_op) {
      // Skip, because it is the same island.
      continue;
    } else if (auto other_island_op = llvm::dyn_cast<tf_executor::IslandOp>(
                   fetch.getDefiningOp())) {
      island_control_inputs.push_back(other_island_op.control());
    } else {
      // TODO(parkers): Any defining op that has a control output can be handled
      // just like an island.
      fetch.getDefiningOp()->emitError("fetching non-island as dependency");
      return signalPassFailure();
    }
  }
  // If there are multiple control inputs, create an empty island to group them.
  if (island_control_inputs.size() > 1) {
    auto new_island = CreateIsland({}, island_control_inputs, control_type,
                                   island_op.getLoc(), nullptr, island_op);
    island_control_inputs.clear();
    island_control_inputs.push_back(new_island.control());
  }
  // Find sources and sinks inside the original island.
  auto sources_and_sinks =
      FindSourcesAndSinksInIsland(island_op, side_effect_analysis);
  // The corresponding control output of the new island created for each sub-op.
  llvm::SmallDenseMap<Operation*, Value, 8> new_control_for_sub_ops;
  // Control outputs of newly created islands that are sinks.
  llvm::SmallVector<Value, 8> sink_island_controls;
  // For each operation in the island, construct a new island to wrap the op,
  // yield all the results, and replace all the usages with the results of the
  // new island.
  for (auto& sub_op : llvm::make_early_inc_range(island_body)) {
    const auto predecessors =
        side_effect_analysis.DirectControlPredecessors(&sub_op);
    // Get the controls from the predecessors.
    llvm::SmallVector<Value, 4> predecessor_controls;
    predecessor_controls.reserve(predecessors.size());
    for (auto predecessor : predecessors) {
      predecessor_controls.push_back(new_control_for_sub_ops[predecessor]);
    }
    // If sub_op is a source, use island_control_inputs, because that's required
    // by inter-islands dependencies; otherwise, we do not need to include
    // island_control_inputs, since they must have been tracked by the (direct
    // or indirect) control predecessors or operands.
    ArrayRef<Value> control = sources_and_sinks.sources.count(&sub_op) > 0
                                  ? island_control_inputs
                                  : predecessor_controls;
    auto new_island =
        CreateIsland(llvm::to_vector<4>(sub_op.getResultTypes()), control,
                     control_type, sub_op.getLoc(), &sub_op, island_op);
    new_control_for_sub_ops[&sub_op] = new_island.control();
    if (sources_and_sinks.sinks.count(&sub_op)) {
      sink_island_controls.push_back(new_island.control());
    }
  }
  // Create control outputs for the sinks.
  assert(!sink_island_controls.empty());
  // If there are multiple control outputs, create an empty island to group
  // them.
  if (sink_island_controls.size() > 1) {
    auto new_island = CreateIsland({}, sink_island_controls, control_type,
                                   island_op.getLoc(), nullptr, island_op);
    sink_island_controls.clear();
    sink_island_controls.push_back(new_island.control());
  }
  assert(sink_island_controls.size() == 1);
  auto& sink_island_control = sink_island_controls[0];
  island_op.control().replaceAllUsesWith(sink_island_control);
  // All existing outputs need to add sink_island_control as control input.
  // GraphOp, YieldOp and NextIterationSourceOp don't have control inputs so
  // exclude them below.
  for (Value out : island_op.outputs()) {
    for (auto& use : out.getUses()) {
      Operation* owner = use.getOwner();
      if (auto other_island_op =
              llvm::dyn_cast<tf_executor::IslandOp>(owner->getParentOp())) {
        (*new_control_inputs)[other_island_op].push_back(sink_island_control);
      } else if (owner->getDialect() == island_op.getDialect() &&
                 !llvm::isa<tf_executor::GraphOp>(owner) &&
                 !llvm::isa<tf_executor::YieldOp>(owner) &&
                 !llvm::isa<tf_executor::NextIterationSourceOp>(owner)) {
        (*new_control_inputs)[owner].push_back(sink_island_control);
      } else {
        owner->emitOpError("adding control dependency not supported");
        return signalPassFailure();
      }
    }
  }
  for (auto item :
       llvm::zip(island_op.outputs(), island_op.GetYield().fetches()))
    std::get<0>(item).replaceAllUsesWith(std::get<1>(item));
  island_op.erase();
}

}  // namespace

std::unique_ptr<OpPassBase<FuncOp>> CreateBreakUpIslandsPass() {
  return std::make_unique<BreakUpIslands>();
}

}  // namespace mlir

static mlir::PassRegistration<mlir::BreakUpIslands> pass(
    "tf-executor-break-up-islands",
    "Transform from TF control dialect to TF executor dialect.");
