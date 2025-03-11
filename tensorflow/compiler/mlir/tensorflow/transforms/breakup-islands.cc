/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cassert>
#include <memory>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
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

class BreakUpIslands : public TF::PerFunctionAggregateAnalysisConsumerPass<
                           BreakUpIslands, TF::SideEffectAnalysis> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<tf_executor::TensorFlowExecutorDialect>();
  }

 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BreakUpIslands)

  StringRef getArgument() const final { return "tf-executor-break-up-islands"; }

  StringRef getDescription() const final {
    return "Transform from TF control dialect to TF executor dialect.";
  }

  void runOnFunction(func::FuncOp func,
                     const TF::SideEffectAnalysis::Info& side_effect_analysis);

  void BreakUpIsland(tf_executor::IslandOp island_op,
                     const TF::SideEffectAnalysis::Info& side_effect_analysis,
                     llvm::DenseMap<Operation*, llvm::SmallVector<Value, 4>>*
                         new_control_inputs);
};

// Returns true if the operation is a stateful If, Case, or While op.
bool IsStatefulFunctionalControlFlowOp(Operation* op) {
  if (!isa<TF::IfOp, TF::CaseOp, TF::WhileOp>(op)) {
    return false;
  }

  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless")) {
    return !is_stateless.getValue();
  }
  return false;
}

// Add control dependencies from stateful control-flow ops to graph fetch op.
// This is needed to avoid that such control-flow ops get pruned because of a
// bug in common runtime (see b/185483669).
void AddStatefulControlFlowDependencies(tf_executor::GraphOp graph_op) {
  llvm::SmallDenseSet<Value, 8> graph_fetches;
  for (Value value : graph_op.GetFetch().getFetches()) {
    graph_fetches.insert(value);
  }
  for (Operation& op : graph_op.GetBody().without_terminator()) {
    auto island = dyn_cast<tf_executor::IslandOp>(&op);
    if (!island) continue;
    if (!island.WrapsSingleOp()) continue;
    Operation& wrapped_op = island.GetBody().front();
    if (!IsStatefulFunctionalControlFlowOp(&wrapped_op)) continue;
    if (graph_fetches.contains(island.getControl())) continue;

    graph_op.GetFetch().getFetchesMutable().append(island.getControl());
  }
}

void BreakUpIslands::runOnFunction(
    func::FuncOp func,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  auto graph_op_range = func.front().without_terminator();
  tf_executor::GraphOp graph_op;

  if (llvm::hasSingleElement(graph_op_range))
    graph_op = dyn_cast<tf_executor::GraphOp>(func.front().front());

  if (!graph_op) {
    func.emitError("expected function to contain only a graph_op");
    signalPassFailure();
    return;
  }

  // New control inputs to be added. For an operation x, new_control_inputs[x]
  // contains all control inputs that need to be added to x as operands.
  llvm::DenseMap<Operation*, llvm::SmallVector<Value, 4>> new_control_inputs;
  // Iterate in reverse order to avoid invalidating Operation* stored in
  // new_control_inputs.
  for (auto& item :
       llvm::make_early_inc_range(llvm::reverse(graph_op.GetBody()))) {
    if (auto island = dyn_cast<tf_executor::IslandOp>(&item)) {
      BreakUpIsland(island, side_effect_analysis, &new_control_inputs);
    }
  }
  OpBuilder builder(func);

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
    Operation* new_op = builder.create(state);
    item.replaceAllUsesWith(new_op);
    new_op->setAttrs(item.getAttrDictionary());
    item.erase();
  }
  AddStatefulControlFlowDependencies(graph_op);
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
    for (auto it : llvm::enumerate(identity_n.getResults()))
      yield.setOperand(it.index(), it.value());
  }
}

// Helper that creates an island. If `sub_op` is not nullptr, it will be moved
// to the island. Otherwise a NoOp will be added to the island.
tf_executor::IslandOp CreateIsland(TypeRange result_types,
                                   ValueRange control_inputs,
                                   const tf_executor::ControlType& control_type,
                                   const Location& loc, Operation* sub_op,
                                   tf_executor::IslandOp original_island) {
  OpBuilder builder(original_island);
  auto island = builder.create<tf_executor::IslandOp>(
      loc, result_types, control_type, control_inputs);
  island.getBody().push_back(new Block);
  Block* block = &island.getBody().back();
  OpBuilder island_builder(original_island);
  island_builder.setInsertionPointToEnd(block);
  if (sub_op) {
    sub_op->replaceAllUsesWith(island.getOutputs());
    sub_op->moveBefore(block, block->begin());
    island_builder.create<tf_executor::YieldOp>(loc, sub_op->getResults());
  } else {
    island_builder.create<TF::NoOp>(island.getLoc(), TypeRange{}, ValueRange{});
    island_builder.create<tf_executor::YieldOp>(loc, ValueRange{});
  }
  return island;
}

// A struct that contains the operations in an island that need explicit control
// dependencies added going into and out of the island to capture inter-island
// dependencies properly.
struct IslandSourcesAndSinks {
  // Sub-ops that need a control dependency going into the island. This includes
  // sub-ops that do not depend on other sub-ops in the island and functional
  // control ops (e.g. if, while, case) with side effects that must not take
  // effect before the previous island is finished executing.
  llvm::SmallPtrSet<Operation*, 4> sources;

  // Sub-ops that need a control dependency going out of the island. This
  // includes sub-ops that do not have other sub-ops in the island depending on
  // them (excluding yield) and functional control ops (e.g. if, while, case)
  // with side effects that must take effect before the next island starts
  // executing.
  llvm::SmallPtrSet<Operation*, 4> sinks;
};

// Finds IslandSourcesAndSinks for an unmodified island.
IslandSourcesAndSinks FindSourcesAndSinksInIsland(
    tf_executor::IslandOp island,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
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
      has_in_island_operands = true;

      // Remove operands from sinks.
      // We don't remove the operand if it is a stateful functional control flow
      // op to work around an issue in LowerFunctionalOpsPass where the operand
      // dependency isn't enough to ensure the side effects take place
      // (b/185483669).
      if (!IsStatefulFunctionalControlFlowOp(defining_op)) {
        result.sinks.erase(defining_op);
      }
    }
    if (predecessors.empty() && (!has_in_island_operands ||
                                 IsStatefulFunctionalControlFlowOp(&sub_op))) {
      result.sources.insert(&sub_op);
    }
  }
  return result;
}

// Converts a single island into multiple islands (one for each op). The islands
// are chained together by control flow values.
void BreakUpIslands::BreakUpIsland(
    tf_executor::IslandOp island_op,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    llvm::DenseMap<Operation*, llvm::SmallVector<Value, 4>>*
        new_control_inputs) {
  auto island_body = island_op.GetBody().without_terminator();
  // Populate islands that are empty (only yield).
  if (island_body.empty()) {
    PopulateEmptyIsland(island_op);
    return;
  }

  // Skip islands that are already only a single op.
  if (island_op.WrapsSingleOp()) return;

  auto control_type = tf_executor::ControlType::get(&getContext());
  auto island_control_inputs = llvm::to_vector<4>(island_op.getControlInputs());
  // Add control dependencies for yields of values defined by other islands to
  // the island that defines that fetched value.
  for (auto fetch : island_op.GetYield().getFetches()) {
    if (!fetch.getDefiningOp()) {
      // Skip, because there is no op to add control to (eg: function args).
      continue;
    } else if (fetch.getDefiningOp()->getParentOp() == island_op) {
      // Skip, because it is the same island.
      continue;
    } else if (auto other_island_op = llvm::dyn_cast<tf_executor::IslandOp>(
                   fetch.getDefiningOp())) {
      island_control_inputs.push_back(other_island_op.getControl());
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
    island_control_inputs.push_back(new_island.getControl());
  }
  // Find sources and sinks inside the original island.
  IslandSourcesAndSinks sources_and_sinks =
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
        CreateIsland(sub_op.getResultTypes(), control, control_type,
                     sub_op.getLoc(), &sub_op, island_op);
    new_control_for_sub_ops[&sub_op] = new_island.getControl();
    if (sources_and_sinks.sinks.count(&sub_op)) {
      sink_island_controls.push_back(new_island.getControl());
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
    sink_island_controls.push_back(new_island.getControl());
  }
  assert(sink_island_controls.size() == 1);
  auto& sink_island_control = sink_island_controls[0];
  island_op.getControl().replaceAllUsesWith(sink_island_control);
  // All existing outputs need to add sink_island_control as control input.
  // GraphOp, YieldOp and NextIterationSourceOp don't have control inputs so
  // exclude them below.
  for (Value out : island_op.getOutputs()) {
    for (auto& use : out.getUses()) {
      Operation* owner = use.getOwner();
      if (owner->getDialect() == island_op->getDialect() &&
          !llvm::isa<tf_executor::GraphOp, tf_executor::YieldOp,
                     tf_executor::NextIterationSourceOp>(owner)) {
        (*new_control_inputs)[owner].push_back(sink_island_control);
        // Note that we cannot assume that the island containing `owner` is a
        // direct parent:
        // For example, ops with regions usually don't expose values used in a
        // region to the op's interface which means that the usage of a value
        // can be 2 or more levels below an island (see b/242920486).
      } else if (auto other_island_op =
                     owner->getParentOfType<tf_executor::IslandOp>()) {
        (*new_control_inputs)[other_island_op].push_back(sink_island_control);
      } else {
        owner->emitOpError("adding control dependency not supported");
        return signalPassFailure();
      }
    }
  }
  for (auto item :
       llvm::zip(island_op.getOutputs(), island_op.GetYield().getFetches()))
    std::get<0>(item).replaceAllUsesWith(std::get<1>(item));
  island_op.erase();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateBreakUpIslandsPass() {
  return std::make_unique<BreakUpIslands>();
}

}  // namespace mlir
