/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>
#include <vector>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

namespace mlir {
namespace tf_executor {
namespace {

#define GEN_PASS_DEF_EXECUTORUPDATECONTROLDEPENDENCIESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// Note that `SetVector` provides efficient lookup and deletion as well as
// deterministic iteration order which we need here.
using OpToIslandsMap = llvm::DenseMap<Operation*, llvm::SetVector<IslandOp>>;

class UpdateControlDependenciesPass
    : public impl::ExecutorUpdateControlDependenciesPassBase<
          UpdateControlDependenciesPass> {
 public:
  void runOnOperation() override;
};

// Returns true iff the islands are guaranteed to have different devices
// assigned.
bool HaveDifferentDevices(IslandOp first_island, IslandOp second_island) {
  Operation& first_op = first_island.GetBody().front();
  Operation& second_op = second_island.GetBody().front();
  llvm::SmallVector<tensorflow::DeviceNameUtils::ParsedName, 2> parsed_names;

  for (Operation* op : {&first_op, &second_op}) {
    auto device_attr = op->getAttrOfType<StringAttr>(tensorflow::kDeviceAttr);
    // For empty device we can't guarantee that devices are different.
    if (!device_attr || device_attr.getValue().empty()) return false;

    tensorflow::DeviceNameUtils::ParsedName parsed_name;
    bool success = tensorflow::DeviceNameUtils::ParseFullOrLocalName(
        device_attr.getValue(), &parsed_name);
    // If parsing was not successful, then we can't guarantee that devices are
    // different.
    if (!success) return false;
    parsed_names.push_back(parsed_name);
  }
  // If device names are not compatible, then corresponding devices must be
  // different.
  return !tensorflow::DeviceNameUtils::AreCompatibleDevNames(parsed_names[0],
                                                             parsed_names[1]);
}

// Returns true iff we should ignore a dependency between both islands.
bool ShouldIgnoreDependency(IslandOp first_island, IslandOp second_island) {
  return HaveDifferentDevices(first_island, second_island);
}

// Collects direct control predecessors per op by querying side effect analysis.
//
// We only collect control predecessor that are islands, others (if any) are
// irrelevant for this pass.
void CollectDirectControlPredecessors(
    Operation* op, const TF::SideEffectAnalysis::Info& analysis_for_func,
    OpToIslandsMap& control_predecessors_map) {
  for (Operation* control_predecessor :
       analysis_for_func.DirectControlPredecessors(op)) {
    if (auto control_pred_island =
            dyn_cast<mlir::tf_executor::IslandOp>(control_predecessor)) {
      control_predecessors_map[op].insert(control_pred_island);
    }
  }
}

// Propagates control predecessors for cases where we don't want to create a
// control dependency even though side effect analysis sees a dependency.
//
// Currently, this is the case for ops with different assigned devices: It can
// happen that side effect analysis sees a dependency because the ops may use
// the same resource (which is basically a modeling issue we have to work
// around here). In such a case, we ignore the dependency, but we have to make
// sure that we don't lose any indirect dependencies we want to keep.
// For example, say side effect analysis sees dependencies A -> B -> C, and A
// and C have the same assigned device and B has a different assigned device.
// Then we want to ignore the dependencies A -> B and B -> C but keep the
// transitive dependency A -> C.
// This function updates `control_predecessors_map` such that this is always the
// case.
void PropagateControlPredecessors(
    IslandOp island, const TF::SideEffectAnalysis::Info& analysis_for_func,
    OpToIslandsMap& control_predecessors_map) {
  // Find control predecessors we want to ignore and mark them for propagation.
  llvm::SmallVector<IslandOp, 8> control_predecessors_to_propagate;
  for (IslandOp control_pred_island : control_predecessors_map[island]) {
    if (ShouldIgnoreDependency(island, control_pred_island)) {
      control_predecessors_to_propagate.push_back(control_pred_island);
    }
  }
  // For all control predecessors to propagate, remove them from island's
  // control predecessors and add them as control predecessors for all control
  // successors of island (this is to make sure we don't lose any transitive
  // dependencies).
  for (IslandOp control_pred_island : control_predecessors_to_propagate) {
    control_predecessors_map[island].remove(control_pred_island);
    for (Operation* control_successor :
         analysis_for_func.DirectControlSuccessors(island)) {
      control_predecessors_map[control_successor].insert(control_pred_island);
    }
  }
}

void UpdateAllControlDependencies(
    func::FuncOp func, const TF::SideEffectAnalysis::Info& analysis_for_func) {
  int control_inputs_added = 0;
  llvm::SmallVector<Value, 8> new_control_inputs;
  llvm::SmallVector<Operation*, 8> fetch_control_predecessors;

  OpToIslandsMap control_predecessors_map;
  auto graph_op = cast<GraphOp>(func.front().front());
  graph_op.walk([&](Operation* op) {
    if (!isa<IslandOp, FetchOp>(op)) return WalkResult::advance();
    CollectDirectControlPredecessors(op, analysis_for_func,
                                     control_predecessors_map);
    if (auto island = dyn_cast<mlir::tf_executor::IslandOp>(op)) {
      PropagateControlPredecessors(island, analysis_for_func,
                                   control_predecessors_map);
    }
    return WalkResult::advance();
  });

  graph_op.walk([&](IslandOp island) {
    // Update control inputs for island.
    for (Operation* control_predecessor : control_predecessors_map[island]) {
      if (auto control_pred_island =
              dyn_cast<mlir::tf_executor::IslandOp>(control_predecessor)) {
        new_control_inputs.push_back(control_pred_island.getControl());
      }
    }
    // None of the originally given control deps are necessary.
    island.getControlInputsMutable().clear();
    island.getControlInputsMutable().append(new_control_inputs);
    control_inputs_added += new_control_inputs.size();
    new_control_inputs.clear();
  });

  // Update control inputs for fetch op.
  FetchOp fetch_op = graph_op.GetFetch();

  // None of the originally given control deps are necessary.
  int num_control_fetches =
      fetch_op.getNumOperands() - graph_op.getNumResults();
  if (num_control_fetches > 0) {
    fetch_op.getFetchesMutable().erase(graph_op.getNumResults(),
                                       num_control_fetches);
  }
  for (Operation* control_predecessor : control_predecessors_map[fetch_op]) {
    if (auto control_pred_island =
            dyn_cast<tf_executor::IslandOp>(control_predecessor)) {
      new_control_inputs.push_back(control_pred_island.getControl());
    }
  }
  control_inputs_added += new_control_inputs.size();
  fetch_op.getFetchesMutable().append(new_control_inputs);

  VLOG(2) << "Number of control inputs added: " << control_inputs_added;
}

void UpdateControlDependenciesPass::runOnOperation() {
  ModuleOp module = getOperation();
  // This pass assumes that all functions are suitable for export, i.e., each
  // function has a single tf_executor.graph op and all islands wrap single
  // ops.
  if (failed(tensorflow::VerifyExportSuitable(module))) {
    signalPassFailure();
    return;
  }
  TF::SideEffectAnalysis side_effect_analysis(module);

  // Recompute control dependencies between all islands.
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    const auto& analysis_for_func =
        side_effect_analysis.GetAnalysisForFunc(func);
    UpdateAllControlDependencies(func, analysis_for_func);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorUpdateControlDependenciesPass() {
  return std::make_unique<UpdateControlDependenciesPass>();
}

}  // namespace tf_executor
}  // namespace mlir
