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

#include <algorithm>
#include <cassert>
#include <memory>

#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_remaining_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

namespace mlir {
namespace tf_executor {
namespace {

// Comparator for `OpsInReverseProgramOrder`.
struct IsAfterInBlock {
  bool operator()(Operation* op, Operation* other_op) const {
    // This function has an average complexity of O(1).
    return other_op->isBeforeInBlock(op);
  }
};

#define GEN_PASS_DEF_EXECUTORUPDATECONTROLDEPENDENCIESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class UpdateControlDependenciesPass
    : public impl::ExecutorUpdateControlDependenciesPassBase<
          UpdateControlDependenciesPass> {
 public:
  void runOnOperation() override;
};

void ClearControlInputs(Operation* op, int& num_control_inputs_removed) {
  // We only call this function for island or fetch ops. The second pair of
  // parentheses is needed for successful compilation.
  assert((isa<IslandOp, FetchOp>(op)));
  if (auto island = dyn_cast<IslandOp>(op)) {
    num_control_inputs_removed += island.getControlInputs().size();
    island.getControlInputsMutable().clear();
  } else if (auto fetch = dyn_cast<FetchOp>(op)) {
    GraphOp graph = fetch->getParentOfType<GraphOp>();
    int num_control_fetches = fetch.getNumOperands() - graph.getNumResults();
    if (num_control_fetches > 0) {
      fetch.getFetchesMutable().erase(graph.getNumResults(),
                                      num_control_fetches);
      num_control_inputs_removed += num_control_fetches;
    }
  }
}

void SetControlInputs(
    Operation* op,
    const llvm::SmallVector<Operation*, 8>& preds_in_reverse_program_order,
    int& num_control_inputs_added) {
  // We only call this function for island or fetch ops. The second pair of
  // parentheses is needed for successful compilation.
  assert((isa<IslandOp, FetchOp>(op)));
  mlir::MutableOperandRange mutable_control_inputs =
      isa<IslandOp>(op) ? cast<IslandOp>(op).getControlInputsMutable()
                        : cast<FetchOp>(op).getFetchesMutable();
  // Add control inputs in program order of the defining ops.
  for (auto iter = preds_in_reverse_program_order.rbegin();
       iter != preds_in_reverse_program_order.rend();
       ++iter) {
    Operation* pred = *iter;
    if (auto pred_island = dyn_cast<mlir::tf_executor::IslandOp>(pred)) {
      mutable_control_inputs.append(pred_island.getControl());
    }
  }
  num_control_inputs_added += preds_in_reverse_program_order.size();
}

// Fills `op_to_parallel_ids_map` from parallel execution attributes in `graph`.
// Returns `failure` iff any attribute is malformed.
LogicalResult FillOpToParallelIdsMap(
    GraphOp graph, TF::OpToParallelIdsMap& op_to_parallel_ids_map) {
  for (Operation& op : graph.GetBody()) {
    auto island = dyn_cast<IslandOp>(&op);
    if (!island) continue;

    // We call `VerifyExportSuitable` in the beginning of the pass, so every
    // island wraps a single op.
    Operation& wrapped_op = island.GetBody().front();
    TF::ParallelExecutionIdPairs id_pairs;
    if (failed(TF::ParseParallelExecutionIds(&wrapped_op, id_pairs))) {
      wrapped_op.emitError()
          << "Malformed " << TF::kParallelExecAnnotation << " attribute";
      return failure();
    }
    if (id_pairs.empty()) continue;

    TF::ParallelIdsMap& ids_map = op_to_parallel_ids_map[island];
    for (const auto& [group_id, branch_id] : id_pairs)
      ids_map[group_id] = branch_id;
  }
  return success();
}

// Fills `op_to_parallel_ids_map` from parallel execution attributes in
// `module_op`. Returns `failure` iff any attribute is malformed.
LogicalResult FillOpToParallelIdsMap(
    ModuleOp module_op, TF::OpToParallelIdsMap& op_to_parallel_ids_map) {
  auto result = module_op->walk([&](GraphOp graph) {
    if (failed(FillOpToParallelIdsMap(graph, op_to_parallel_ids_map)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) return failure();
  return success();
}

// Computes and sets direct control inputs for `op`.
void UpdateControlDependenciesForOp(
    Operation* op, const TF::SideEffectAnalysis::Info& analysis_for_func,
    int& num_control_inputs_removed, int& num_control_inputs_added) {
  llvm::SmallVector<Operation*, 4> control_deps =
      analysis_for_func.DirectControlPredecessors(op);
  ClearControlInputs(op, num_control_inputs_removed);
  llvm::SmallVector<Operation*, 8> preds_in_reverse_program_order(
      control_deps.begin(), control_deps.end());
  std::sort(preds_in_reverse_program_order.begin(),
            preds_in_reverse_program_order.end(), IsAfterInBlock());
  SetControlInputs(op, preds_in_reverse_program_order,
                   num_control_inputs_added);
}

// This function updates all control dependencies in `func`, represented as
// control inputs for island and fetch ops of the graph body in `func`.
// We rely on side effect analysis and propagate the queried dependencies to
// the island and fetch ops.
LogicalResult UpdateAllControlDependencies(
    func::FuncOp func, const TF::SideEffectAnalysis::Info& analysis_for_func) {
  int num_control_inputs_removed = 0;
  int num_control_inputs_added = 0;

  // We call `VerifyExportSuitable` in the beginning of the pass, so every
  // function has a single graph op.
  auto graph = cast<GraphOp>(func.front().front());
  for (Operation& op_ref : graph.GetBody()) {
    Operation* op = &op_ref;
    // We only represent control dependencies between island and fetch ops.
    if (!isa<IslandOp, FetchOp>(op)) continue;
    UpdateControlDependenciesForOp(op, analysis_for_func,
                                   num_control_inputs_removed,
                                   num_control_inputs_added);
  }
  VLOG(2) << "Number of control inputs removed: " << num_control_inputs_removed;
  VLOG(2) << "Number of control inputs added: " << num_control_inputs_added;
  return success();
}

void UpdateControlDependenciesPass::runOnOperation() {
  ModuleOp module_op = getOperation();
  // This pass assumes that all functions are suitable for export, i.e., each
  // function has a single tf_executor.graph op and all islands wrap single
  // ops.
  if (failed(tensorflow::VerifyExportSuitable(module_op))) {
    signalPassFailure();
    return;
  }
  TF::OpToParallelIdsMap op_to_parallel_ids_map;
  if (failed(FillOpToParallelIdsMap(module_op, op_to_parallel_ids_map))) {
    signalPassFailure();
    return;
  }
  TF::SideEffectAnalysis side_effect_analysis(module_op,
                                              op_to_parallel_ids_map);
  for (auto func : module_op.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    const auto& analysis_for_func =
        side_effect_analysis.GetAnalysisForFunc(func);
    if (failed(UpdateAllControlDependencies(func, analysis_for_func))) {
      signalPassFailure();
      return;
    }
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorUpdateControlDependenciesPass() {
  return std::make_unique<UpdateControlDependenciesPass>();
}

}  // namespace tf_executor
}  // namespace mlir
