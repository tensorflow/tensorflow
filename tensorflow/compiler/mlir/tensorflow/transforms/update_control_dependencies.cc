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
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
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

// Maps group IDs to branch IDs.
using GroupIdToBranchIdMap = absl::flat_hash_map<std::string, std::string>;
// Maps an op to parallel execution IDs.
using OpToParallelIdsMap =
    absl::flat_hash_map<Operation*, GroupIdToBranchIdMap>;
// Maps an op to a set of ops.
using OpToOpsMap =
    absl::flat_hash_map<Operation*, absl::flat_hash_set<Operation*>>;
// Represents a set of ops in reverse program order.
using OpsInReverseProgramOrder = absl::btree_set<Operation*, IsAfterInBlock>;

#define GEN_PASS_DEF_EXECUTORUPDATECONTROLDEPENDENCIESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class UpdateControlDependenciesPass
    : public impl::ExecutorUpdateControlDependenciesPassBase<
          UpdateControlDependenciesPass> {
 public:
  void runOnOperation() override;
};

const GroupIdToBranchIdMap& EmptyGroupIdToBranchIdMap() {
  // clang-format off
  static auto* empty_map = new absl::flat_hash_map<std::string, std::string>{};
  return *empty_map;
}

// Returns map whose elements are the (group ID,branch ID) pairs for `op`.
const GroupIdToBranchIdMap& GetGroupIdToBranchIdMap(
    Operation* op, const OpToParallelIdsMap& op_to_parallel_ids_map) {
  auto iter = op_to_parallel_ids_map.find(op);
  if (iter == op_to_parallel_ids_map.end()) return EmptyGroupIdToBranchIdMap();
  return iter->second;
}

// Returns true iff a control dependency between both ops is considered valid,
// depending on their parallel execution IDs.
// A control dependency is invalid if both ops share a common parallel execution
// group with different branch IDs (in that case, the ops are expected to run in
// parallel).
bool IsValidDependency(Operation* op, Operation* other_op,
                          const OpToParallelIdsMap& op_to_parallel_ids_map) {
  const GroupIdToBranchIdMap& parallel_ids_map =
      GetGroupIdToBranchIdMap(op, op_to_parallel_ids_map);
  const GroupIdToBranchIdMap& other_parallel_ids_map =
      GetGroupIdToBranchIdMap(other_op, op_to_parallel_ids_map);

  for (auto [group_id, branch_id] : parallel_ids_map) {
    auto iter = other_parallel_ids_map.find(group_id);
    // `other_op` has same group as `op`, with different branch ID.
    if (iter != other_parallel_ids_map.end() && iter->second != branch_id) {
      return false;
    }
  }
  // The ops don't share a common group with different branch IDs.
  return true;
}

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
    GraphOp graph, OpToParallelIdsMap& op_to_parallel_ids_map) {
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

    GroupIdToBranchIdMap& ids_map = op_to_parallel_ids_map[island];
    for (auto [group_id, branch_id] : id_pairs) ids_map[group_id] = branch_id;
  }
  return success();
}

// Computes and sets direct control inputs for `op`. Also fills
// `active_transitive_preds` and `inactive_transitive_preds` for `op`.
void
UpdateControlDependenciesForOp(
    Operation* op, const TF::SideEffectAnalysis::Info& analysis_for_func,
    const OpToParallelIdsMap& op_to_parallel_ids_map,
    OpToOpsMap& active_transitive_preds,
    OpToOpsMap& inactive_transitive_preds,
    int& num_control_inputs_removed,
    int& num_control_inputs_added,
    int& num_invalid_dependencies) {
  OpsInReverseProgramOrder potential_preds;
  active_transitive_preds[op].insert(op);
  for (Operation* pred : analysis_for_func.DirectControlPredecessors(op)) {
    // Propagate inactive transitive dependencies from `pred` to `op`.
    inactive_transitive_preds[op].insert(
        inactive_transitive_preds[pred].begin(),
        inactive_transitive_preds[pred].end());
    // Inactive transitive predecessors of `pred` are potential direct
    // predecessors of `op` (they are not tracked by `pred`).
    for (Operation* transitive_pred : inactive_transitive_preds[pred]) {
      potential_preds.insert(transitive_pred);
    }
    if (IsValidDependency(pred, op, op_to_parallel_ids_map)) {
      // We know that any active transitive predecessors will still be covered
      // by (pred, op), so we don't have to add them to `potential_preds`.
      potential_preds.insert(pred);
    } else {
      // Active transitive predecessors will not be covered by (pred, op)
      // anymore, so add them all as candidates.
      for (Operation* transitive_pred : active_transitive_preds[pred]) {
        potential_preds.insert(transitive_pred);
      }
      ++num_invalid_dependencies;
    }
  }
  llvm::SmallVector<Operation*, 8> preds_in_reverse_program_order;
  for (Operation* potential_pred : potential_preds) {
    bool is_valid =
        IsValidDependency(potential_pred, op, op_to_parallel_ids_map);
    if (!is_valid) {
      // We don't keep the (pred, op) dependency, so all active transitive
      // dependencies become inactive.
      inactive_transitive_preds[op].insert(
          active_transitive_preds[potential_pred].begin(),
          active_transitive_preds[potential_pred].end());
    } else if (!active_transitive_preds[op].contains(potential_pred)) {
      // `potential_pred` is not an active transitive predecessor of `op` yet,
      // so we must add it as a direct predecessor.
      preds_in_reverse_program_order.push_back(potential_pred);
      // We keep the (pred, op) dependency, so all active transitive
      // dependencies stay active.
      active_transitive_preds[op].insert(
          active_transitive_preds[potential_pred].begin(),
          active_transitive_preds[potential_pred].end());
    }
  }
  ClearControlInputs(op, num_control_inputs_removed);
  SetControlInputs(op, preds_in_reverse_program_order,
                   num_control_inputs_added);
}

// This function updates all control dependencies in `func`, represented as
// control inputs for island and fetch ops of the graph body in `func`.
// Ideally, we would purely rely on side effect analysis here and propagate
// the queried dependencies to the island and fetch ops. However, this is
// currently not in line with execution semantics in case of replication and
// parallel executes: If two ops originated from different branches of a
// `tf_device.replicate` or `tf_device.parallel_execute` op, then there should
// be no control dependency between them irrespective of side effects, even if
// this could cause a race condition (see b/262304795).
// Because of this, we need to keep track of the origin of such ops which we do
// via `kParallelExecAnnotation` attributes that are interpreted in this pass.
//
// NOTE: This pass guarantees the minimum number of control inputs. Its runtime
// and space complexity can be quadratic in the number of side-effecting ops per
// function. If that becomes a problem in practice, we could look into speed-ups
// used for `DependencyOptimizer::TransitiveReduction` which solves a similar
// problem and also has worst-case quadratic runtime and space complexity.
// Alternatively, we could allow redundant control inputs (less bookkeeping).
LogicalResult UpdateAllControlDependencies(
    func::FuncOp func, const TF::SideEffectAnalysis::Info& analysis_for_func) {
  int num_control_inputs_removed = 0;
  int num_control_inputs_added = 0;
  int num_invalid_dependencies = 0;

  // Maps island ops to parallel IDs of the wrapped ops.
  OpToParallelIdsMap op_to_parallel_ids_map;
  OpToOpsMap active_transitive_preds, inactive_transitive_preds;

  // We call `VerifyExportSuitable` in the beginning of the pass, so every
  // function has a single graph op.
  auto graph = cast<GraphOp>(func.front().front());
  if (failed(FillOpToParallelIdsMap(graph, op_to_parallel_ids_map))) {
    return failure();
  }
  for (Operation& op_ref : graph.GetBody()) {
    Operation* op = &op_ref;
    // We only represent control dependencies between island and fetch ops.
    if (!isa<IslandOp, FetchOp>(op)) continue;
    UpdateControlDependenciesForOp(
        op,
        analysis_for_func,
        op_to_parallel_ids_map,
        active_transitive_preds,
        inactive_transitive_preds,
        num_control_inputs_removed,
        num_control_inputs_added,
        num_invalid_dependencies);
  }
  VLOG(2) << "Number of control inputs removed: " << num_control_inputs_removed;
  VLOG(2) << "Number of control inputs added: " << num_control_inputs_added;
  VLOG(2) << "Number of invalid dependencies: " << num_invalid_dependencies;
  return success();
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
  for (auto func : module.getOps<func::FuncOp>()) {
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
