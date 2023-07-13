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
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
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

// Maps group IDs to branch IDs.
using GroupIdToBranchIdMap = absl::flat_hash_map<std::string, std::string>;
// Maps an op to parallel execution IDs.
using OpToParallelIdsMap =
    absl::flat_hash_map<Operation*, GroupIdToBranchIdMap>;
using OpToOpsMap =
    absl::flat_hash_map<Operation*, absl::flat_hash_set<Operation*>>;

// Many operations have the same dependency and parallel id set. We cache the
// processed result of these operations to speed execution.
struct OpCacheEntry {
  Operation* template_op;
  llvm::SmallVector<Operation*, 8> preds_in_reverse_program_order;
};

struct OpCacheKey {
  const llvm::SmallVector<Operation*, 4> deps;
  const GroupIdToBranchIdMap& group_id_to_branch_id_map;

  template <typename H>
  friend H AbslHashValue(H h, const OpCacheKey& c) {
    for (Operation* dep : c.deps) {
      h = H::combine(std::move(h), dep);
    }
    for (auto [group_id, branch_id] : c.group_id_to_branch_id_map) {
      h = H::combine(std::move(h), group_id, branch_id);
    }
    return h;
  }

  bool operator==(const OpCacheKey& other) const {
    return deps == other.deps &&
           group_id_to_branch_id_map == other.group_id_to_branch_id_map;
  }
};

using OpCache = absl::flat_hash_map<OpCacheKey, OpCacheEntry>;

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
//
// `active_transitive_preds` are those dominated by `op`: taking a dependency
// on `op` will also ensure all `active_transitive_preds[op]` are waited
// for.
//
// `inactive_transitive_preds` are transitive dependencies of op in the original
// graph but are not dominated by `op`. (They run in a different parallel
// execution group). They must be separately considered when processing
// successor operations.
void UpdateControlDependenciesForOp(
    Operation* op, const TF::SideEffectAnalysis::Info& analysis_for_func,
    const OpToParallelIdsMap& op_to_parallel_ids_map,
    OpCache& op_cache,
    OpToOpsMap& active_transitive_preds,
    OpToOpsMap& inactive_transitive_preds,
    int& num_control_inputs_removed,
    int& num_control_inputs_added,
    int& num_invalid_dependencies) {
  auto& op_inactive = inactive_transitive_preds[op];
  auto& op_active = active_transitive_preds[op];

  llvm::SmallVector<Operation*, 4> control_deps =
      analysis_for_func.DirectControlPredecessors(op);
  OpCacheKey key = {
    control_deps,
    GetGroupIdToBranchIdMap(op, op_to_parallel_ids_map)
  };

  // We matched with another op in the cache. We will have the same active and
  // inactive dependency sets and control inputs, except we swap out our current
  // op for the template op in the active set.
  if (op_cache.contains(key)) {
    auto& entry = op_cache[key];
    op_active = active_transitive_preds[entry.template_op];
    op_active.insert(op);
    op_active.erase(entry.template_op);

    op_inactive = inactive_transitive_preds[entry.template_op];
    ClearControlInputs(op, num_control_inputs_removed);
    SetControlInputs(op, entry.preds_in_reverse_program_order,
                    num_control_inputs_added);
    return;
  }

  op_active.insert(op);

  // First iterate over all direct control dependencies and collect the set of
  // potential active dependencies.
  absl::flat_hash_set<Operation*> pred_set;
  for (Operation* pred : control_deps) {
    // Inactive transitive predecessors of `pred` are potential direct
    // predecessors of `op` (they are not tracked by `pred`).
    for (Operation* transitive_pred : inactive_transitive_preds[pred]) {
      pred_set.insert(transitive_pred);
      op_inactive.insert(transitive_pred);
    }

    if (IsValidDependency(pred, op, op_to_parallel_ids_map)) {
      // We know that any active transitive predecessors will still be covered
      // by (pred, op), so we don't have to add them to `potential_preds`.
      pred_set.insert(pred);
    } else {
      // Active transitive predecessors will not be covered by (pred, op)
      // anymore, so add them all as candidates.
      pred_set.insert(
          active_transitive_preds[pred].begin(),
          active_transitive_preds[pred].end());
      ++num_invalid_dependencies;
    }
  }

  // Now collect a list of valid dependencies and sort them in program order.
  std::vector<Operation*> potential_preds;
  potential_preds.reserve(pred_set.size());

  for (Operation* potential_pred : pred_set) {
    if (IsValidDependency(potential_pred, op, op_to_parallel_ids_map)) {
      potential_preds.push_back(potential_pred);
    } else {
      // We don't keep the (pred, op) dependency, so all active transitive
      // dependencies become inactive.
      op_inactive.insert(
          active_transitive_preds[potential_pred].begin(),
          active_transitive_preds[potential_pred].end());
    }
  }
  std::sort(potential_preds.begin(), potential_preds.end(), IsAfterInBlock());

  // Finally, accumulate dependencies until we have coverage over all active
  // dependencies.
  llvm::SmallVector<Operation*, 8> preds_in_reverse_program_order;
  for (Operation* potential_pred : potential_preds) {
    if (!op_active.contains(potential_pred)) {
      // `potential_pred` is not an active transitive predecessor of `op` yet,
      // so we must add it as a direct predecessor.
      preds_in_reverse_program_order.push_back(potential_pred);
      // We keep the (pred, op) dependency, so all active transitive
      // dependencies stay active.
      op_active.insert(
          active_transitive_preds[potential_pred].begin(),
          active_transitive_preds[potential_pred].end());
    }
  }

  for (Operation* pred : op_active) {
    op_inactive.erase(pred);
  }

  ClearControlInputs(op, num_control_inputs_removed);
  SetControlInputs(op, preds_in_reverse_program_order,
                   num_control_inputs_added);

  op_cache[key] = {op, preds_in_reverse_program_order};
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
  OpCache op_cache;
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
        op_cache,
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
