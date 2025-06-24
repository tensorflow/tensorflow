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
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/resource_alias_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/verify_suitable_for_graph_export.h"

// This pass is used for checking control dependencies. It is under "transforms"
// because it adds op warnings and could potentially fix unexpected control
// dependencies in the future (therefore changing the IR).

namespace mlir {
namespace tf_executor {

namespace {

#define GEN_PASS_DEF_EXECUTORCHECKCONTROLDEPENDENCIESPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class TFExecutorCheckControlDependencies
    : public impl::ExecutorCheckControlDependenciesPassBase<
          TFExecutorCheckControlDependencies> {
 public:
  void runOnOperation() override;
};

using ResourceIdVec = llvm::SmallVector<std::pair<TF::ResourceId, bool>>;
// MapVector provides both fast lookup and deterministic iteration order.
using IslandToIslandMapVec =
    llvm::MapVector<IslandOp, llvm::SmallVector<IslandOp>>;
using IslandToIslandHashMap = llvm::SmallDenseMap<IslandOp, IslandOp>;

// Returns true iff the IDs could potentially point to the same resource.
//
// Note: This doesn't yet take self-dependent-only resources into account. We
// would need to return false if one resource is unknown and the other one is
// self-dependent-only. This could cause unreported extra dependencies for such
// cases.
bool IsPotentiallySameResource(TF::ResourceId resource_id,
                               TF::ResourceId other_resource_id) {
  return (resource_id == TF::kUnknownResourceId ||
          other_resource_id == TF::kUnknownResourceId ||
          resource_id == other_resource_id);
}

// Returns true iff there is any dependency between the IDs in `resource_ids`
// and the IDs in `other_resource_ids`.
// Note that this can be made more efficient if necessary. For current use cases
// this runtime is negligible (typically at least one of the resource ID vectors
// is small).
bool ResourceIdsHaveDependency(const ResourceIdVec& resource_ids,
                               const ResourceIdVec& other_resource_ids) {
  for (const auto& [resource_id, read_only] : resource_ids) {
    for (const auto& [other_resource_id, other_read_only] :
         other_resource_ids) {
      if (IsPotentiallySameResource(resource_id, other_resource_id) &&
          !(read_only && other_read_only)) {
        return true;
      }
    }
  }
  return false;
}

// Returns true iff there should be a dependency between `source_op` and
// `target_op`.
bool ShouldOpsHaveDependency(
    Operation* source_op, Operation* target_op,
    const TF::SideEffectAnalysis::Info& analysis_for_func) {
  const ResourceIdVec& source_op_resource_ids =
      analysis_for_func.GetResourceIds(source_op);
  const ResourceIdVec& target_op_resource_ids =
      analysis_for_func.GetResourceIds(target_op);
  return ResourceIdsHaveDependency(source_op_resource_ids,
                                   target_op_resource_ids);
}

// Returns true iff the op wrapped by `island` is an intermediate op used for
// grouping control dependencies. We don't want to report warnings for these
// ops, unless they belong to a control path between two side-effecting ops that
// should not have any dependencies.
bool IsIntermediateOp(IslandOp island) {
  // These two side-effect-free ops are known to be used for control dependency
  // grouping (e.g., in `BreakUpIslands` pass).
  return isa<TF::IdentityOp, TF::NoOp>(island.GetBody().front());
}

// Finds a path from `source_op` to `target_op` and stores it in `path`, in
// reverse order (from target to source). Returns false if no path was found,
// true otherwise.
// Note that there can be multiple paths, we don't care about which one we find.
// BFS guarantees that we find a path with minimal number of edges.
bool FindPathBfs(IslandOp source_op, IslandOp target_op,
                 std::vector<IslandOp>& path) {
  std::queue<IslandOp> worklist;
  // Stores predecessor pointers.
  IslandToIslandHashMap pred_map;

  // BFS starting from `source_op`, stop when `target_op` is reached.
  worklist.push(source_op);
  while (!worklist.empty()) {
    IslandOp curr_op = worklist.front();
    worklist.pop();

    if (curr_op == target_op) break;

    for (Operation* user : curr_op.getControl().getUsers()) {
      auto user_island = dyn_cast<IslandOp>(user);
      if (!user_island) continue;
      // We have labeled `user_island` before so it also must have been added to
      // `worklist` before.
      if (pred_map.count(user_island) > 0) continue;

      worklist.push(user_island);
      pred_map[user_island] = curr_op;
    }
  }

  // Construct path by following predecessor pointers.
  IslandOp curr_op = target_op;
  while (curr_op != source_op) {
    // If we don't have a predecessor pointer here, then there is no
    // source-target path.
    if (pred_map.count(curr_op) == 0) return false;
    path.push_back(curr_op);
    curr_op = pred_map[curr_op];
  }
  path.push_back(source_op);
  return true;
}

// Emits dependency warning for `op`.
void EmitDependencyWarningForOp(Operation* op, int path_idx, int node_idx,
                                absl::string_view pos_str) {
  op->emitWarning("unexpected control dependency path: ")
      << "path " << path_idx << ", node " << node_idx << " (" << pos_str << ")";
}

// Constructs path from `source_op` to `target_op` and emits warnings.
//
// Note that in case of many reported paths this can be inefficient since we
// perform one BFS per path. However, the vast majority of paths should not be
// reported so this should be fine (otherwise we have bigger problems). Also,
// the expected path length is very small (typically < 5) which means that BFS
// searches should terminate quickly (this makes BFS preferable to DFS).
// If higher performance is needed, the paths can be cached during the first
// traversal in function `CheckControlDependenciesForFunc`.
void EmitDependencyWarningsForPath(IslandOp source_op, IslandOp target_op,
                                   int path_idx) {
  std::vector<IslandOp> path;
  bool found_path = FindPathBfs(source_op, target_op, path);
  if (!found_path) {
    // This shouldn't happen, returning an error instead of asserting so it
    // doesn't go unnoticed if it ever happens.
    target_op.emitError("no path to target op found, cannot emit warnings");
    return;
  }

  // Emit warnings for path.
  int node_idx = 0;
  for (auto iter = path.rbegin(); iter != path.rend(); ++iter) {
    Operation* op = *iter;
    std::string pos_str;
    if (node_idx == 0) {
      pos_str = "source";
    } else if (node_idx + 1 == path.size()) {
      pos_str = "target";
    } else {
      pos_str = "intermediate";
    }
    EmitDependencyWarningForOp(op, path_idx, node_idx++, pos_str);
  }
}

void CheckControlDependenciesForFunc(
    func::FuncOp func, const TF::SideEffectAnalysis::Info& analysis_for_func,
    int& path_idx) {
  IslandToIslandMapVec op_to_control_sources;

  // Traverse islands in topological order.
  func.walk([&](IslandOp source_island) {
    for (Operation* user : source_island.getControl().getUsers()) {
      auto target_island = dyn_cast<IslandOp>(user);
      if (!target_island) continue;

      if (IsIntermediateOp(source_island)) {
        // Add all sources from intermediate op to target op's sources.
        op_to_control_sources[target_island].append(
            op_to_control_sources[source_island]);
      } else {
        // Add source island to target op's sources.
        op_to_control_sources[target_island].push_back(source_island);
      }
    }
  });

  // Find all control sources for `target_op` for which we should not have a
  // dependency and emit corresponding warnings.
  for (const auto& [target_op, control_sources] : op_to_control_sources) {
    if (IsIntermediateOp(target_op)) continue;
    for (IslandOp source_op : control_sources) {
      if (!ShouldOpsHaveDependency(source_op, target_op, analysis_for_func)) {
        EmitDependencyWarningsForPath(source_op, target_op, path_idx++);
      }
    }
  }
}

void TFExecutorCheckControlDependencies::runOnOperation() {
  ModuleOp module = getOperation();
  // This pass assumes that all functions are suitable for export, i.e., each
  // function has a single tf_executor.graph op and all islands wrap single
  // ops.
  if (failed(tensorflow::VerifyExportSuitable(module))) {
    module.emitOpError() << "not suitable for checking control dependencies";
    return;
  }
  TF::SideEffectAnalysis side_effect_analysis(module);
  // Use a global path index across functions to make it easier to follow one
  // path for debugging purposes.
  int path_idx = 0;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func.isExternal()) continue;
    const auto& analysis_for_func =
        side_effect_analysis.GetAnalysisForFunc(func);
    CheckControlDependenciesForFunc(func, analysis_for_func, path_idx);
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorCheckControlDependenciesPass() {
  return std::make_unique<TFExecutorCheckControlDependencies>();
}

}  // namespace tf_executor
}  // namespace mlir
