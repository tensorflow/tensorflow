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

#include "tensorflow/compiler/mlir/tensorflow/utils/cluster_util.h"

#include <functional>
#include <string>
#include <vector>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypeInterfaces.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"

namespace mlir::TF {

namespace {
// An op can be merged into cluster if it satisfies both of the following
// conditions:
//
//  * All of its operands are one of the following:
//    1) A block argument
//    2) A value produced by other clusters.
//    3) Defined before the cluster
//    4) Defined by an operation in the cluster
//    5) Defined by a constant
//  * Merging the op into the cluster does not reorder control dependencies.
//
// TODO(ycao): This is not optimal as it doesn't consider the situation of
// defining_op's operands all meet the requirements above. In that case, the
// defining_op can be moved and to_merge op would be legal to absorb.
bool CanMergeIntoCluster(
    const Cluster& c, Operation* to_merge,
    const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target) {
  // If any of the op's control predecessors appears after the last op in the
  // cluster, merging the op may cause control dependencies to be reordered.
  // Hence, the op cannot be merged to the cluster in such a case.
  const bool has_control_predecessors_after_cluster =
      !side_effect_analysis
           .DirectControlPredecessors(
               to_merge,
               [&c](Operation* pred) {
                 Operation* const last_c_op = c.ops.back();
                 return last_c_op->getBlock() == pred->getBlock() &&
                        last_c_op->isBeforeInBlock(pred);
               })
           .empty();
  if (has_control_predecessors_after_cluster) {
    return false;
  }

  return llvm::all_of(to_merge->getOperands(), [&](Value operand) {
    // Block arguments.
    if (operand.isa<BlockArgument>()) return true;

    if (matchPattern(operand, m_Constant())) return true;

    Operation* defining_op = operand.getDefiningOp();

    // Operand produced by other islands.
    if (defining_op->getBlock() != c.ops.front()->getBlock()) return true;

    // Defining op is before the cluster.
    if (defining_op->isBeforeInBlock(c.ops.front())) return true;

    // Defining op is between first and last operation in cluster. Note that
    // cluster may contain operations that are non-continuous in their original
    // block, thus we also need to check defining_op is also assigned to
    // cluster's target to be sure. This is a faster check than linearly
    // searching through all ops in cluster.
    if (defining_op->isBeforeInBlock(c.ops.back()->getNextNode()) &&
        get_target(defining_op) == c.target)
      return true;

    // Other cases, operand is generated after or outside the cluster, this
    // means it is illegal to merge operation.
    return false;
  });
}
}  // namespace

llvm::StringMap<SmallVector<Cluster>> BuildAllClusters(
    Block& block, const TF::SideEffectAnalysis::Info& side_effect_analysis,
    std::function<std::string(Operation*)> get_target,
    std::function<bool(Operation*)> is_ignored_op) {
  // Iteratively find clusters of different targets within the `block`.
  // Whenever we see an operation that is assigned to an accelerator target
  // (ie. get_target(op) != ""), we try to merge it into the last cluster
  // of same target. If that is infeasible (say because of violating
  // def-before-use), create a new cluster with that operation and move on.
  llvm::StringMap<SmallVector<Cluster>> all_clusters;

  llvm::StringMap<Cluster> nearest_clusters;
  for (Operation& op : llvm::make_early_inc_range(block)) {
    if (is_ignored_op(&op)) {
      continue;
    }
    std::string target_name = get_target(&op);

    // If no cluster of same target has been formed yet, create a new cluster
    // with op alone.
    auto it = nearest_clusters.find(target_name);
    if (it == nearest_clusters.end()) {
      nearest_clusters[target_name] = Cluster{{&op}, target_name};
      continue;
    }

    // Check if it is legal to merge op into nearest cluster of same target.
    // If positive, update cluster and move on to next operation.
    Cluster& nearest_cluster = it->second;
    if (CanMergeIntoCluster(nearest_cluster, &op, side_effect_analysis,
                            get_target)) {
      nearest_cluster.ops.emplace_back(&op);
      continue;
    }

    // If nearest cluster of same target can not absorb `op`, then that
    // cluster needs to be finalized by inserting into the final cluster map
    // that contains all operations in clusters.
    all_clusters[target_name].push_back(nearest_cluster);

    // Create a new cluster to hold op alone and update nearest_clusters.
    nearest_clusters[target_name] = Cluster{{&op}, target_name};
  }

  // At the end, there might be left-over found clusters that need to be
  // built.
  for (auto& target_cluster : nearest_clusters) {
    all_clusters[target_cluster.first()].push_back(target_cluster.second);
  }

  return all_clusters;
}

void ReorderOpResultUses(mlir::Operation* cluster) {
  mlir::Block* const cluster_block = cluster->getBlock();
  llvm::SetVector<mlir::Operation*> ops_to_reorder;

  llvm::SmallVector<mlir::Value> worklist;
  llvm::append_range(worklist, cluster->getResults());

  while (!worklist.empty()) {
    mlir::Value value = worklist.back();
    worklist.pop_back();

    for (mlir::Operation* const user : value.getUsers()) {
      mlir::Operation* const op = cluster_block->findAncestorOpInBlock(*user);
      if (op == nullptr || !op->isBeforeInBlock(cluster)) {
        continue;
      }

      if (ops_to_reorder.insert(op)) {
        llvm::append_range(worklist, op->getResults());
      }
    }
  }

  std::vector<mlir::Operation*> sorted = ops_to_reorder.takeVector();
  llvm::sort(sorted, [](mlir::Operation* lhs, mlir::Operation* rhs) {
    return lhs->isBeforeInBlock(rhs);
  });

  for (mlir::Operation* const op : llvm::reverse(sorted)) {
    op->moveAfter(cluster);
  }
}

}  // namespace mlir::TF
