/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/analysis/side_effect_analysis.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

using RegionVec2D = llvm::SmallVector<llvm::SmallVector<TF::IfRegionOp, 8>, 8>;
using OperationVec2D = llvm::SmallVector<llvm::SmallVector<Operation*, 8>, 8>;
using MapToRegionVec2D = llvm::SmallDenseMap<Value, RegionVec2D>;
using MapToOperationVec2D = llvm::SmallDenseMap<Value, OperationVec2D>;
using IfOpIter =
    llvm::SmallVectorTemplateCommon<mlir::TF::IfRegionOp>::iterator;

struct MergeControlFlowPass
    : public TF::MergeControlFlowPassBase<MergeControlFlowPass> {
  void runOnOperation() override;
};

// Gets the IfRegion op and all of ops in the then and else branches.
llvm::SmallSetVector<Operation*, 4> GetAllOpsFromIf(TF::IfRegionOp if_op) {
  llvm::SmallSetVector<Operation*, 4> all_ops;
  all_ops.insert(if_op);
  for (Operation& op : if_op.then_branch().front()) {
    all_ops.insert(&op);
  }
  for (Operation& op : if_op.else_branch().front()) {
    all_ops.insert(&op);
  }
  return all_ops;
}

// Returns whether it is safe to merge `source` IfRegion into `destination`
// IfRegion. `source` must come after `destination`.
bool SafeToMerge(TF::IfRegionOp source, TF::IfRegionOp destination,
                 llvm::SmallSetVector<Operation*, 4>& middle_if_ops,
                 const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  // IfRegion ops must be in the same block.
  assert(source.getOperation() != nullptr);
  assert(destination.getOperation() != nullptr);
  if (source.getOperation()->getBlock() !=
      destination.getOperation()->getBlock()) {
    return false;
  }
  assert(destination.getOperation()->isBeforeInBlock(source.getOperation()));

  llvm::SmallSetVector<Operation*, 4> source_ops = GetAllOpsFromIf(source);
  llvm::SmallSetVector<Operation*, 4> destination_ops =
      GetAllOpsFromIf(destination);

  // If there is an intermediate data or side effect dependency between the
  // ops in destination and the ops in the source, it's not safe to merge
  // them.
  std::vector<Operation*> dependencies;
  for (auto* user : destination.getOperation()->getUsers()) {
    if (!source_ops.contains(user) && !middle_if_ops.contains(user)) {
      dependencies.push_back(user);
    }
  }
  for (auto* successor : side_effect_analysis.DirectControlSuccessors(
           destination.getOperation())) {
    if (!source_ops.contains(successor) && !middle_if_ops.contains(successor)) {
      dependencies.push_back(successor);
    }
  }
  for (Operation& op : destination.then_branch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!source_ops.contains(successor) &&
          !destination_ops.contains(successor) &&
          !middle_if_ops.contains(successor))
        dependencies.push_back(successor);
    }
  }
  for (Operation& op : destination.else_branch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!source_ops.contains(successor) &&
          !destination_ops.contains(successor) &&
          !middle_if_ops.contains(successor))
        dependencies.push_back(successor);
    }
  }

  bool safe_to_merge = true;

  llvm::SmallPtrSet<Operation*, 4> visited;
  while (!dependencies.empty()) {
    Operation* dependency = dependencies.back();
    dependencies.pop_back();
    if (visited.count(dependency)) continue;
    visited.insert(dependency);
    for (auto* user : dependency->getUsers()) {
      if (source_ops.contains(user) || middle_if_ops.contains(user)) {
        safe_to_merge = false;
        break;
      } else {
        dependencies.push_back(user);
      }
    }
    for (auto* successor :
         side_effect_analysis.DirectControlSuccessors(dependency)) {
      if (source_ops.contains(successor) || middle_if_ops.contains(successor)) {
        safe_to_merge = false;
        break;
      } else {
        dependencies.push_back(successor);
      }
    }
    // If the op is nested, then also consider the users and successors of the
    // parent op.
    if (dependency->getBlock() != destination.getOperation()->getBlock())
      dependencies.push_back(dependency->getParentOp());
    if (!safe_to_merge) break;
  }
  return safe_to_merge;
}

// Checks whether a return indice should be kep for `first_if_op` by checking
// for results in `second_if_op`.
llvm::SmallVector<int, 4> GetReturnIndicesToKeep(TF::IfRegionOp first_if_op,
                                                 TF::IfRegionOp second_if_op) {
  llvm::SmallVector<int, 4> return_indices_to_keep;
  for (auto& index_and_value : llvm::enumerate(first_if_op.getResults())) {
    if (!llvm::all_of(index_and_value.value().getUsers(), [&](Operation* op) {
          return second_if_op->isProperAncestor(op);
        })) {
      return_indices_to_keep.push_back(index_and_value.index());
    }
  }
  return return_indices_to_keep;
}

// Move the body excluding the terminators of else and then regions from
// 'source' to 'destination'.
void MoveBranches(TF::IfRegionOp source, TF::IfRegionOp destination) {
  Block& destination_then_block = destination.then_branch().front();
  auto& source_then_body = source.then_branch().front().getOperations();
  destination_then_block.getOperations().splice(
      destination_then_block.without_terminator().end(), source_then_body,
      source_then_body.begin(), std::prev(source_then_body.end()));

  Block& destination_else_block = destination.else_branch().front();
  auto& source_else_body = source.else_branch().front().getOperations();
  destination_else_block.getOperations().splice(
      destination_else_block.without_terminator().end(), source_else_body,
      source_else_body.begin(), std::prev(source_else_body.end()));
}

IfOpIter FindSubgroupLastIfOpIter(
    IfOpIter current, IfOpIter potential_last,
    llvm::SmallVector<mlir::TF::IfRegionOp, 8>& if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  assert(potential_last);
  if (current == potential_last) {
    return potential_last;
  }

  // middle_if_ops contain ops in those IfRegions between source IfRegion
  // and destination IfRegion
  llvm::SmallSetVector<Operation*, 4> middle_if_ops;
  auto get_middle_if_ops = [&](TF::IfRegionOp intermediate_op) {
    auto all_ops = GetAllOpsFromIf(intermediate_op);
    for (auto& op : all_ops) {
      middle_if_ops.insert(op);
    }
    return middle_if_ops;
  };
  TF::IfRegionOp first_if_op = *current;
  FuncOp func = first_if_op->getParentOfType<FuncOp>();
  const TF::SideEffectAnalysis::Info& analysis =
      side_effect_analysis->GetAnalysisForFunc(func);

  for (auto it = std::next(current); it != std::next(potential_last); it++) {
    TF::IfRegionOp second_if_op = *it;
    if (SafeToMerge(second_if_op, first_if_op, middle_if_ops, analysis)) {
      middle_if_ops = get_middle_if_ops(second_if_op);
      continue;
    }
    potential_last = std::prev(it);
    break;
  }
  if (potential_last == current || potential_last == std::next(current)) {
    return potential_last;
  }
  return FindSubgroupLastIfOpIter(std::next(current), potential_last, if_ops,
                                  side_effect_analysis);
}

absl::flat_hash_set<Operation*> getMoveOpsBetweenTwoIfRegion(
    Operation* result_op, Operation* after_op,
    llvm::SmallSetVector<Operation*, 4> middle_if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  Block* block = after_op->getBlock();
  std::queue<Operation*> queue;
  absl::flat_hash_set<Operation*> visited;
  absl::flat_hash_set<Operation*> move_ops;

  FuncOp func = result_op->getParentOfType<FuncOp>();
  const TF::SideEffectAnalysis::Info& analysis =
      side_effect_analysis->GetAnalysisForFunc(func);

  // Enqueue dependencies of source_op into queue.
  auto enqueue_deps = [&](Operation* source_op) {
    for (Operation* user : source_op->getUsers()) {
      if (!visited.count(user) && !middle_if_ops.count(user)) {
        visited.insert(user);
        queue.push(user);
      }
    }
    source_op->walk([&](Operation* walked_op) {
      for (Operation* successor : analysis.DirectControlSuccessors(walked_op)) {
        if (!source_op->isProperAncestor(successor)) {
          if (!visited.count(successor) && !middle_if_ops.count(successor)) {
            visited.insert(successor);
            queue.push(successor);
          }
        }
      }
    });
  };
  enqueue_deps(result_op);

  // Populate move_ops.
  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
    while (op->getBlock() != block) op = op->getParentOp();
    if (op->isBeforeInBlock(after_op)) {
      move_ops.insert(op);
      enqueue_deps(op);
    }
  }
  return move_ops;
}

llvm::SmallVector<Operation*, 8> getMoveOplist(
    llvm::SmallVector<TF::IfRegionOp, 8>& sub_if_group,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  absl::flat_hash_set<Operation*> all_move_ops;
  Operation* last_if_op = sub_if_group.back().getOperation();
  llvm::SmallSetVector<Operation*, 4> middle_if_ops;

  // reversely calculate the all ops need to be moved because in this way,
  // ops in the middle if regions can be easily gained.
  for (auto it = std::prev(std::prev(sub_if_group.end()));
       std::next(it) != sub_if_group.begin(); --it) {
    auto op_list = getMoveOpsBetweenTwoIfRegion(
        it->getOperation(), last_if_op, middle_if_ops, side_effect_analysis);
    all_move_ops.insert(op_list.begin(), op_list.end());
    auto first_if_ops = GetAllOpsFromIf(*it);
    middle_if_ops.insert(first_if_ops.begin(), first_if_ops.end());
  }

  llvm::SmallVector<Operation*, 8> move_ops_ordered;
  move_ops_ordered.reserve(all_move_ops.size());
  for (Operation& op : *last_if_op->getBlock()) {
    if (all_move_ops.count(&op)) {
      move_ops_ordered.push_back(&op);
    }
  }

  return move_ops_ordered;
}

void GenerateSegmentsPerIfGroups(
    const mlir::Value& if_cond,
    llvm::SmallVector<mlir::TF::IfRegionOp, 8>& if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis,
    MapToRegionVec2D& merged_groups, MapToOperationVec2D& move_ops_groups) {
  auto it = merged_groups.try_emplace(if_cond);
  auto it2 = move_ops_groups.try_emplace(if_cond);
  llvm::SmallVector<TF::IfRegionOp, 8> sub_merged_groups;
  auto begin_if_op_iter = if_ops.begin();
  auto last_if_op_iter = std::prev(if_ops.end());

  while (begin_if_op_iter != if_ops.end()) {
    auto current_last_if_op_iter = FindSubgroupLastIfOpIter(
        begin_if_op_iter, last_if_op_iter, if_ops, side_effect_analysis);
    assert(current_last_if_op_iter != if_ops.end());
    llvm::SmallVector<TF::IfRegionOp, 8> sub_if_group;
    for (auto it = begin_if_op_iter; it != std::next(current_last_if_op_iter);
         ++it) {
      sub_if_group.push_back(*it);
    }
    it.first->getSecond().push_back(sub_if_group);
    it2.first->getSecond().push_back(
        getMoveOplist(sub_if_group, side_effect_analysis));
    begin_if_op_iter = std::next(current_last_if_op_iter);
  }
}

llvm::SmallVector<llvm::SmallVector<int, 4>> GetReturnIndicesVec(
    const llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment) {
  llvm::SmallVector<llvm::SmallVector<int, 4>> res;
  llvm::SmallSetVector<int, 4> indices_to_keep_set;
  for (auto it = if_op_segment.begin(); it != if_op_segment.end(); ++it) {
    indices_to_keep_set.clear();
    for (auto it2 = if_op_segment.begin(); it2 != if_op_segment.end(); ++it2) {
      if (it == it2) {
        continue;
      }
      auto return_indices = GetReturnIndicesToKeep(*it, *it2);
      indices_to_keep_set.insert(return_indices.begin(), return_indices.end());
    }

    llvm::SmallVector<int, 4> indices_to_keep_vec(indices_to_keep_set.begin(),
                                                  indices_to_keep_set.end());

    llvm::sort(indices_to_keep_vec.begin(), indices_to_keep_vec.end());

    res.push_back(indices_to_keep_vec);
  }
  return res;
}

void ReplaceInternalUsage(llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment) {
  for (auto it = if_op_segment.begin(); it != if_op_segment.end(); ++it) {
    for (auto it2 = std::next(it); it2 != if_op_segment.end(); ++it2) {
      for (OpResult result : it->getResults()) {
        replaceAllUsesInRegionWith(
            result,
            it->then_branch().front().getTerminator()->getOperand(
                result.getResultNumber()),
            it2->then_branch());
        replaceAllUsesInRegionWith(
            result,
            it->else_branch().front().getTerminator()->getOperand(
                result.getResultNumber()),
            it2->else_branch());
      }
    }
  }
}

void MoveOpsAfterNewIfOps(
    Operation* after_op,
    const llvm::SmallVector<Operation*, 8>& move_ops_ordered) {
  // Move ops in order.
  for (Operation* op : move_ops_ordered) {
    if (op->isBeforeInBlock(after_op)) {
      op->moveAfter(after_op);
      after_op = op;
    }
  }
}

void ReplaceExternalUsage(
    llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment,
    TF::IfRegionOp new_if_op,
    llvm::SmallVector<llvm::SmallVector<int, 4>>& return_indices) {
  int new_return_index = 0;
  for (const auto& index_and_value : llvm::enumerate(if_op_segment)) {
    auto old_if_op = index_and_value.value();
    for (int i : return_indices[index_and_value.index()]) {
      old_if_op.getResult(i).replaceAllUsesWith(
          new_if_op.getResult(new_return_index++));
    }
  }
}

void UpdateMoveOpList(llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment,
                      MapToOperationVec2D& move_ops_groups,
                      TF::IfRegionOp new_if_op) {
  Block* block = new_if_op.getOperation()->getBlock();
  auto if_cond = new_if_op.cond();
  for (auto& entry : move_ops_groups) {
    for (auto& move_op_ordered : entry.second) {
      if (entry.first == if_cond) {
        continue;
      }
      bool needAddNewIfOp = false;
      for (auto& if_op : if_op_segment) {
        auto iter = std::find(move_op_ordered.begin(), move_op_ordered.end(),
                              if_op.getOperation());
        if (iter != move_op_ordered.end()) {
          move_op_ordered.erase(iter);
          needAddNewIfOp = true;
        }
      }
      if (needAddNewIfOp) {
        move_op_ordered.push_back(new_if_op.getOperation());
        absl::flat_hash_set<Operation*> all_move_ops(move_op_ordered.begin(),
                                                     move_op_ordered.end());
        move_op_ordered.clear();
        for (Operation& op : *block) {
          if (all_move_ops.count(&op)) {
            move_op_ordered.push_back(&op);
          }
        }
      }
    }
  }
}

void MergeIfPerSegment(llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment,
                       llvm::SmallVector<Operation*, 8>& move_ops_list,
                       MapToOperationVec2D& move_ops_groups) {
  TF::IfRegionOp destination = if_op_segment[0];
  llvm::SmallVector<Type, 4> merged_return_types;
  llvm::SmallVector<TF::IfRegionOp, 8> sources_if_ops(
      std::next(if_op_segment.begin()), if_op_segment.end());

  // Create new IfRegion's merged results
  auto return_indices = GetReturnIndicesVec(if_op_segment);
  for (const auto& index_and_value : llvm::enumerate(return_indices)) {
    TF::IfRegionOp if_op = if_op_segment[index_and_value.index()];
    for (auto i : index_and_value.value()) {
      merged_return_types.push_back(if_op.getResult(i).getType());
    }
  }

  // Create new IfRegion for merged all IfRegions in if_op_segmemt
  OpBuilder builder(destination);
  builder.setInsertionPoint(if_op_segment.back().getOperation());

  auto new_if_op = builder.create<TF::IfRegionOp>(
      destination.getLoc(), merged_return_types, destination.cond(),
      llvm::all_of(if_op_segment,
                   [&](TF::IfRegionOp op) { return op.is_stateless(); }),
      destination._then_func_nameAttr(), destination._else_func_nameAttr());
  new_if_op.then_branch().push_back(new Block);
  new_if_op.else_branch().push_back(new Block);

  // Replace internal usages of merged if ops
  ReplaceInternalUsage(if_op_segment);

  // Replace external usages of merged if ops
  ReplaceExternalUsage(if_op_segment, new_if_op, return_indices);

  // Move ops after the new merged If region
  MoveOpsAfterNewIfOps(new_if_op.getOperation(), move_ops_list);

  // Create the Yield ops for both branches with merged results
  llvm::SmallVector<Value, 4> merged_then_yield_values;
  for (const auto& index_and_value : llvm::enumerate(if_op_segment)) {
    auto if_op = index_and_value.value();
    for (auto i : return_indices[index_and_value.index()]) {
      merged_then_yield_values.push_back(
          if_op.then_branch().front().getTerminator()->getOperand(i));
    }
  }
  builder.setInsertionPointToEnd(&new_if_op.then_branch().front());
  builder.create<TF::YieldOp>(
      destination.then_branch().front().getTerminator()->getLoc(),
      /*operands=*/merged_then_yield_values);

  llvm::SmallVector<Value, 4> merged_else_yield_values;
  for (const auto& index_and_value : llvm::enumerate(if_op_segment)) {
    auto if_op = index_and_value.value();
    for (auto i : return_indices[index_and_value.index()]) {
      merged_else_yield_values.push_back(
          if_op.else_branch().front().getTerminator()->getOperand(i));
    }
  }
  builder.setInsertionPointToEnd(&new_if_op.else_branch().front());
  builder.create<TF::YieldOp>(
      destination.else_branch().front().getTerminator()->getLoc(),
      /*operands=*/merged_else_yield_values);

  for (auto& old_if_op : if_op_segment) {
    MoveBranches(/*source=*/old_if_op, /*destination=*/new_if_op);
  }

  // Move op list need to be updated due to the cases
  // when the merged if regions are in the moved op list.
  // We need remove those old if regions and add the new if region into the list
  UpdateMoveOpList(if_op_segment, move_ops_groups, new_if_op);

  for (auto& old_if_op : if_op_segment) {
    old_if_op.erase();
  }
}

void MergeIfPerIfGroups(const Value& if_cond,
                        MapToRegionVec2D& plan_merged_groups,
                        MapToOperationVec2D& move_ops_groups) {
  OperationVec2D move_ops_group = move_ops_groups[if_cond];
  RegionVec2D segments = plan_merged_groups[if_cond];

  assert(segments.size() == move_ops_group.size());
  for (auto i = 0; i < segments.size(); ++i) {
    if (segments[i].size() >= 2) {
      MergeIfPerSegment(segments[i], move_ops_group[i], move_ops_groups);
    }
  }
}

// Groups if regions by common predicate and attemps to merge them.
void OptimizeIfRegions(Block* block, ModuleOp module) {
  // Do side effect analysis only one time in the beginning
  auto side_effect_analysis = std::make_unique<TF::SideEffectAnalysis>(module);

  // Determine IfRegions with the same predicate.
  llvm::SmallDenseMap<Value, llvm::SmallVector<TF::IfRegionOp, 8>, 8>
      grouped_if_ops;
  llvm::SmallVector<Value, 4> if_cond_order;
  block->walk([&](TF::IfRegionOp if_op) {
    auto it = grouped_if_ops.try_emplace(if_op.cond());
    if (it.second) {
      if_cond_order.push_back(if_op.cond());
    }
    it.first->getSecond().push_back(if_op);
  });

  MapToRegionVec2D plan_merged_groups;
  MapToOperationVec2D move_ops_groups;

  // For each if group, determine the segments of each if groups
  // that can be merged and their related ops to be moved after
  // the new generated if regions
  // We cache the infomation into two maps:
  // plan_merged_groups and move_ops_groups
  for (const auto& if_cond : if_cond_order) {
    GenerateSegmentsPerIfGroups(if_cond, grouped_if_ops[if_cond],
                                side_effect_analysis, plan_merged_groups,
                                move_ops_groups);
  }

  // Meerge if regions for each if regiond groups
  for (const auto& if_cond : if_cond_order) {
    MergeIfPerIfGroups(if_cond, plan_merged_groups, move_ops_groups);
  }
}

void MergeControlFlowPass::runOnOperation() {
  ModuleOp module = getOperation();
  auto result = module.walk([&](tf_device::ClusterOp cluster) {
    OptimizeIfRegions(&cluster.GetBody(), module);
    return WalkResult::advance();
  });

  if (result.wasInterrupted()) return signalPassFailure();
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateMergeControlFlowPass() {
  return std::make_unique<MergeControlFlowPass>();
}

}  // namespace TFDevice
}  // namespace mlir
