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
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
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
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace TFDevice {

namespace {

// For the 2d vector type aliases defined below, the first dimension represents
// the class of the IfRegion group and the second dimension represents the
// segments of the IfRegion group.
// For example, if we want to merge the following six IfRegions
// which share the same if_cond (regionA)
// `````````````
// IfRegionA(1)
// IfRegionA(2)
// IfRegionA(3)
// IfRegionA(4)
// IfRegionA(5)
// IfRegionA(6)
// ``````````````
// After the analysis, we consider IfRegionA(1), IfRegionA(2) and IfRegionA(3)
// can be merged, IfRegionA(4) is standalone, IfRegionA(5) and IfRegionA(6)
// can be merged. Then the defined 2D vector is
// [[IfRegionA(1), IfRegionA(2), IfRegionA(3)],
//  [IfRegionA(4)],
//  [IfRegionA(5), IfRegionA(6)]]
using RegionVec2D = llvm::SmallVector<llvm::SmallVector<TF::IfRegionOp, 8>, 8>;
using OperationVec2D = llvm::SmallVector<llvm::SmallVector<Operation*, 8>, 8>;
using MapToRegionVec2D = llvm::SmallDenseMap<Value, RegionVec2D>;
using MapToOperationVec2D = llvm::SmallDenseMap<Value, OperationVec2D>;
using IfOpIterConst =
    llvm::SmallVectorTemplateCommon<mlir::TF::IfRegionOp>::const_iterator;

#define GEN_PASS_DEF_MERGECONTROLFLOWPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct MergeControlFlowPass
    : public impl::MergeControlFlowPassBase<MergeControlFlowPass> {
  void runOnOperation() override;
};

// Gets the IfRegion op and all of ops in the then and else branches.
llvm::SmallSetVector<Operation*, 4> GetAllOpsFromIf(TF::IfRegionOp if_op) {
  llvm::SmallSetVector<Operation*, 4> all_ops;
  all_ops.insert(if_op);
  for (Operation& op : if_op.getThenBranch().front()) {
    all_ops.insert(&op);
  }
  for (Operation& op : if_op.getElseBranch().front()) {
    all_ops.insert(&op);
  }
  return all_ops;
}

// Returns whether it is safe to merge `second_if` IfRegion into `first_if`
// IfRegion. `second if` must come after `first_if`.
// Note that `downstream_if_ops` means the ops in IfRegions except`first_if`.
bool SafeToMerge(TF::IfRegionOp first_if, TF::IfRegionOp second_if,
                 llvm::SmallSetVector<Operation*, 4>& downstream_if_ops,
                 const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  // IfRegion ops must be in the same block.
  if (second_if.getOperation()->getBlock() !=
      first_if.getOperation()->getBlock()) {
    return false;
  }
  assert(first_if.getOperation()->isBeforeInBlock(second_if.getOperation()));

  llvm::SmallSetVector<Operation*, 4> destination_ops =
      GetAllOpsFromIf(first_if);

  // If there is an intermediate data or side effect dependency between the
  // ops in first_if and the ops in second_if, it's not safe to merge
  // them.
  std::vector<Operation*> dependencies;
  for (auto* user : first_if.getOperation()->getUsers()) {
    if (!downstream_if_ops.contains(user)) {
      dependencies.push_back(user);
    }
  }
  for (auto* successor :
       side_effect_analysis.DirectControlSuccessors(first_if.getOperation())) {
    if (!downstream_if_ops.contains(successor)) {
      dependencies.push_back(successor);
    }
  }
  for (Operation& op : first_if.getThenBranch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!downstream_if_ops.contains(successor) &&
          !destination_ops.contains(successor))
        dependencies.push_back(successor);
    }
  }
  for (Operation& op : first_if.getElseBranch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!downstream_if_ops.contains(successor) &&
          !destination_ops.contains(successor))
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
      if (downstream_if_ops.contains(user)) {
        safe_to_merge = false;
        break;
      } else {
        dependencies.push_back(user);
      }
    }
    for (auto* successor :
         side_effect_analysis.DirectControlSuccessors(dependency)) {
      if (downstream_if_ops.contains(successor)) {
        safe_to_merge = false;
        break;
      } else {
        dependencies.push_back(successor);
      }
    }
    // If the op is nested, then also consider the users and successors of the
    // parent op.
    if (dependency->getBlock() != first_if.getOperation()->getBlock())
      dependencies.push_back(dependency->getParentOp());
    if (!safe_to_merge) break;
  }
  return safe_to_merge;
}

// Move the body excluding the terminators of else and then regions from
// 'second_if' to 'first_if'.
void MoveBranches(TF::IfRegionOp first_if, TF::IfRegionOp second_if) {
  Block& first_if_then_block = first_if.getThenBranch().front();
  auto& second_if_then_body = second_if.getThenBranch().front().getOperations();
  first_if_then_block.getOperations().splice(
      first_if_then_block.without_terminator().end(), second_if_then_body,
      second_if_then_body.begin(), std::prev(second_if_then_body.end()));

  Block& first_if_else_block = first_if.getElseBranch().front();
  auto& second_if_else_body = second_if.getElseBranch().front().getOperations();
  first_if_else_block.getOperations().splice(
      first_if_else_block.without_terminator().end(), second_if_else_body,
      second_if_else_body.begin(), std::prev(second_if_else_body.end()));
}

// Check if the `last` IfRegion can be added to the segment of
// IfRegion start with `first` IfRegion.
bool CanAddToIfSegment(
    IfOpIterConst first, IfOpIterConst last,
    const llvm::SmallVector<mlir::TF::IfRegionOp, 8>& if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  if (last == if_ops.end()) {
    return false;
  }
  // downstream_if_ops contain ops in those IfRegions between first IfRegion
  // and last IfRegion plus the ops in the last IfRegion.
  llvm::SmallSetVector<Operation*, 4> downstream_if_ops;

  TF::IfRegionOp second_if_op = *last;

  for (auto iter = std::prev(last); std::next(iter) != first; iter--) {
    TF::IfRegionOp first_if_op = *iter;
    func::FuncOp func = first_if_op->getParentOfType<func::FuncOp>();
    const TF::SideEffectAnalysis::Info& analysis =
        side_effect_analysis->GetAnalysisForFunc(func);
    auto all_ops = GetAllOpsFromIf(*(std::next(iter)));
    downstream_if_ops.insert(all_ops.begin(), all_ops.end());
    if (!SafeToMerge(first_if_op, second_if_op, downstream_if_ops, analysis)) {
      return false;
    }
  }
  return true;
}

// Return the iterator of the IfRegion Op. This is the last IfRegion
// in the segment.
// For example, we have the following sequence of IfRegions
// `````
//      1          2          3         4           5
// IfRegionA, IfRegionA, IfRegionA, IfRegionA, IfRegionA
// `````
// The first three IfRegionA are in one group and the last two are in another
// group. Then when we call FindLastIfInSegment for the first segment, it
// will return iterator of the 3rd IfRegionA.
// In the same way, when we call it for the second segment, it will return
// iterator of the 5th IfRegionA.
IfOpIterConst FindLastIfInSegment(
    IfOpIterConst first_if,
    const llvm::SmallVector<mlir::TF::IfRegionOp, 8>& if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  IfOpIterConst last_if = first_if;
  for (; CanAddToIfSegment(first_if, last_if, if_ops, side_effect_analysis);
       last_if = std::next(last_if)) {
  }
  return std::prev(last_if);
}

// Returns a set of ops to be moved after merged IfRegion between two IfRegions.
absl::flat_hash_set<Operation*> GetMoveOpsBetweenTwoIfRegions(
    Operation* result_op, Operation* after_op,
    llvm::SmallSetVector<Operation*, 4> middle_if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  Block* block = after_op->getBlock();
  std::queue<Operation*> queue;
  absl::flat_hash_set<Operation*> visited;
  absl::flat_hash_set<Operation*> moved_ops;

  func::FuncOp func = result_op->getParentOfType<func::FuncOp>();
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

  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
    while (op->getBlock() != block) op = op->getParentOp();
    if (op->isBeforeInBlock(after_op)) {
      moved_ops.insert(op);
      enqueue_deps(op);
    }
  }
  return moved_ops;
}

// Returns a vector that contains the ops to be moved after merged IfRegion.
// `sub_if_group` refers to a segment of IfRegions.
// The returned vector preserves op order.
llvm::SmallVector<Operation*, 8> GetMoveOpList(
    llvm::SmallVector<TF::IfRegionOp, 8>& sub_if_group,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis) {
  absl::flat_hash_set<Operation*> all_moved_ops;
  Operation* last_if_op = sub_if_group.back().getOperation();
  llvm::SmallSetVector<Operation*, 4> middle_if_ops;

  // reversely calculate the all ops need to be moved because in this way,
  // ops in the middle IfRegions can be easily obtained by simply adding to the
  // current set.
  for (auto it = std::prev(std::prev(sub_if_group.end()));
       std::next(it) != sub_if_group.begin(); --it) {
    auto op_list = GetMoveOpsBetweenTwoIfRegions(
        it->getOperation(), last_if_op, middle_if_ops, side_effect_analysis);
    all_moved_ops.insert(op_list.begin(), op_list.end());
    auto first_if_ops = GetAllOpsFromIf(*it);
    middle_if_ops.insert(first_if_ops.begin(), first_if_ops.end());
  }

  llvm::SmallVector<Operation*, 8> moved_ops_ordered;
  moved_ops_ordered.reserve(all_moved_ops.size());
  for (Operation& op : *last_if_op->getBlock()) {
    if (all_moved_ops.count(&op)) {
      moved_ops_ordered.push_back(&op);
    }
  }

  return moved_ops_ordered;
}

// Generate the segments for each IfRegion groups. Each element in the segments
// are supposed to can be merged into one new IfRegion.`if_cond` refers to the
// if condition of the segment of IfRegions. `if_ops` refers to the segment of
// IfRegions. `merged_groups` refers to all segments of IfRegions.
// `moved_ops_groups` refers to the ops need to be moved after new merged
// IfRegions associated with each segment of IfRegions.
void GenerateSegmentsPerIfGroups(
    const mlir::Value& if_cond,
    const llvm::SmallVector<mlir::TF::IfRegionOp, 8>& if_ops,
    const std::unique_ptr<TF::SideEffectAnalysis>& side_effect_analysis,
    MapToRegionVec2D& merged_groups, MapToOperationVec2D& moved_ops_groups) {
  auto it_merged = merged_groups.try_emplace(if_cond);
  auto it_moved = moved_ops_groups.try_emplace(if_cond);
  llvm::SmallVector<TF::IfRegionOp, 8> sub_merged_groups;
  auto begin_if_op_iter = if_ops.begin();

  while (begin_if_op_iter != if_ops.end()) {
    auto current_last_if_op_iter =
        FindLastIfInSegment(begin_if_op_iter, if_ops, side_effect_analysis);
    assert(current_last_if_op_iter != if_ops.end());
    llvm::SmallVector<TF::IfRegionOp, 8> sub_if_group;
    for (auto it = begin_if_op_iter; it != std::next(current_last_if_op_iter);
         ++it) {
      sub_if_group.push_back(*it);
    }
    it_merged.first->getSecond().push_back(sub_if_group);
    it_moved.first->getSecond().push_back(
        GetMoveOpList(sub_if_group, side_effect_analysis));
    begin_if_op_iter = std::next(current_last_if_op_iter);
  }
}

// Checks whether a return index should be kept for `current_if_op` by checking
// for results in `if_op_segment`.
llvm::SmallVector<int, 4> GetReturnIndicesToKeep(
    TF::IfRegionOp current_if_op,
    const llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment) {
  llvm::SmallVector<int, 4> return_indices_to_keep;
  auto is_op_inside_IfRegions = [&](Operation* op) {
    for (auto& if_op : if_op_segment) {
      if (if_op == current_if_op) {
        continue;
      }
      if (if_op->isProperAncestor(op)) {
        return true;
      }
    }
    return false;
  };
  for (auto& index_and_value : llvm::enumerate(current_if_op.getResults())) {
    if (!llvm::all_of(index_and_value.value().getUsers(),
                      is_op_inside_IfRegions)) {
      return_indices_to_keep.push_back(index_and_value.index());
    }
  }
  return return_indices_to_keep;
}

// Return a vector of the return indices.
llvm::SmallVector<llvm::SmallVector<int, 4>> GetReturnIndicesVec(
    const llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment) {
  llvm::SmallVector<llvm::SmallVector<int, 4>> return_indices_vec;
  for (auto it = if_op_segment.begin(); it != if_op_segment.end(); ++it) {
    llvm::SmallVector<int, 4> indices_to_keep_vec =
        GetReturnIndicesToKeep(*it, if_op_segment);
    return_indices_vec.push_back(indices_to_keep_vec);
  }
  return return_indices_vec;
}

// Replace the internal usage in each pair of IfRegions from top to bottom for
// both then branch and else branch.
void ReplaceInternalUsage(llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment) {
  for (auto it = if_op_segment.begin(); it != if_op_segment.end(); ++it) {
    for (auto it2 = std::next(it); it2 != if_op_segment.end(); ++it2) {
      for (OpResult result : it->getResults()) {
        replaceAllUsesInRegionWith(
            result,
            it->getThenBranch().front().getTerminator()->getOperand(
                result.getResultNumber()),
            it2->getThenBranch());
        replaceAllUsesInRegionWith(
            result,
            it->getElseBranch().front().getTerminator()->getOperand(
                result.getResultNumber()),
            it2->getElseBranch());
      }
    }
  }
}

// Move ops in the `moved_ops_ordered` after `last_op`.
void MoveOpsAfter(Operation* last_op,
                  llvm::SmallVector<Operation*, 8>& moved_ops_ordered) {
  auto block = last_op->getBlock();
  absl::flat_hash_set<Operation*> all_moved_ops(moved_ops_ordered.begin(),
                                                moved_ops_ordered.end());
  moved_ops_ordered.clear();
  for (Operation& op : *block) {
    // There are no mutations in the loop. So each call of `isBeforeInBlock`
    // is O(1).
    if (all_moved_ops.count(&op) && op.isBeforeInBlock(last_op)) {
      moved_ops_ordered.push_back(&op);
    }
  }
  // Move ops in order.
  for (Operation* op : moved_ops_ordered) {
    op->moveAfter(last_op);
    last_op = op;
  }
}

// Replace all external usage for each IfRegion in the segment of IfRegions.
// `if_op_segment` refers to the segment of IfRegions, `new_if_op` refers to the
// new merged IfRegion, `return_indices` refers to the indices to be kept in new
// merged IfRegion.
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

// Update the moved op list to remove old IfRegions from the list and add new
// merged IfRegions. `old_to_new_IfRegions_map` refers to a map from old
// IfRegion to new merged IfRegion. `moved_ops_list` refers to the list of ops
// to be moved after new merged IfRegion.
void UpdateMovedOpList(
    llvm::SmallDenseMap<Operation*, TF::IfRegionOp>& old_to_new_IfRegion_map,
    llvm::SmallVector<Operation*, 8>& moved_ops_list) {
  llvm::SmallDenseSet<TF::IfRegionOp> new_if_ops;
  bool need_add_new_if_op = false;
  for (auto iter = moved_ops_list.begin(); iter != moved_ops_list.end();
       iter++) {
    if (old_to_new_IfRegion_map.count(*iter)) {
      need_add_new_if_op = true;
      auto new_if_op = old_to_new_IfRegion_map[*iter];
      new_if_ops.insert(new_if_op);
      moved_ops_list.erase(iter--);
    }
  }
  if (need_add_new_if_op) {
    for (auto& new_if_op : new_if_ops) {
      moved_ops_list.push_back(new_if_op.getOperation());
    }
  }
}

// Create the Yield ops for both branches with merged results.
// `builder` is the OpBuilder.
// `if_op_segment` refers to the segment of IfRegions to be merged.
// `return_indices` refers to the return indices to be kept in merged IfRegion
// `new_if_op` refers to the created new IfRegion
void CreateYieldOps(
    OpBuilder& builder, llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment,
    llvm::SmallVector<llvm::SmallVector<int, 4>>& return_indices,
    TF::IfRegionOp new_if_op, TF::IfRegionOp first_if) {
  llvm::SmallVector<Value, 4> merged_then_yield_values;
  for (const auto& index_and_value : llvm::enumerate(if_op_segment)) {
    auto if_op = index_and_value.value();
    for (auto i : return_indices[index_and_value.index()]) {
      merged_then_yield_values.push_back(
          if_op.getThenBranch().front().getTerminator()->getOperand(i));
    }
  }
  builder.setInsertionPointToEnd(&new_if_op.getThenBranch().front());
  builder.create<TF::YieldOp>(
      first_if.getThenBranch().front().getTerminator()->getLoc(),
      /*operands=*/merged_then_yield_values);

  llvm::SmallVector<Value, 4> merged_else_yield_values;
  for (const auto& index_and_value : llvm::enumerate(if_op_segment)) {
    auto if_op = index_and_value.value();
    for (auto i : return_indices[index_and_value.index()]) {
      merged_else_yield_values.push_back(
          if_op.getElseBranch().front().getTerminator()->getOperand(i));
    }
  }
  builder.setInsertionPointToEnd(&new_if_op.getElseBranch().front());
  builder.create<TF::YieldOp>(
      first_if.getElseBranch().front().getTerminator()->getLoc(),
      /*operands=*/merged_else_yield_values);
}

// Merge the IfRegions in each segment. In the meantime, the old IfRegions in
// the segment will be added to `regions_to_remove`. They will be erased in the
// end.
// `if_op_segment` refers to segments of IfRegions. `moved_op_list` refers to
// the ops to be moved after new merged IfRegion. `regions_to_remove` refers to
// the regions to be removed from the `moved_ops_list`.
// `old_to_new_IfRegion_map` refers to a map from old IfRegion to new merged
// IfRegion.
void MergeIfPerSegment(
    llvm::SmallVector<TF::IfRegionOp, 8>& if_op_segment,
    llvm::SmallVector<Operation*, 8>& moved_ops_list,
    llvm::SmallSetVector<TF::IfRegionOp, 8>& regions_to_remove,
    llvm::SmallDenseMap<Operation*, TF::IfRegionOp>& old_to_new_IfRegion_map) {
  TF::IfRegionOp first_if = if_op_segment[0];
  llvm::SmallVector<Type, 4> merged_return_types;
  llvm::SmallVector<TF::IfRegionOp, 8> sources_if_ops(
      std::next(if_op_segment.begin()), if_op_segment.end());

  // Create new IfRegion's merged results.
  auto return_indices = GetReturnIndicesVec(if_op_segment);
  for (const auto& index_and_value : llvm::enumerate(return_indices)) {
    TF::IfRegionOp if_op = if_op_segment[index_and_value.index()];
    for (auto i : index_and_value.value()) {
      merged_return_types.push_back(if_op.getResult(i).getType());
    }
  }

  // Create new IfRegion for merged all IfRegions in if_op_segmemt.
  OpBuilder builder(first_if);
  builder.setInsertionPoint(if_op_segment.back().getOperation());

  auto new_if_op = builder.create<TF::IfRegionOp>(
      first_if.getLoc(), merged_return_types, first_if.getCond(),
      llvm::all_of(if_op_segment,
                   [&](TF::IfRegionOp op) { return op.getIsStateless(); }),
      first_if.get_thenFuncNameAttr(), first_if.get_elseFuncNameAttr());
  new_if_op.getThenBranch().push_back(new Block);
  new_if_op.getElseBranch().push_back(new Block);

  // Replace internal usages of merged if ops.
  ReplaceInternalUsage(if_op_segment);

  // Replace external usages of merged if ops.
  ReplaceExternalUsage(if_op_segment, new_if_op, return_indices);

  // Move ops after the new merged If region.
  MoveOpsAfter(new_if_op.getOperation(), moved_ops_list);

  // Create the Yield ops for both branches with merged results.
  CreateYieldOps(builder, if_op_segment, return_indices, new_if_op, first_if);

  for (auto& old_if_op : if_op_segment) {
    MoveBranches(/*first_if=*/new_if_op, /*second_if=*/old_if_op);
  }

  for (auto& old_if_op : if_op_segment) {
    old_to_new_IfRegion_map[old_if_op.getOperation()] = new_if_op;
    regions_to_remove.insert(old_if_op);
  }
}

// Merge IfRegions for each IfRegion group. Each IfRegion group contains
// several segments of IfRegions and each segment of IfRegions can be merged
// into one IfRegion.
// `if_cond` refers to the if condition of the segments of IfRegions.
// `planned_merged_groups` refers to the groups of IfRegions to be merged
// `moved_ops_groups` refers to the ops need to be moved after new merged
// IfRegions associated with each segment of IfRegions.
// `regions_to_remove` refers to the regions to be removed
// `old_to_new_IfRegion_map` refers to a map from old IfRegion to new merged
// IfRegion.
void MergeIfPerIfGroups(
    const Value& if_cond, MapToRegionVec2D& planned_merged_groups,
    MapToOperationVec2D& moved_ops_groups,
    llvm::SmallSetVector<TF::IfRegionOp, 8>& regions_to_remove,
    llvm::SmallDenseMap<Operation*, TF::IfRegionOp>& old_to_new_IfRegion_map) {
  OperationVec2D& moved_ops_group = moved_ops_groups[if_cond];
  RegionVec2D& segments = planned_merged_groups[if_cond];

  for (auto i = 0; i < segments.size(); ++i) {
    if (segments[i].size() >= 2) {
      UpdateMovedOpList(old_to_new_IfRegion_map, moved_ops_group[i]);
      MergeIfPerSegment(segments[i], moved_ops_group[i], regions_to_remove,
                        old_to_new_IfRegion_map);
    }
  }
}

// Groups IfRegions by common predicate and attemps to merge them.
void OptimizeIfRegions(Block* block, ModuleOp module) {
  // Do side effect analysis only one time in the beginning
  auto side_effect_analysis = std::make_unique<TF::SideEffectAnalysis>(module);

  // Determine IfRegions with the same predicate.
  llvm::SmallDenseMap<Value, llvm::SmallVector<TF::IfRegionOp, 8>, 8>
      grouped_if_ops;
  llvm::SmallVector<Value, 4> if_cond_order;
  block->walk([&](TF::IfRegionOp if_op) {
    auto it = grouped_if_ops.try_emplace(if_op.getCond());
    if (it.second) {
      if_cond_order.push_back(if_op.getCond());
    }
    it.first->getSecond().push_back(if_op);
  });

  MapToRegionVec2D planned_merged_groups;
  MapToOperationVec2D moved_ops_groups;
  llvm::SmallSetVector<TF::IfRegionOp, 8> regions_to_remove;
  llvm::SmallDenseMap<Operation*, TF::IfRegionOp> old_to_new_IfRegion_map;

  // For each if group, determine the segments of each if groups
  // that can be merged and their related ops to be moved after
  // the new generated IfRegions
  // We cache the infomation into two maps:
  // planned_merged_groups and moved_ops_groups
  for (const auto& if_cond : if_cond_order) {
    GenerateSegmentsPerIfGroups(if_cond, grouped_if_ops[if_cond],
                                side_effect_analysis, planned_merged_groups,
                                moved_ops_groups);
  }

  // Merge IfRegions for each IfRegion groups.
  for (const auto& if_cond : if_cond_order) {
    MergeIfPerIfGroups(if_cond, planned_merged_groups, moved_ops_groups,
                       regions_to_remove, old_to_new_IfRegion_map);
  }

  // Remove all old IfRegions that already been merged.
  for (auto old_if_region : regions_to_remove) {
    old_if_region.erase();
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
