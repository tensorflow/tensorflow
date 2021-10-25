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

#include <memory>
#include <queue>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
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
                 const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  // IfRegion ops must be in the same block.
  if (source.getOperation()->getBlock() !=
      destination.getOperation()->getBlock())
    return false;
  assert(destination.getOperation()->isBeforeInBlock(source.getOperation()));

  llvm::SmallSetVector<Operation*, 4> source_ops = GetAllOpsFromIf(source);
  llvm::SmallSetVector<Operation*, 4> destination_ops =
      GetAllOpsFromIf(destination);

  // If there is an intermediate data or side effect dependency between the
  // ops in destination and the ops in the source, it's not safe to merge
  // them.
  std::vector<Operation*> dependencies;
  for (auto* user : destination.getOperation()->getUsers()) {
    if (!source_ops.contains(user)) dependencies.push_back(user);
  }
  for (auto* successor : side_effect_analysis.DirectControlSuccessors(
           destination.getOperation())) {
    if (!source_ops.contains(successor)) dependencies.push_back(successor);
  }
  for (Operation& op : destination.then_branch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!source_ops.contains(successor) &&
          !destination_ops.contains(successor))
        dependencies.push_back(successor);
    }
  }
  for (Operation& op : destination.else_branch().front()) {
    for (auto* successor : side_effect_analysis.DirectControlSuccessors(&op)) {
      if (!source_ops.contains(successor) &&
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
      if (source_ops.contains(user)) {
        safe_to_merge = false;
        break;
      } else {
        dependencies.push_back(user);
      }
    }
    for (auto* successor :
         side_effect_analysis.DirectControlSuccessors(dependency)) {
      if (source_ops.contains(successor)) {
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

// Move all ops that depends on the results from `result_op` after `after_op`.
void MoveResultsAfter(
    Operation* result_op, Operation* after_op,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  std::queue<Operation*> queue;

  auto enqueue_deps = [&](Operation* source_op) {
    for (Operation* user : source_op->getUsers()) {
      queue.push(user);
    }
    source_op->walk([&](Operation* walked_op) {
      for (Operation* successor :
           side_effect_analysis.DirectControlSuccessors(walked_op)) {
        if (!source_op->isProperAncestor(successor)) queue.push(successor);
      }
    });
  };
  enqueue_deps(result_op);

  while (!queue.empty()) {
    auto* op = queue.front();
    queue.pop();
    while (op->getBlock() != after_op->getBlock()) op = op->getParentOp();
    if (op->isBeforeInBlock(after_op)) {
      op->moveAfter(after_op);
      after_op = op;
      enqueue_deps(op);
    }
  }
}

TF::IfRegionOp CreateMergedIf(
    ArrayRef<int> source_return_indices_to_keep,
    ArrayRef<int> destination_return_indices_to_keep, TF::IfRegionOp source,
    TF::IfRegionOp destination,
    const TF::SideEffectAnalysis::Info& side_effect_analysis) {
  llvm::SmallVector<Type, 4> merged_return_types;
  for (int i : destination_return_indices_to_keep)
    merged_return_types.push_back(destination.getResult(i).getType());
  for (int i : source_return_indices_to_keep)
    merged_return_types.push_back(source.getResult(i).getType());

  OpBuilder builder(destination);
  // Create new IfRegion with correct merged results.
  builder.setInsertionPoint(source.getOperation());

  auto new_if_op = builder.create<TF::IfRegionOp>(
      destination.getLoc(), merged_return_types, destination.cond(),
      destination.is_stateless() && source.is_stateless(),
      destination._then_func_nameAttr(), destination._else_func_nameAttr());
  new_if_op.then_branch().push_back(new Block);
  new_if_op.else_branch().push_back(new Block);
  // Replace internal usages of merged if ops.
  for (OpResult result : destination.getResults()) {
    replaceAllUsesInRegionWith(
        result,
        destination.then_branch().front().getTerminator()->getOperand(
            result.getResultNumber()),
        source.then_branch());
    replaceAllUsesInRegionWith(
        result,
        destination.else_branch().front().getTerminator()->getOperand(
            result.getResultNumber()),
        source.else_branch());
  }

  MoveResultsAfter(destination.getOperation(), new_if_op.getOperation(),
                   side_effect_analysis);

  // Replace external usages of merged if ops.
  int new_return_index = 0;
  for (int i : destination_return_indices_to_keep) {
    destination.getResult(i).replaceAllUsesWith(
        new_if_op.getResult(new_return_index++));
  }
  for (int i : source_return_indices_to_keep) {
    source.getResult(i).replaceAllUsesWith(
        new_if_op.getResult(new_return_index++));
  }

  // Create the Yield ops for both branches with merged results.
  llvm::SmallVector<Value, 4> merged_then_yield_values;
  for (int i : destination_return_indices_to_keep)
    merged_then_yield_values.push_back(
        destination.then_branch().front().getTerminator()->getOperand(i));
  for (int i : source_return_indices_to_keep)
    merged_then_yield_values.push_back(
        source.then_branch().front().getTerminator()->getOperand(i));
  builder.setInsertionPointToEnd(&new_if_op.then_branch().front());
  builder.create<TF::YieldOp>(
      destination.then_branch().front().getTerminator()->getLoc(),
      /*operands=*/merged_then_yield_values);

  llvm::SmallVector<Value, 4> merged_else_yield_values;
  for (int i : destination_return_indices_to_keep)
    merged_else_yield_values.push_back(
        destination.else_branch().front().getTerminator()->getOperand(i));
  for (int i : source_return_indices_to_keep)
    merged_else_yield_values.push_back(
        source.else_branch().front().getTerminator()->getOperand(i));
  builder.setInsertionPointToEnd(&new_if_op.else_branch().front());
  builder.create<TF::YieldOp>(
      destination.else_branch().front().getTerminator()->getLoc(),
      /*operands=*/merged_else_yield_values);

  // Merge the two branch regions from both IfRegionOps into new IfRegionOp.
  MoveBranches(/*source=*/destination, /*destination=*/new_if_op);
  destination.erase();
  MoveBranches(/*source=*/source, /*destination=*/new_if_op);
  source.erase();
  return new_if_op;
}

// Groups if regions by common predicate and attemps to merge them.
void OptimizeIfRegions(Block* block, ModuleOp module) {
  // Determine IfRegions with the same predicate.
  llvm::SmallDenseMap<Value, llvm::SmallVector<TF::IfRegionOp, 8>, 8>
      grouped_if_ops;
  block->walk([&](TF::IfRegionOp if_op) {
    auto it = grouped_if_ops.try_emplace(if_op.cond());
    it.first->getSecond().push_back(if_op);
  });

  auto side_effect_analysis = std::make_unique<TF::SideEffectAnalysis>(module);

  for (auto& entry : grouped_if_ops) {
    auto& if_ops = entry.second;
    for (auto it = if_ops.begin(); it != if_ops.end(); ++it) {
      TF::IfRegionOp first_if_op = *it;
      for (auto it2 = std::next(it); it2 != if_ops.end(); ++it2) {
        FuncOp func = first_if_op->getParentOfType<FuncOp>();
        const TF::SideEffectAnalysis::Info& analysis =
            side_effect_analysis->GetAnalysisForFunc(func);

        TF::IfRegionOp second_if_op = *it2;
        if (!SafeToMerge(second_if_op, first_if_op, analysis)) break;

        // For both check if there are uses outside of IfRegion, keep these as
        // part of the return and replace the internal uses.
        auto first_return_indices_to_keep =
            GetReturnIndicesToKeep(first_if_op, second_if_op);
        auto second_return_indices_to_keep =
            GetReturnIndicesToKeep(second_if_op, first_if_op);

        auto new_if_op = CreateMergedIf(second_return_indices_to_keep,
                                        first_return_indices_to_keep,
                                        second_if_op, first_if_op, analysis);

        if_ops.erase(it2--);
        first_if_op = new_if_op;
        // We regenerate the side effect analysis since merging the IfRegions
        // invalidates the side effect analysis.  This approach is O(N*M) where
        // N is the number of ops in `module` and M is the number of pairs of
        // IfRegion ops that are merged.
        side_effect_analysis = std::make_unique<TF::SideEffectAnalysis>(module);
      }
    }
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
