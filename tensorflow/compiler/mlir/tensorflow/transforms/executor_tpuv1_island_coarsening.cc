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

// This transformation pass takes TensorFlow executor dialect IslandOps and
// merges the one that contains operation marked to run on TPU.

#include <algorithm>
#include <iterator>
#include <queue>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/platform/logging.h"

#define DEBUG_TYPE "tf-executor-tpu-v1-island-coarsening"

namespace mlir {
namespace tf_executor {

namespace {

constexpr llvm::StringRef kTpuReplicateAttr = "_tpu_replicate";
constexpr llvm::StringRef kTpuStatusAttr = "_tpu_compilation_status";

// This pass is a variant of the island coarsening that is limited to
// TPU-annotated operations and intended to preserve backward compatibility with
// TFv1.
struct TpuV1BridgeExecutorIslandCoarsening
    : public PassWrapper<TpuV1BridgeExecutorIslandCoarsening,
                         OperationPass<ModuleOp>> {
  void runOnOperation() override;
};

// Sorts the operations in the provided range to enforce dominance.
// This is useful after fusing / reorganizing Operations in a block and later
// needing to readjust the ordering to ensure dominance.
LogicalResult SortTopologically(Block::iterator begin, Block::iterator end) {
  Block* block = begin->getBlock();
  // Either sort from `begin` to end of block or both `begin` and
  // `end` should belong to the same block.
  assert(end == block->end() ||
         end->getBlock() == block && "ops must be in the same block");

  // Track the ops that still need to be scheduled in a set.
  SmallPtrSet<Operation*, 16> unscheduled_ops;
  for (Operation& op : llvm::make_range(begin, end))
    unscheduled_ops.insert(&op);

  Block::iterator last_scheduled_op = begin;
  while (!unscheduled_ops.empty()) {
    bool scheduled_at_least_once = false;
    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (move it before the last_scheduled_op).
    for (Operation& op : llvm::make_range(last_scheduled_op, end)) {
      WalkResult ready_to_schedule = op.walk([&](Operation* nested_op) {
        for (Value operand : nested_op->getOperands()) {
          Operation* defining_op = operand.getDefiningOp();
          if (!defining_op) continue;
          Operation* producer_in_block =
              block->findAncestorOpInBlock(*defining_op);
          if (producer_in_block && producer_in_block != &op &&
              unscheduled_ops.count(producer_in_block)) {
            // Found an operand that isn't scheduled yet, interrupt the walk.
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (ready_to_schedule.wasInterrupted()) continue;
      unscheduled_ops.erase(&op);
      if (Block::iterator(op) != last_scheduled_op)
        op.moveBefore(block, last_scheduled_op);
      else
        ++last_scheduled_op;
      scheduled_at_least_once = true;
    }
    if (!scheduled_at_least_once) return failure();
  }
  return success();
}

// Looks for an IslandOp that wraps a single operation tagged with the
// _tpu_replicate attribute, and merges it with all the following operations in
// the block. Sets the `changed` boolean to true if any island is merged.
// Returns a failure if a cycle prevents the merge from happening correctly
// without breaking dominance. The IR is left in invalid state in case of
// failure.
LogicalResult MergeIsland(llvm::function_ref<bool(StringAttr, Operation*)>
                              is_op_calling_func_for_cluster,
                          Operation* op, bool* changed) {
  // Find the first island wrapping a single operation with the `_tpu_replicate`
  // attribute, it'll be used as the root of the algorithm to find the other
  // operations that are part of the same cluster.
  IslandOp island = dyn_cast<IslandOp>(*op);
  if (!island || !island.WrapsSingleOp()) return success();
  Operation& wrapped_op = island.GetBody().front();

  // TODO(b/188046643): Conservatively fail until pass is extended to fuse
  // chains of these ops.
  if (isa<TF::TPUPartitionedInputOp, TF::TPUPartitionedOutputOp>(wrapped_op)) {
    return failure();
  }

  StringAttr cluster_name =
      wrapped_op.getAttrOfType<StringAttr>(kTpuReplicateAttr);
  if (!cluster_name)
    cluster_name = wrapped_op.getAttrOfType<StringAttr>(kTpuStatusAttr);
  if (!cluster_name) return success();

  // We found a _tpu_replicate, let's build an island for the full cluster!
  LLVM_DEBUG(llvm::dbgs() << "Processing candidate island: "
                          << *island.getOperation() << "\n");

  // Collect the islands to merge together in this new cluster starting with the
  // given island.
  SmallVector<IslandOp, 16> islands;
  SmallPtrSet<Operation*, 16> wrapped_ops;
  for (Operation& candidate_op : llvm::make_early_inc_range(
           llvm::make_range(op->getIterator(), op->getBlock()->end()))) {
    IslandOp candidate_island = dyn_cast<IslandOp>(candidate_op);
    if (!candidate_island || !candidate_island.WrapsSingleOp()) continue;
    // Check if we have an operation with the expected attribute.
    Operation& candidate_wrapped_op = candidate_island.GetBody().front();

    // TODO(b/188046643): Conservatively fail until pass is extended to fuse
    // chains of these ops.
    if (isa<TF::TPUPartitionedInputOp, TF::TPUPartitionedOutputOp>(
            candidate_wrapped_op)) {
      return failure();
    }

    StringAttr candidate_cluster_name =
        candidate_wrapped_op.getAttrOfType<StringAttr>(kTpuReplicateAttr);
    if (!candidate_cluster_name)
      candidate_cluster_name =
          candidate_wrapped_op.getAttrOfType<StringAttr>(kTpuStatusAttr);
    if (candidate_cluster_name != cluster_name &&
        !is_op_calling_func_for_cluster(cluster_name, &candidate_wrapped_op))
      continue;

    // Look at captured operands to bring-in ReplicatedInputOp in the
    // island as well. Consider pulling in tf.Const, some optimizations can
    // benefit from this.
    for (Value operand : candidate_wrapped_op.getOperands()) {
      IslandOp wrapper = dyn_cast_or_null<IslandOp>(operand.getDefiningOp());
      if (!wrapper || !wrapper.WrapsSingleOp()) continue;
      Operation& wrapped_op = wrapper.GetBody().front();
      if (!isa<TF::TPUReplicatedInputOp>(wrapped_op)) continue;
      if (wrapped_ops.count(&wrapped_op)) continue;
      wrapped_ops.insert(&wrapped_op);
      islands.push_back(wrapper);
    }
    islands.push_back(candidate_island);
    wrapped_ops.insert(&candidate_wrapped_op);

    // Look at results to bring-in ReplicatedOutputOp in the island as well.
    for (Value result : candidate_island.getResults()) {
      for (OpOperand use : result.getUsers()) {
        Operation* user = use.getOwner();
        if (!isa<TF::TPUReplicatedOutputOp>(user)) continue;
        assert(!wrapped_ops.count(user) &&
               "unexpected already processed TPUReplicatedOutputOp");
        wrapped_ops.insert(user);
        islands.push_back(cast<IslandOp>(user->getParentOp()));
      }
    }
  }

  // If no other island was found to merge with the existing one, just
  // move on.
  if (islands.size() <= 1) return success();

  *changed = true;
  auto first_op_after =
      std::next(Block::iterator(islands.back().getOperation()));

  // Compute the result of the merged island, these are the values produced by
  // the islands that are merged if they have a use in an island not merged,
  // i.e. a value that escapes.
  llvm::SmallVector<Type, 4> result_types;
  for (IslandOp new_op : islands) {
    for (Value result : new_op.outputs()) {
      if (llvm::any_of(result.getUsers(), [&](OpOperand user) {
            return !wrapped_ops.count(user.getOwner());
          }))
        result_types.push_back(result.getType());
    }
  }

  IslandOp new_island = OpBuilder(island).create<IslandOp>(
      island.getLoc(), result_types,
      /*control=*/ControlType::get(island.getContext()),
      /*controlInputs=*/island.getOperands());
  new_island.body().push_back(new Block);

  // Move the operations in the new island, gather the results of the new yield.
  Block& island_body = new_island.GetBody();
  SmallVector<Value, 16> yield_operands;
  for (IslandOp island : islands) {
    Operation& wrapped_op = island.GetBody().front();
    wrapped_op.moveBefore(&island_body, island_body.end());

    // For every result of the wrapped_op, it needs to get passed to the yield
    // operation, only if it escapes the island.
    for (auto result : llvm::zip(island.outputs(), wrapped_op.getResults())) {
      if (llvm::any_of(std::get<0>(result).getUsers(), [&](OpOperand user) {
            return !wrapped_ops.count(user.getOwner());
          }))
        yield_operands.push_back(std::get<1>(result));
    }
  }
  OpBuilder::atBlockEnd(&island_body)
      .create<YieldOp>(new_island.getLoc(), yield_operands);

  // remap results of the new islands to the user outside of the island.
  int current_result = 0;
  Value control = new_island.control();
  for (IslandOp island : islands) {
    YieldOp yield_op = island.GetYield();
    for (auto idx_result : llvm::enumerate(island.outputs())) {
      Value result = idx_result.value();

      bool has_external_use = false;
      for (OpOperand& use : llvm::make_early_inc_range(result.getUses())) {
        if (wrapped_ops.count(use.getOwner()))
          use.set(yield_op.getOperand(idx_result.index()));
        else
          has_external_use = true;
      }
      if (has_external_use) {
        result.replaceAllUsesWith(new_island.getResult(current_result));
        ++current_result;
      }
    }
    island.control().replaceAllUsesWith(control);
    island.erase();
  }

  // Ensure dominance by sorting the range of islands that were merged.
  return SortTopologically(Block::iterator(new_island.getOperation()),
                           first_op_after);
}

void TpuV1BridgeExecutorIslandCoarsening::runOnOperation() {
  SymbolTable symbol_table(getOperation());

  // Map tpu cluster names to the functions that contain operations for this
  // cluster.
  DenseMap<StringRef, DenseSet<FuncOp>> tpu_funcs;
  for (FuncOp func_op : getOperation().getOps<FuncOp>()) {
    func_op.walk([&](Operation* op) {
      StringAttr cluster_name =
          op->getAttrOfType<StringAttr>(kTpuReplicateAttr);
      if (!cluster_name)
        cluster_name = op->getAttrOfType<StringAttr>(kTpuStatusAttr);
      if (!cluster_name) return;
      tpu_funcs[cluster_name.getValue()].insert(func_op);
    });
  }

  // Return true if the operation is containing a reference to a function
  // containing operations for this cluster.
  auto is_op_calling_func_for_cluster = [&](StringAttr cluster, Operation* op) {
    auto funcs_for_cluster = tpu_funcs.find(cluster.getValue());
    assert(funcs_for_cluster != tpu_funcs.end());
    assert(!funcs_for_cluster->second.empty());
    if (funcs_for_cluster->second.size() == 1) return false;
    for (NamedAttribute attr : op->getAttrs()) {
      auto symbol_ref = attr.second.dyn_cast<FlatSymbolRefAttr>();
      if (!symbol_ref) continue;
      FuncOp callee = symbol_table.lookup<FuncOp>(symbol_ref.getValue());
      if (!callee) continue;
      if (funcs_for_cluster->second.count(callee)) return true;
    }
    return false;
  };

  for (FuncOp func_op : getOperation().getOps<FuncOp>()) {
    func_op.walk([&](GraphOp graph) {
      Block& graph_body = graph.GetBody();

      // Iterate until fixed point on the block, as it may contain multiple
      // clusters.
      bool changed = true;
      while (changed) {
        changed = false;
        for (Operation& op : graph_body) {
          if (failed(
                  MergeIsland(is_op_calling_func_for_cluster, &op, &changed))) {
            graph.emitError()
                << "Merging island failed: the TPU cluster likely "
                << "contains a cycle with non-TPU operations or has "
                   "unsupported ops\n";
            signalPassFailure();
            return WalkResult::interrupt();
          }
          // If islands were merged, restart scanning the block from the
          // beginning as we lost track of where to continue.
          if (changed) break;
        }
      }
      return WalkResult::advance();
    });
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
CreateTFExecutorTPUV1IslandCoarseningPass() {
  return std::make_unique<TpuV1BridgeExecutorIslandCoarsening>();
}

static PassRegistration<TpuV1BridgeExecutorIslandCoarsening> tpu_pass(
    "tf-executor-tpu-v1-island-coarsening",
    "Merges TPU clusters IslandOps, intended for V1 compatibility mode");

}  // namespace tf_executor
}  // namespace mlir
