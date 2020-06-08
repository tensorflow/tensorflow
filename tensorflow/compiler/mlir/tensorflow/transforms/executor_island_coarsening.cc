/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
// merges them. Note, this currently does not handle TensorFlow V1 style control
// flow/frames or side effecting ops yet.

#include <iterator>
#include <tuple>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace tf_executor {

namespace {

// IslandType is an enum representing if an island is the island (parent)
// merging another island or is the island (child) being being merged.
enum IslandType { kParentIsland, kChildIsland };

// IslandResult is a helper struct holding an islands result and associated
// inner op result.
struct IslandResult {
  IslandResult(Value inner_op_result, Value island_result)
      : inner_op_result(inner_op_result), island_result(island_result) {}

  Value inner_op_result;
  Value island_result;
};

struct ExecutorIslandCoarsening
    : public PassWrapper<ExecutorIslandCoarsening, FunctionPass> {
  void runOnFunction() override;
};

// Finds the operation leading to an island that the island can be merged with.
// This looks for the operation, either control input or data input to an op,
// that is closest to the island in the graph. If no candidate can be found or
// the op found is not an island, an empty optional is returned.
llvm::Optional<IslandOp> GetOperandCandidateToMergeWith(IslandOp island) {
  Operation* graph_op = island.getParentOp();
  Operation* candidate = nullptr;

  // Check island control operands.
  for (Value input : island.controlInputs()) {
    Operation* def = input.getDefiningOp();
    DCHECK_EQ(def->getParentOp(), graph_op);
    if (!candidate || candidate->isBeforeInBlock(def)) candidate = def;
  }

  // Check island data operands.
  island.walk([graph_op, &candidate](Operation* op) {
    for (Value input : op->getOperands()) {
      Operation* def = input.getDefiningOp();
      if (!def || def->getParentOp() != graph_op) continue;
      if (!candidate || candidate->isBeforeInBlock(def)) candidate = def;
    }
  });

  if (!candidate || !llvm::isa<IslandOp>(candidate)) return llvm::None;

  return llvm::Optional<IslandOp>(llvm::cast<IslandOp>(candidate));
}

// Finds the operation leading from an island that the island can be merged
// with. This looks for the operation, either control result or data result to
// an op, that is closest to the island in the graph. If no candidate can be
// found or the op found is not an island, an empty optional is returned.
llvm::Optional<IslandOp> GetResultCandidateToMergeWith(IslandOp island) {
  Operation* graph_op = island.getParentOp();
  Operation* candidate = nullptr;

  // Check island control results.
  for (Operation* user : island.control().getUsers()) {
    DCHECK_EQ(user->getParentOp(), graph_op);
    if (!candidate || user->isBeforeInBlock(candidate)) candidate = user;
  }

  // Check island data results.
  Block& graph_body = llvm::cast<GraphOp>(graph_op).GetBody();
  for (Value result : island.outputs()) {
    for (Operation* user : result.getUsers()) {
      Operation* def = graph_body.findAncestorOpInBlock(*user);
      DCHECK_NE(def, nullptr);
      if (!candidate || def->isBeforeInBlock(candidate)) candidate = def;
    }
  }

  if (!candidate || !llvm::isa<IslandOp>(candidate)) return llvm::None;

  return llvm::Optional<IslandOp>(llvm::cast<IslandOp>(candidate));
}

// Collects the operands for the new island by collecting all control inputs of
// the islands being merged.
llvm::SmallSetVector<Value, 8> GetNewIslandOperands(IslandOp parent,
                                                    IslandOp child) {
  llvm::SmallSetVector<Value, 8> operands;
  operands.insert(parent.getOperands().begin(), parent.getOperands().end());
  operands.insert(child.getOperands().begin(), child.getOperands().end());
  operands.remove(parent.control());
  return operands;
}

// Collects the results for the new island by going through each data result of
// the islands being merged. Unused results outside of the merged island to be
// formed are pruned. If the child island inner ops consume the parent island
// control result, the child island inner ops will have that respective control
// input pruned. Results of the parent island that are consumed by the child
// island are replaced by the respective inner ops result from the parent
// island.
llvm::SmallVector<IslandResult, 8> GetNewIslandResultsAndForwardResults(
    IslandOp parent, IslandOp child) {
  llvm::SmallVector<IslandResult, 8> results;

  Block& child_body = child.GetBody();
  for (auto ret_vals :
       llvm::zip(parent.GetYield().getOperands(), parent.outputs())) {
    bool result_captured = false;
    Value inner_op_result = std::get<0>(ret_vals);
    Value island_result = std::get<1>(ret_vals);
    for (auto& use : llvm::make_early_inc_range(island_result.getUses())) {
      if (child_body.findAncestorOpInBlock(*use.getOwner())) {
        // Forward result from inner op.
        use.set(inner_op_result);
      } else if (!result_captured) {
        results.emplace_back(inner_op_result, island_result);
        result_captured = true;
      }
    }
  }

  for (auto ret_vals :
       llvm::zip(child.GetYield().getOperands(), child.outputs())) {
    Value inner_op_result = std::get<0>(ret_vals);
    Value island_result = std::get<1>(ret_vals);
    if (!island_result.use_empty()) {
      results.emplace_back(inner_op_result, island_result);
    }
  }

  return results;
}

// Creates the new merged island.
IslandOp CreateNewIsland(IslandOp parent, IslandOp child,
                         IslandType insert_position,
                         llvm::ArrayRef<Value> operands,
                         llvm::ArrayRef<IslandResult> results) {
  // Collect types from results.
  llvm::SmallVector<Type, 8> result_types;
  for (const auto& result : results)
    result_types.push_back(result.inner_op_result.getType());

  // IslandOps always have a control result.
  result_types.push_back(ControlType::get(parent.getContext()));

  Operation* old_island = insert_position == kParentIsland ? parent : child;
  OpBuilder builder(old_island);
  auto new_island = builder.create<IslandOp>(
      old_island->getLoc(), result_types, operands, ArrayRef<NamedAttribute>{});
  new_island.body().push_back(new Block);
  return new_island;
}

// Creates respective YieldOp for the new merged island.
YieldOp CreateNewIslandYieldOp(IslandOp new_island,
                               llvm::ArrayRef<IslandResult> results) {
  llvm::SmallVector<Value, 8> yield_operands;
  yield_operands.reserve(results.size());

  for (auto ret_vals : llvm::zip(results, new_island.outputs())) {
    const auto& old_result = std::get<0>(ret_vals);

    // Replace original island result with new island result.
    old_result.island_result.replaceAllUsesWith(std::get<1>(ret_vals));

    // Add associated inner op result to operands of the YieldOp.
    yield_operands.push_back(old_result.inner_op_result);
  }

  // Create YieldOp for the new island.
  OpBuilder builder(&new_island.GetBody(), new_island.GetBody().end());
  return builder.create<YieldOp>(new_island.getLoc(), yield_operands);
}

// Moves inner ops (excluding last op/YieldOp) from islands being merged into
// the new merged island.
void MoveInnerOpsToNewIsland(IslandOp parent, IslandOp child,
                             Operation* new_yield_op) {
  Block* block = new_yield_op->getBlock();

  auto move_inner_ops = [block, new_yield_op](IslandOp island) {
    auto& island_body = island.GetBody().getOperations();
    block->getOperations().splice(new_yield_op->getIterator(), island_body,
                                  island_body.begin(),
                                  std::prev(island_body.end()));
  };

  move_inner_ops(parent);
  move_inner_ops(child);
}

// Merges two islands and places new merged island before parent or child.
void MergeIslands(IslandOp parent, IslandOp child, IslandType insert_position) {
  // Collect operands for the new merged island.
  llvm::SmallSetVector<Value, 8> operands = GetNewIslandOperands(parent, child);

  // Collect results for the new merged island.
  llvm::SmallVector<IslandResult, 8> results =
      GetNewIslandResultsAndForwardResults(parent, child);

  // Create the new merged island.
  IslandOp new_island = CreateNewIsland(parent, child, insert_position,
                                        operands.getArrayRef(), results);

  // Create associated YieldOp for the new merged island.
  YieldOp new_yield_op = CreateNewIslandYieldOp(new_island, results);

  // Move inner ops from original islands into the new island.
  MoveInnerOpsToNewIsland(parent, child, new_yield_op.getOperation());

  // Update control inputs to point to the new merged island.
  child.control().replaceAllUsesWith(new_island.control());
  parent.control().replaceAllUsesWith(new_island.control());

  // Remove merged islands.
  child.erase();
  parent.erase();
}

// Merges island with the operand closest to the island in the graph. The
// operand must be another IslandOp for merging to take place. A new island is
// created and the islands being merged are removed if a merge took place.
// Returns true if the island was merged with its operand.
bool MergeIslandWithOperand(IslandOp child) {
  // Find candidate operand to merge island with.
  llvm::Optional<IslandOp> candidate = GetOperandCandidateToMergeWith(child);
  if (!candidate.hasValue()) return false;
  MergeIslands(candidate.getValue(), child, kParentIsland);
  return true;
}

// Merges island with the result closest to the island in the graph. The result
// must be another IslandOp for merging to take place. A new island is created
// and the islands being merged are removed if a merge took place. Returns true
// if the island was merged with its result.
bool MergeIslandWithResult(IslandOp parent) {
  // Find candidate result to merge island with.
  llvm::Optional<IslandOp> candidate = GetResultCandidateToMergeWith(parent);
  if (!candidate.hasValue()) return false;
  MergeIslands(parent, candidate.getValue(), kChildIsland);
  return true;
}

// Takes the inputs to tf_executor.fetch, make a new island that just yields
// them, and replace the fetch's input operands with the new yielded values.
//
// This allows our def-use based island coarsening algorithm to merge
// islands that independently feed into a fetch.
void InsertDummyIslandForFetch(FetchOp fetch) {
  llvm::SmallVector<Value, 4> data_fetches;
  llvm::SmallVector<Type, 4> data_types;
  llvm::SmallVector<Value, 4> control_fetches;
  for (auto value : fetch.fetches()) {
    if (value.getType().isa<ControlType>()) {
      control_fetches.push_back(value);
    } else {
      data_fetches.push_back(value);
      data_types.push_back(value.getType());
    }
  }
  auto island = OpBuilder(fetch).create<IslandOp>(
      fetch.getLoc(), data_types,
      /*control=*/ControlType::get(fetch.getContext()),
      /*controlInputs=*/control_fetches);
  island.body().push_back(new Block);
  OpBuilder::atBlockEnd(&island.GetBody())
      .create<YieldOp>(fetch.getLoc(), data_fetches);
  const int fetch_control_idx = data_fetches.size();
  for (int i = 0, e = fetch.getNumOperands(); i < e; i++) {
    // The fetch could have multiple control operands (all at the end of its
    // operand list). We replace them all with the island's single control
    // operand.
    if (i <= fetch_control_idx) {
      fetch.setOperand(i, island.getResult(i));
    } else {
      fetch.getOperation()->eraseOperand(fetch.getNumOperands() - 1);
    }
  }
}

void ExecutorIslandCoarsening::runOnFunction() {
  getFunction().walk([](GraphOp graph) {
    InsertDummyIslandForFetch(graph.GetFetch());

    Block& graph_body = graph.GetBody();

    bool updated = false;
    do {
      updated = false;

      auto reversed = llvm::reverse(graph_body);
      for (Operation& operation : llvm::make_early_inc_range(reversed)) {
        auto island = llvm::dyn_cast<IslandOp>(operation);
        if (!island) continue;
        updated |= MergeIslandWithResult(island);
      }

      for (Operation& operation : llvm::make_early_inc_range(graph_body)) {
        auto island = llvm::dyn_cast<IslandOp>(operation);
        if (!island) continue;
        updated |= MergeIslandWithOperand(island);
      }
    } while (updated);
  });
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> CreateTFExecutorIslandCoarseningPass() {
  return std::make_unique<ExecutorIslandCoarsening>();
}

static PassRegistration<ExecutorIslandCoarsening> pass(
    "tf-executor-island-coarsening",
    "Merges TensorFlow executor dialect IslandOps");

}  // namespace tf_executor
}  // namespace mlir
