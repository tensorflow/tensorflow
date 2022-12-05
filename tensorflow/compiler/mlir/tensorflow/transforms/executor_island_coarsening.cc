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

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

// This structure represents a merged island. It includes all of the islands
// that can be merged together and the point of insertion of the merged island.
struct MergedIsland {
  // Construct a new island from the given root.
  explicit MergedIsland(IslandOp root) : insert_point(root) {
    islands.push_back(root);
  }

  // The insertion point anchor of the merged island, or where the merged island
  // will be inserted when created.
  Operation* const insert_point;

  // The set of islands that are to be merged together.
  SmallVector<IslandOp> islands;
};

// This structure contains all of the merge decisions for islands within a
// graph. We compute which islands to merge first, so that we don't need to
// recursively mutate the IR (resulting in quadratic behavior when moving
// operations). A rough sketch of the coarsening algorithm is shown below:
//
// // The algorithm iterates until a fixpoint is reached, i.e. when no more
// // islands can be merged.
// while (changed) {
//   // In the first phase we try to merge islands with their nearest consumer
//   // iff the consumer is another island.
//   // Note: A consumer is an operation that consumes one of our outputs.
//   changed |= tryMergedIslandsIntoNearestConsumer();
//
//   // In the second phase we try to merge islands with their nearest producer
//   // of a value they consume, iff the producer is another island.
//   // Note: A producer is an operation that produces one of our inputs.
//   changed |= tryMergedIslandsIntoNearestProducer();
// }
//
class CoarseningAnalysis {
 public:
  // Compute the coarsening analysis over the given graph.
  explicit CoarseningAnalysis(GraphOp graph);

  // Returns a list of all of the mergable islands found in the graph.
  iterator_range<
      llvm::filter_iterator<SmallVector<MergedIsland>::const_iterator,
                            function_ref<bool(const MergedIsland&)>>>
  GetMergableIslands() const {
    function_ref<bool(const MergedIsland&)> filter_fn =
        [](const MergedIsland& merged_island) {
          return merged_island.islands.size() > 1;
        };
    return llvm::make_filter_range(merged_islands_, filter_fn);
  }

 private:
  // Attempt to find an island group that produces a value consumed by one of
  // the islands (or operation therein) within the given `merged_island`. If no
  // candidate can be found, returns nullptr.
  MergedIsland* GetOperandCandidateToMergeWith(GraphOp graph,
                                               MergedIsland& merged_island);

  // Attempt to find an island group that consumes a result, either control or
  // data, from one of the islands in the given `merged_island`. If no candidate
  // can be found, returns nullptr.
  MergedIsland* GetResultCandidateToMergeWith(GraphOp graph,
                                              MergedIsland& merged_island);

  // All of the merged islands in the graph.
  SmallVector<MergedIsland> merged_islands_;
  // A mapping from an island operation to the current merged island group it
  // is a part of.
  DenseMap<Operation*, MergedIsland*> island_to_merged_island_;
};

CoarseningAnalysis::CoarseningAnalysis(GraphOp graph) {
  // As an initial step, construct a merged island for each island in the
  // graph.
  for (IslandOp island : graph.SingleBlock::getBody()->getOps<IslandOp>())
    merged_islands_.push_back(MergedIsland(island));

  // Record the mapping from the island to the merge group as a secondary step,
  // as we are taking the address of the islands here and the push_back step
  // above may invalidate previously inserted islands mid-loop.
  for (MergedIsland& island : merged_islands_)
    island_to_merged_island_.try_emplace(island.insert_point, &island);

  // This functor merges the given `old_merged_island` into the
  // `new_merged_island`. `merge_in_front` is whether the old island should be
  // merged into the front of the new island, or the back.
  auto merge_islands = [&](MergedIsland& old_merged_island,
                           MergedIsland& new_merged_island,
                           bool merge_in_front) {
    for (IslandOp island : old_merged_island.islands)
      island_to_merged_island_[island] = &new_merged_island;

    auto insert_point = merge_in_front ? new_merged_island.islands.begin()
                                       : new_merged_island.islands.end();
    new_merged_island.islands.insert(insert_point,
                                     old_merged_island.islands.begin(),
                                     old_merged_island.islands.end());
    old_merged_island.islands.clear();
  };

  // Iterate over all of the island groups attempting to merge as many islands
  // groups as possible.
  bool updated = false;
  do {
    updated = false;

    // Attempt to merge an island into an island consuming one of its results.
    for (MergedIsland& merged_island : llvm::reverse(merged_islands_)) {
      if (merged_island.islands.empty()) continue;

      MergedIsland* candidate =
          GetResultCandidateToMergeWith(graph, merged_island);
      if (candidate) {
        merge_islands(merged_island, *candidate, /*merge_in_front=*/true);
        updated = true;
      }
    }

    // Attempt to merge an island into an island producing one of its operands.
    for (MergedIsland& merged_island : merged_islands_) {
      if (merged_island.islands.empty()) continue;

      MergedIsland* candidate =
          GetOperandCandidateToMergeWith(graph, merged_island);
      if (candidate) {
        merge_islands(merged_island, *candidate, /*merge_in_front=*/false);
        updated = true;
      }
    }
  } while (updated);
}

MergedIsland* CoarseningAnalysis::GetOperandCandidateToMergeWith(
    GraphOp graph, MergedIsland& merged_island) {
  // The candidate operation to consider merging the current island group with.
  Operation* candidate = nullptr;
  // The island group of the current candidate if it is an IslandOp, nullptr
  // otherwise.
  MergedIsland* candidate_island = nullptr;

  // Given an input operation, try to replace the current candidate operation
  // with it.
  auto try_update_current_candidate = [&](Operation* rhs) {
    MergedIsland* rhs_island = nullptr;
    // Check if this is an island operation we can merge with.
    auto rhs_it = island_to_merged_island_.find(rhs);
    if (rhs_it != island_to_merged_island_.end()) {
      rhs_island = rhs_it->second;

      // Ignore islands that are already a part of the current island group.
      if (rhs_island == &merged_island) return;

      rhs = rhs_island->insert_point;
    }
    if (!candidate || candidate->isBeforeInBlock(rhs)) {
      candidate = rhs;
      candidate_island = rhs_island;
    }
  };

  // Check island control operands.
  for (IslandOp island : merged_island.islands) {
    for (Value input : island.getControlInputs()) {
      Operation* def = input.getDefiningOp();
      DCHECK_EQ(def->getParentOp(), graph);
      try_update_current_candidate(def);
    }

    // Check island data operands.
    island.walk([&](Operation* op) {
      for (Value input : op->getOperands()) {
        Operation* def = input.getDefiningOp();
        if (!def || def->getParentOp() != graph) continue;

        try_update_current_candidate(def);
      }
    });
  }

  return candidate_island;
}

MergedIsland* CoarseningAnalysis::GetResultCandidateToMergeWith(
    GraphOp graph, MergedIsland& merged_island) {
  // The candidate operation to consider merging the current island group with.
  Operation* candidate = nullptr;
  // The island group of the current candidate if it is an IslandOp, nullptr
  // otherwise.
  MergedIsland* candidate_island = nullptr;

  // Given an input operation, try to replace the current candidate operation
  // with it.
  auto try_update_current_candidate = [&](Operation* rhs) {
    MergedIsland* rhs_island = nullptr;

    // Check if this is an island operation we can merge with.
    auto rhs_it = island_to_merged_island_.find(rhs);
    if (rhs_it != island_to_merged_island_.end()) {
      rhs_island = rhs_it->second;

      // Ignore islands that are already a part of the current island group.
      if (rhs_island == &merged_island) return;

      rhs = rhs_island->insert_point;
    }
    if (!candidate || rhs->isBeforeInBlock(candidate)) {
      candidate = rhs;
      candidate_island = rhs_island;
    }
  };

  // Check island control results.
  for (IslandOp island : merged_island.islands) {
    for (Operation* user : island.getControl().getUsers()) {
      DCHECK_EQ(user->getParentOp(), graph);
      try_update_current_candidate(user);
    }

    // Check island data results.
    Block& graph_body = llvm::cast<GraphOp>(graph).GetBody();
    for (Value result : island.getOutputs()) {
      for (Operation* user : result.getUsers()) {
        Operation* def = graph_body.findAncestorOpInBlock(*user);
        DCHECK_NE(def, nullptr);
        try_update_current_candidate(def);
      }
    }
  }

  return candidate_island;
}

//===----------------------------------------------------------------------===//
// Transformation
//===----------------------------------------------------------------------===//

// IslandResult is a helper struct holding an islands result and associated
// inner op result.
struct IslandResult {
  IslandResult(Value inner_op_result, Value island_result)
      : inner_op_result(inner_op_result), island_result(island_result) {}

  Value inner_op_result;
  Value island_result;
};

// This structure is used to gather the new operands and result of an island
// during merging.
struct IslandOperandsAndResults {
  llvm::SmallSetVector<Value, 8> operands;
  llvm::SmallVector<IslandResult> results;
};

// Collects the results for the new island by going through each data result of
// the islands being merged. Unused results outside of the merged island to be
// formed are pruned. If the child island inner ops consume the parent island
// control result, the child island inner ops will have that respective control
// input pruned. Results of the parent island that are consumed by the child
// island are replaced by the respective inner ops result from the parent
// island.
void GetNewIslandResultsAndForwardResults(
    const MergedIsland& merged_island,
    llvm::SmallVector<IslandResult>& results) {
  results.clear();

  // Collect all of the blocks within each of the island operations, these will
  // be used to detect when an operation has a use within one of the merged
  // islands.
  llvm::SmallPtrSet<Block*, 8> islandBlocks;
  for (IslandOp island : merged_island.islands)
    island->walk([&](Block* block) { islandBlocks.insert(block); });

  for (IslandOp island : merged_island.islands) {
    for (auto ret_vals :
         llvm::zip(island.GetYield().getOperands(), island.getOutputs())) {
      bool result_captured = false;
      Value inner_op_result = std::get<0>(ret_vals);
      Value island_result = std::get<1>(ret_vals);
      for (auto& use : llvm::make_early_inc_range(island_result.getUses())) {
        if (islandBlocks.count(use.getOwner()->getBlock())) {
          // If the use is within our island group, forward the result from
          // inner op.
          use.set(inner_op_result);
        } else if (!result_captured) {
          results.emplace_back(inner_op_result, island_result);
          result_captured = true;
        }
      }
    }
  }
}

// Creates the new merged island.
IslandOp CreateNewIsland(const MergedIsland& merged_island,
                         llvm::ArrayRef<Value> operands,
                         llvm::ArrayRef<IslandResult> results) {
  // Collect types from results.
  llvm::SmallVector<Type, 8> result_types;
  result_types.reserve(results.size());
  for (const auto& result : results)
    result_types.push_back(result.inner_op_result.getType());

  // IslandOps always have a control result.
  result_types.push_back(
      ControlType::get(merged_island.insert_point->getContext()));

  OpBuilder builder(merged_island.insert_point);
  auto new_island = builder.create<IslandOp>(
      merged_island.insert_point->getLoc(), result_types, operands);
  new_island.getBody().push_back(new Block);
  return new_island;
}

// Creates respective YieldOp for the new merged island.
YieldOp CreateNewIslandYieldOp(IslandOp new_island,
                               llvm::ArrayRef<IslandResult> results) {
  llvm::SmallVector<Value, 8> yield_operands;
  yield_operands.reserve(results.size());

  for (auto ret_vals : llvm::zip(results, new_island.getOutputs())) {
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
void MoveInnerOpsToNewIsland(const MergedIsland& merged_island,
                             Operation* new_yield_op) {
  Block* block = new_yield_op->getBlock();

  auto move_inner_ops = [block, new_yield_op](IslandOp island) {
    auto& island_body = island.GetBody().getOperations();
    block->getOperations().splice(new_yield_op->getIterator(), island_body,
                                  island_body.begin(),
                                  std::prev(island_body.end()));
  };
  for (IslandOp island : merged_island.islands) move_inner_ops(island);
}

// Merges the islands within the given island group.
// `island_operands_and_results` is passed in as scrach storage for the duration
// of this function.
void MergeIslands(const MergedIsland& merged_island,
                  IslandOperandsAndResults& island_operands_and_results) {
  // Collect operands for the new merged island.
  island_operands_and_results.operands.clear();
  for (IslandOp island : merged_island.islands)
    island_operands_and_results.operands.insert(island.operand_begin(),
                                                island.operand_end());
  for (IslandOp island : merged_island.islands)
    island_operands_and_results.operands.remove(island.getControl());

  // Collect results for the new merged island.
  GetNewIslandResultsAndForwardResults(merged_island,
                                       island_operands_and_results.results);

  // Create the new merged island.
  IslandOp new_island = CreateNewIsland(
      merged_island, island_operands_and_results.operands.getArrayRef(),
      island_operands_and_results.results);

  // Create associated YieldOp for the new merged island.
  YieldOp new_yield_op =
      CreateNewIslandYieldOp(new_island, island_operands_and_results.results);

  // Move inner ops from original islands into the new island.
  MoveInnerOpsToNewIsland(merged_island, new_yield_op.getOperation());

  // Update control inputs to point to the new merged island.
  for (IslandOp island : merged_island.islands)
    island.getControl().replaceAllUsesWith(new_island.getControl());
  for (IslandOp island : merged_island.islands) island->erase();
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
  data_fetches.reserve(fetch.getFetches().size());
  data_types.reserve(data_fetches.capacity());
  control_fetches.reserve(data_fetches.capacity());

  for (auto value : fetch.getFetches()) {
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
  island.getBody().push_back(new Block);
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

//===----------------------------------------------------------------------===//
// Pass Entry Point
//===----------------------------------------------------------------------===//

#define GEN_PASS_DEF_EXECUTORISLANDCOARSENINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

struct ExecutorIslandCoarseningPass
    : public impl::ExecutorIslandCoarseningPassBase<
          ExecutorIslandCoarseningPass> {
  void runOnOperation() override;
};

void ExecutorIslandCoarseningPass::runOnOperation() {
  // Temporary datastructure to keep operands and results for each island.
  // We define it here to grow and reuse the storage for the duration of the
  // pass.
  IslandOperandsAndResults island_operands_and_results;

  getOperation().walk([&](GraphOp graph) {
    InsertDummyIslandForFetch(graph.GetFetch());

    // Compute an analysis that decides which islands should be merged together,
    // and merge any island groups it finds.
    CoarseningAnalysis analysis(graph);
    for (const MergedIsland& island : analysis.GetMergableIslands())
      MergeIslands(island, island_operands_and_results);
  });
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreateTFExecutorIslandCoarseningPass() {
  return std::make_unique<ExecutorIslandCoarseningPass>();
}

}  // namespace tf_executor
}  // namespace mlir
