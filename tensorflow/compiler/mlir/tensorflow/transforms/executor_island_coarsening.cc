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

// This transformation pass takes TFExecutor dialect IslandOps and merges them.
// Note, this currently does not handle TensorFlow V1 style control flow/frames
// or side effecting ops yet.

#include <iterator>
#include <tuple>

#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Block.h"  // TF:local_config_mlir
#include "mlir/IR/Builders.h"  // TF:local_config_mlir
#include "mlir/IR/Location.h"  // TF:local_config_mlir
#include "mlir/IR/Operation.h"  // TF:local_config_mlir
#include "mlir/Pass/Pass.h"  // TF:local_config_mlir
#include "mlir/Pass/PassRegistry.h"  // TF:local_config_mlir
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/core/platform/logging.h"

namespace mlir {
namespace TFExecutor {

namespace {

// IslandType is an enum representing if an island is the island (parent)
// merging another island or is the island (child) being being merged.
enum class IslandType : bool { kParentIsland, kChildIsland };

// Output is a helper struct holding a result index and island type (parent or
// child).
struct Output {
  Output(IslandType island_type, int result_index)
      : island_type(island_type), result_index(result_index) {}

  IslandType island_type;
  int result_index;
};

struct ExecutorIslandCoarsening
    : public FunctionPass<ExecutorIslandCoarsening> {
  void runOnFunction() override;

 private:
  void MergeIslands(OpBuilder* builder, tf_executor::IslandOp* parent,
                    tf_executor::IslandOp* child, IslandType insert_position);
  bool MergeIslandWithOperand(OpBuilder* builder, tf_executor::IslandOp* child);
  bool MergeIslandWithResult(OpBuilder* builder, tf_executor::IslandOp* parent);
};

// Finds the operation leading to an island that the island can be merged with.
// This looks for the operation, either control input or data input to an op,
// that is closest to the island in the graph. If no candidate can be found or
// the op found is not an island, an empty optional is returned.
llvm::Optional<tf_executor::IslandOp> GetOperandCandidateToMergeWith(
    tf_executor::IslandOp* island) {
  Operation* graph_op = island->getParentOp();
  Operation* candidate = nullptr;

  // Check island control operands.
  for (Value* input : island->controlInputs()) {
    Operation* def = input->getDefiningOp();
    DCHECK_EQ(def->getParentOp(), graph_op);
    if (!candidate || candidate->isBeforeInBlock(def)) candidate = def;
  }

  // Check island data operands.
  island->walk([graph_op, &candidate](Operation* op) {
    for (Value* input : op->getOperands()) {
      Operation* def = input->getDefiningOp();
      if (!def || def->getParentOp() != graph_op) continue;
      if (!candidate || candidate->isBeforeInBlock(def)) candidate = def;
    }
  });

  if (!candidate || !llvm::isa<tf_executor::IslandOp>(candidate))
    return llvm::None;

  return llvm::Optional<tf_executor::IslandOp>(
      llvm::cast<tf_executor::IslandOp>(candidate));
}

// Finds the operation leading from an island that the island can be merged
// with. This looks for the operation, either control output or data output to
// an op, that is closest to the island in the graph. If no candidate can be
// found or the op found is not an island, an empty optional is returned.
llvm::Optional<tf_executor::IslandOp> GetResultCandidateToMergeWith(
    tf_executor::IslandOp* island) {
  Operation* graph_op = island->getParentOp();
  Operation* candidate = nullptr;

  // Check island control results.
  for (Operation* user : island->control()->getUsers()) {
    DCHECK_EQ(user->getParentOp(), graph_op);
    if (!candidate || candidate->isBeforeInBlock(user)) candidate = user;
  }

  // Check island data results.
  Block& graph_block = llvm::cast<tf_executor::GraphOp>(graph_op).GetBody();
  for (Value* result : island->outputs()) {
    for (Operation* user : result->getUsers()) {
      Operation* def = graph_block.findAncestorInstInBlock(*user);
      DCHECK_NE(def, nullptr);
      if (!candidate || def->isBeforeInBlock(candidate)) candidate = def;
    }
  }

  if (!candidate || !llvm::isa<tf_executor::IslandOp>(candidate))
    return llvm::None;

  return llvm::Optional<tf_executor::IslandOp>(
      llvm::cast<tf_executor::IslandOp>(candidate));
}

// Collects the operands for the new island by collecting all control inputs of
// the islands being merged.
llvm::SmallSetVector<Value*, 8> GetNewIslandOperands(
    tf_executor::IslandOp* parent, tf_executor::IslandOp* child) {
  llvm::SmallSetVector<Value*, 8> operands;
  operands.insert(parent->getOperands().begin(), parent->getOperands().end());
  operands.insert(child->getOperands().begin(), child->getOperands().end());
  operands.remove(parent->control());
  return operands;
}

// Collects the results for the new island by going through each data output of
// the islands being merged. Unused results outside of the merged island to be
// formed are pruned. If the child island inner ops consume the parent island
// control output, the child island inner ops will have that respective control
// input pruned. Results of the parent island that are consumed by the child
// island are replaced by the respective inner ops output from the parent
// island.
llvm::SmallVector<Output, 8> GetNewIslandResultsAndForwardOutputs(
    mlir::MLIRContext* context, tf_executor::IslandOp* parent,
    tf_executor::IslandOp* child, llvm::SmallVector<Type, 8>* result_types) {
  llvm::SmallVector<Output, 8> results;
  Block& child_body = child->GetBody();
  int result_index = 0;

  Operation& last_op = parent->GetBody().back();
  auto yield_op = cast<tf_executor::YieldOp>(last_op);
  for (Value* output : parent->outputs()) {
    bool output_captured = false;
    Value* yield_input = yield_op.getOperand(result_index);
    for (auto& use : llvm::make_early_inc_range(output->getUses())) {
      if (child_body.findAncestorInstInBlock(*use.getOwner())) {
        // Forward output from inner op.
        use.set(yield_input);
      } else if (!output_captured) {
        results.push_back(Output(IslandType::kParentIsland, result_index));
        result_types->push_back(output->getType());
        output_captured = true;
      }
    }
    result_index++;
  }

  result_index = 0;
  for (Value* output : child->outputs()) {
    if (!output->use_empty()) {
      results.push_back(Output(IslandType::kChildIsland, result_index));
      result_types->push_back(output->getType());
    }
    result_index++;
  }

  // IslandOps always have a control output.
  result_types->push_back(tf_executor::ControlType::get(context));

  return results;
}

// Creates the new merged island.
tf_executor::IslandOp CreateNewIsland(
    OpBuilder* builder, Operation* old_island,
    const llvm::SmallVector<Type, 8>& result_types,
    const llvm::SmallSetVector<Value*, 8>& operands) {
  builder->setInsertionPoint(old_island);
  auto new_island = builder->create<tf_executor::IslandOp>(
      old_island->getLoc(), result_types, operands.getArrayRef(),
      ArrayRef<NamedAttribute>{});
  new_island.body().push_back(new Block);
  return new_island;
}

// Creates respective YieldOp for the new merged island.
tf_executor::YieldOp CreateNewIslandYieldOp(
    OpBuilder* builder, tf_executor::IslandOp* new_island,
    const llvm::SmallVector<Output, 8>& results, tf_executor::IslandOp* parent,
    tf_executor::IslandOp* child) {
  llvm::SmallVector<Value*, 8> yield_operands;
  yield_operands.reserve(results.size());
  for (auto ret_vals : llvm::zip(results, new_island->outputs())) {
    // Get consumed output (island type and result index).
    const auto& output = std::get<0>(ret_vals);
    tf_executor::IslandOp* output_island =
        output.island_type == IslandType::kParentIsland ? parent : child;
    Value* result = output_island->getResult(output.result_index);
    // Replace original result with new island result.
    result->replaceAllUsesWith(std::get<1>(ret_vals));
    // Find YieldOp in original island, grab the associated operand (inner op
    // output) and add it as a operand to the YieldOp of the merged island.
    yield_operands.push_back(
        output_island->GetBody().back().getOperand(output.result_index));
  }

  // Create YieldOp for the new island.
  builder->setInsertionPoint(&new_island->GetBody(),
                             new_island->GetBody().end());
  return builder->create<tf_executor::YieldOp>(new_island->getLoc(),
                                               yield_operands);
}

// Moves inner ops (excluding last op/YieldOp) from islands being merged into
// the new merged island.
void MoveInnerOpsToNewIsland(tf_executor::IslandOp* parent,
                             tf_executor::IslandOp* child,
                             Operation* new_yield_op) {
  Block* block = new_yield_op->getBlock();

  auto move_inner_ops = [block, new_yield_op](tf_executor::IslandOp* island) {
    auto& island_body = island->GetBody().getOperations();
    block->getOperations().splice(new_yield_op->getIterator(), island_body,
                                  island_body.begin(),
                                  std::prev(island_body.end()));
  };

  move_inner_ops(parent);
  move_inner_ops(child);
}

// Merges two islands and places new merged island before parent or child.
void ExecutorIslandCoarsening::MergeIslands(OpBuilder* builder,
                                            tf_executor::IslandOp* parent,
                                            tf_executor::IslandOp* child,
                                            IslandType insert_position) {
  // Collect operands for the new merged island.
  llvm::SmallSetVector<Value*, 8> operands =
      GetNewIslandOperands(parent, child);

  // Collect results and result types for the new merged island.
  llvm::SmallVector<Type, 8> result_types;
  llvm::SmallVector<Output, 8> results = GetNewIslandResultsAndForwardOutputs(
      &getContext(), parent, child, &result_types);

  // Create the new merged island.
  tf_executor::IslandOp new_island = CreateNewIsland(
      builder, insert_position == IslandType::kParentIsland ? *parent : *child,
      result_types, operands);

  // Create associated YieldOp for the new merged island.
  tf_executor::YieldOp new_yield_op =
      CreateNewIslandYieldOp(builder, &new_island, results, parent, child);

  // Move inner ops from original islands into the new island.
  MoveInnerOpsToNewIsland(parent, child, new_yield_op.getOperation());

  // Update control inputs to point to the new merged island.
  child->control()->replaceAllUsesWith(new_island.control());
  parent->control()->replaceAllUsesWith(new_island.control());

  // Remove merged islands.
  child->erase();
  parent->erase();
}

// Merges island with the operand closest to the island in the graph. The
// operand must be another IslandOp for merging to take place. A new island is
// created and the islands being merged are removed if a merge took place.
// Returns true if the island was merged with its operand.
bool ExecutorIslandCoarsening::MergeIslandWithOperand(
    OpBuilder* builder, tf_executor::IslandOp* child) {
  // Find candidate operand to merge island with.
  llvm::Optional<tf_executor::IslandOp> candidate =
      GetOperandCandidateToMergeWith(child);
  if (!candidate.hasValue()) return false;
  auto& parent = candidate.getValue();
  MergeIslands(builder, &parent, child, IslandType::kParentIsland);
  return true;
}

// Merges island with the result closest to the island in the graph. The result
// must be another IslandOp for merging to take place. A new island is created
// and the islands being merged are removed if a merge took place. Returns true
// if the island was merged with its result.
bool ExecutorIslandCoarsening::MergeIslandWithResult(
    OpBuilder* builder, tf_executor::IslandOp* parent) {
  // Find candidate result to merge island with.
  llvm::Optional<tf_executor::IslandOp> candidate =
      GetResultCandidateToMergeWith(parent);
  if (!candidate.hasValue()) return false;
  auto& child = candidate.getValue();
  MergeIslands(builder, parent, &child, IslandType::kChildIsland);
  return false;
}

void ExecutorIslandCoarsening::runOnFunction() {
  getFunction().walk<tf_executor::GraphOp>([this](tf_executor::GraphOp graph) {
    Block& graph_body = graph.GetBody();
    OpBuilder builder(&graph_body);

    bool updated = false;
    do {
      updated = false;

      auto reversed = llvm::reverse(graph_body);
      for (Operation& operation : llvm::make_early_inc_range(reversed)) {
        auto island = llvm::dyn_cast<tf_executor::IslandOp>(operation);
        if (!island) continue;
        updated |= MergeIslandWithResult(&builder, &island);
      }

      for (Operation& operation : llvm::make_early_inc_range(graph_body)) {
        auto island = llvm::dyn_cast<tf_executor::IslandOp>(operation);
        if (!island) continue;
        updated |= MergeIslandWithOperand(&builder, &island);
      }
    } while (updated);
  });
  // TODO(lyandy): Add canonicalization for dedupping control inputs.
}

}  // namespace

FunctionPassBase* CreateTFExecutorIslandCoarseningPass() {
  return new ExecutorIslandCoarsening();
}

static PassRegistration<ExecutorIslandCoarsening> pass(
    "tf-executor-island-coarsening", "Merges TFExecutor dialect IslandOps");

}  // namespace TFExecutor
}  // namespace mlir
