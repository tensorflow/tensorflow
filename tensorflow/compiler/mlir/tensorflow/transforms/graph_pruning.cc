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

#include <memory>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/RegionUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"

namespace mlir {
namespace tf_executor {
namespace {

#define GEN_PASS_DEF_EXECUTORGRAPHPRUNINGPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

// This transformation pass prunes a TF graph eliminating dead-nodes.
class GraphPruningPass
    : public impl::ExecutorGraphPruningPassBase<GraphPruningPass> {
 public:
  GraphPruningPass() = default;
  explicit GraphPruningPass(llvm::ArrayRef<std::string> ops_to_preserve);
  void runOnOperation() override;

 private:
  bool ShouldPreserveOp(Operation* op);
  bool ShouldPreserveIsland(IslandOp island);
  void PruneGraph(GraphOp graph);

  llvm::SmallDenseSet<mlir::StringAttr, 4> ops_to_preserve_ids_;
};

// Checks if a tf_executor.Graph can be pruned.
// For TensorFlow V1.0 compatibility: when importing a graph without providing
// feeds/fetches/targets we should not attempt to prune. The best approximation
// here is to check if the graph is of the "main" function and does not have the
// "tf.entry_function" attribute defined.
bool CanPruneGraph(func::FuncOp func) {
  return func.getName() != "main" ||
         func->getAttrOfType<DictionaryAttr>("tf.entry_function") != nullptr;
}

// Visits an op's operand if it is an output of an Operation in the same
// tf_executor.graph.
void VisitOpOperand(GraphOp graph, Value operand,
                    llvm::SmallPtrSetImpl<Operation*>* reachable_ops,
                    llvm::SmallVectorImpl<Operation*>* ops_to_visit) {
  Operation* def = operand.getDefiningOp();
  if (def && def->getParentOp() == graph && reachable_ops->insert(def).second) {
    // Op has not been visited, add to queue to visit later.
    ops_to_visit->push_back(def);
  }
}

// Visits all operands of an op where each operand is an output of an Operation
// in the same tf_executor.graph.
void VisitOpOperands(GraphOp graph, Operation* op,
                     llvm::SmallPtrSetImpl<Operation*>* reachable_ops,
                     llvm::SmallVectorImpl<Operation*>* ops_to_visit) {
  for (Value operand : op->getOperands())
    VisitOpOperand(graph, operand, reachable_ops, ops_to_visit);
}

// Visits an op and it's associated operands. IslandOps are handled differently
// where it's regions op operands are also visited as values may be implicitly
// captured within. NextIterationSourceOp will also visit it's associated
// NextIterationSinkOp.
void VisitOp(GraphOp graph, Operation* op,
             llvm::SmallPtrSetImpl<Operation*>* reachable_ops,
             llvm::SmallVectorImpl<Operation*>* ops_to_visit) {
  if (auto island = llvm::dyn_cast<IslandOp>(op)) {
    mlir::visitUsedValuesDefinedAbove(
        island.getBody(), island.getBody(), [&](OpOperand* operand) {
          VisitOpOperand(graph, operand->get(), reachable_ops, ops_to_visit);
        });
  }

  VisitOpOperands(graph, op, reachable_ops, ops_to_visit);

  // If op is a `tf_executor.NextIteration.Source`, visit its associated
  // `tf_executor.NextIteration.Sink` op.
  if (auto source_op = llvm::dyn_cast<NextIterationSourceOp>(op)) {
    Operation* sink_op = source_op.GetSink().getOperation();
    if (reachable_ops->insert(sink_op).second) ops_to_visit->push_back(sink_op);
  }
}

GraphPruningPass::GraphPruningPass(
    llvm::ArrayRef<std::string> ops_to_preserve) {
  ops_to_preserve_ = ops_to_preserve;
}

void GraphPruningPass::runOnOperation() {
  for (const auto& op_name : ops_to_preserve_) {
    ops_to_preserve_ids_.insert(mlir::StringAttr::get(&getContext(), op_name));
  }
  if (!CanPruneGraph(getOperation())) return;
  getOperation().walk(
      [this](tf_executor::GraphOp graph) { PruneGraph(graph); });
}

// An op should be preserved if either its identifier is contained in
// `ops_to_preserve_ids_` or if it has a `MustExecute` effect.
bool GraphPruningPass::ShouldPreserveOp(Operation* op) {
  if (ops_to_preserve_ids_.contains(op->getName().getIdentifier())) return true;

  llvm::SmallVector<MemoryEffects::EffectInstance, 4> effects;
  auto interface = dyn_cast<MemoryEffectOpInterface>(op);
  if (interface) interface.getEffects(effects);

  for (const auto& effect : effects) {
    if (llvm::isa<TF::ResourceEffects::MustExecute>(effect.getResource())) {
      return true;
    }
  }
  return false;
}

// An island should be preserved if any of its inner ops should be preserved.
bool GraphPruningPass::ShouldPreserveIsland(IslandOp island) {
  auto result = island.walk([this](Operation* inner_op) {
    if (ShouldPreserveOp(inner_op)) return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

// Prunes unreachable operations of a tf_executor.graph operation.
void GraphPruningPass::PruneGraph(GraphOp graph) {
  // A graph has a single block which forms a DAG: operations that aren't
  // reachable from the `fetch` operands can be eliminated.

  llvm::SmallPtrSet<Operation*, 8> reachable_ops;
  llvm::SmallVector<Operation*, 8> ops_to_visit;

  // Visit fetches first to create a starting point for ops that are reachable.
  reachable_ops.insert(graph.GetFetch());
  VisitOpOperands(graph, graph.GetFetch(), &reachable_ops, &ops_to_visit);

  // Find and visit ops that should be preserved regardless of being reachable
  // from a fetch.
  for (Operation& op : graph.GetBody().without_terminator()) {
    auto island = llvm::dyn_cast<IslandOp>(op);
    if (!island) continue;
    if (ShouldPreserveIsland(island)) {
      reachable_ops.insert(&op);
      VisitOp(graph, &op, &reachable_ops, &ops_to_visit);
    }
  }

  // Visit transitive ops until no there are no reachable ops left that have not
  // been visited.
  while (!ops_to_visit.empty()) {
    Operation* op = ops_to_visit.pop_back_val();
    VisitOp(graph, op, &reachable_ops, &ops_to_visit);
  }

  // Erase unreachable ops in reverse order so references don't need to be
  // dropped before removing an op. Going in reverse order will guarantee that
  // when an op to be erased is reached, there are no users left.
  for (Operation& op :
       llvm::make_early_inc_range(llvm::reverse(graph.GetBody())))
    if (!reachable_ops.contains(&op)) op.erase();
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFExecutorGraphPruningPass(
    llvm::ArrayRef<std::string> ops_to_preserve) {
  return std::make_unique<GraphPruningPass>(ops_to_preserve);
}

}  // namespace tf_executor
}  // namespace mlir
