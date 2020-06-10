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

// This transformation pass folds switch and merge nodes.
// This pass assumes/requires:
// 1. Ops in an island execute all under the same condition;
// 2. It is run before graph partitioning (i.e., there are no _Send/_Recv nodes
//    in the graph);
// 3. No other ops, except _Merge, in the graph execute with dead inputs;

#include <climits>
#include <cstdint>
#include <numeric>

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/StandardTypes.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"

#define DEBUG_TYPE "tf-switch-fold"

namespace mlir {
namespace {

class SwitchFoldPass : public mlir::PassWrapper<SwitchFoldPass, FunctionPass> {
 public:
  void runOnFunction() override;
};
}  // namespace

// Returns the defining op for a value looking through islands.
static Operation* GetDefiningOp(Value val) {
  Operation* op = val.getDefiningOp();
  auto island_op = dyn_cast_or_null<tf_executor::IslandOp>(op);
  if (!island_op) return op;
  auto yield_op = island_op.GetYield();
  auto index = val.cast<mlir::OpResult>().getResultNumber();
  return yield_op.getOperand(index).getDefiningOp();
}

// Returns either the value or input to an IdentityOp.
// Note: this should really be handled by constant folding, but identity nodes
// need to be treated specially in general until they are expanded into
// different types of nodes (e.g., recv identity nodes. For conditionals
// identity nodes are common so handle them specially when considering
// predicate in a minimally invasive way until identity's are handled more
// generally.
static Value LookThroughIdentityOp(Value pred_val) {
  if (!pred_val) return pred_val;
  auto op = GetDefiningOp(pred_val);
  if (auto id_op = dyn_cast_or_null<TF::IdentityOp>(op))
    pred_val = id_op.input();
  return pred_val;
}

namespace {

// Worklist queue of ops to be deleted. This is a queue of ops that are dead
// and need to be removed from the graph/their outputs removed. Excluding merge
// that has to be treated specially as it fires with some dead inputs.
class DeadQueue {
 public:
  // Enqueue operation for deletion.
  void Enqueue(Operation* op, bool due_to_control_input) {
    auto merge_op = dyn_cast<tf_executor::MergeOp>(op);

    // Only insert MergeOp if all its inputs are dead.
    if (!merge_op) {
      dead_ops_.insert(op);
      return;
    }

    if (due_to_control_input) return;

    auto pair = merge_nodes_.insert({merge_op, -1});
    auto& count = pair.first->second;
    if (pair.second) {
      // Compute number of non-control inputs. If we have a Switch directly
      // feeding into the Merge then we could have a null value here.
      count = 0;
      for (auto operand : op->getOperands()) {
        if (operand && !operand.getType().isa<tf_executor::ControlType>())
          ++count;
      }
    }
    // Decrement number of unseen inputs.
    --count;
    if (!count) dead_ops_.insert(op);
  }

  // Enqueue users of a value.
  void EnqueueUsers(Value val) {
    for (auto user : val.getUsers()) {
      Enqueue(user, val.getType().isa<tf_executor::ControlType>());
    }
  }

  // Delete dead ops while propagating deadness to consumers.
  void DeleteDeadOps() {
    while (!dead_ops_.empty()) {
      auto dead = dead_ops_.pop_back_val();
      for (auto res : dead->getResults()) {
        EnqueueUsers(res);
      }
      DeleteOp(dead);
    }
  }

  // Iterators over MergeOps. This is used below for merge_nodes_ which maps
  // from merge operation to number of inputs that are dead.
  using MergeMap = llvm::DenseMap<Operation*, int>;
  using const_iterator = MergeMap::const_iterator;
  llvm::iterator_range<const_iterator> merge_nodes() const {
    return llvm::make_range(merge_nodes_.begin(), merge_nodes_.end());
  }

 private:
  void DeleteOp(Operation* op) {
    merge_nodes_.erase(op);
    op->dropAllDefinedValueUses();

    // If a YieldOp is being deleted, then also remove its IslandOp. This is
    // only valid due to requirement that all ops in island execute under same
    // conditions. YieldOp is always inside of an IslandOp and if it is dead,
    // then so is its parent.
    if (isa<tf_executor::YieldOp>(op))
      Enqueue(op->getParentOfType<tf_executor::IslandOp>(), false);
    op->erase();
  }

  // Dead ops that need to be removed/deadness propagated.
  llvm::SetVector<Operation*> dead_ops_;

  // Merge nodes that may be dead.
  MergeMap merge_nodes_;
};  // namespace

}  // namespace

// Enqueues values of foldable switch ops.
static void MatchSwitchFoldOps(tf_executor::SwitchOp switch_op,
                               DeadQueue* queue) {
  Value pred_val = LookThroughIdentityOp(switch_op.predicate());

  // If predicate or input is null then enqueue entire op for deletion.
  if (pred_val == nullptr || switch_op.data() == nullptr) {
    queue->Enqueue(switch_op, false);
    return;
  }

  DenseElementsAttr pred;
  if (!matchPattern(pred_val, m_Constant(&pred))) return;

  bool taken = pred.getSplatValue<bool>();
  Value dead = taken ? switch_op.falseOutput() : switch_op.trueOutput();
  Value live = !taken ? switch_op.falseOutput() : switch_op.trueOutput();
  live.replaceAllUsesWith(switch_op.data());
  queue->EnqueueUsers(dead);

  // Delete switch op.
  switch_op.getOperation()->dropAllDefinedValueUses();
  switch_op.erase();
}

// Folds merge nodes with only a single non-dead input.
static LogicalResult FoldMergeNodes(FuncOp function, const DeadQueue& queue) {
  // Create builder for val_index of MergeOp.
  auto* block = &function.getBlocks().front();
  OpBuilder builder = OpBuilder::atBlockEnd(block);
  auto type = builder.getIntegerType(32);
  auto build_index = [&](Location loc, int value) {
    return builder.create<ConstantOp>(loc, type,
                                      builder.getI32IntegerAttr(value));
  };

  for (auto it : queue.merge_nodes()) {
    // Find the valid input to merge node.
    Value val = nullptr;
    int index = -1;
    auto* merge = it.first;
    auto merge_op = cast<tf_executor::MergeOp>(merge);
    for (auto e : llvm::enumerate(merge->getOperands())) {
      Value operand = e.value();
      if (!operand) continue;
      // Skip control operands.
      if (operand.getType().isa<tf_executor::ControlType>()) break;
      if (val != nullptr) {
        return merge->emitOpError("multiple valid inputs post switch folding");
      }
      val = operand;
      index = e.index();
    }
    assert(val != nullptr && "merge node should have been deleted");
    merge_op.output().replaceAllUsesWith(val);

    // Build and insert value_index only if needed.
    if (!merge_op.value_index().use_empty()) {
      merge_op.value_index().replaceAllUsesWith(
          build_index(merge->getLoc(), index));
    }

    // Propagate control dependencies if used.
    if (!merge_op.control().use_empty()) {
      // Change control dependencies from the merge to being on the parent of
      // the value being propagated.
      auto def_op = val.getDefiningOp();
#ifndef NDEBUG
      auto exec_dialect =
          function.getContext()->getRegisteredDialect("tf_executor");
      assert(def_op->getDialect() == exec_dialect &&
             "unable to forward control dependencies");
#endif
      merge_op.control().replaceAllUsesWith(
          def_op->getResult(def_op->getNumResults() - 1));
    }

    merge->erase();
  }
  return success();
}

// TODO(jpienaar): This should be replace by checking ops in executor dialect.
bool HasSendOrReceive(FuncOp function) {
  return function
      .walk([&](::mlir::Operation* op) {
        auto name = op->getName().getStringRef();
        if (name == "tf._Send" || name == "tf._Recv")
          return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

void SwitchFoldPass::runOnFunction() {
  if (HasSendOrReceive(getFunction())) return;
  DeadQueue queue;
  // Initialize dead queue with dead outputs of foldable SwitchOps.
  getFunction().walk([&](tf_executor::SwitchOp switch_op) {
    MatchSwitchFoldOps(switch_op, &queue);
  });
  queue.DeleteDeadOps();
  if (failed(FoldMergeNodes(getFunction(), queue))) return signalPassFailure();
}  // namespace mlir

namespace tf_executor {
std::unique_ptr<OperationPass<FuncOp>> CreateSwitchFoldPass() {
  return std::make_unique<SwitchFoldPass>();
}
}  // namespace tf_executor

static PassRegistration<SwitchFoldPass> pass(
    "tf-switch-fold", "Fold switch nodes with constant predicates");

}  // namespace mlir
