/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <functional>
#include <memory>
#include <optional>

#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_PARTITIONEDTOPOLOGICALSORTPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

bool IsFlexDelegate(Operation *op) {
  // Unwrap from any ControlNodeOps or CustomTfOp.
  if (auto control_node = dyn_cast<mlir::TFL::ControlNodeOp>(*op)) {
    return IsFlexDelegate(&control_node.getBody().front().front());
  }
  if (auto custom_tf_op = dyn_cast<mlir::TFL::CustomTfOp>(*op)) {
    return IsFlexDelegate(&custom_tf_op.getBody().front().front());
  }

  // Our MLIR might be the result of a conversion from a previously generated
  // flatbuffer file.
  if (auto custom_op = dyn_cast<mlir::TFL::CustomOp>(*op)) {
    return custom_op.getCustomCode().starts_with("Flex");
  }

  // We never see TFL::IfOps in the IR -- it is flatbuffer_export that rewrites
  // them from TF::IfOps.
  if (isa<TF::IfOp>(op)) {
    return false;
  }

  // We assume all other TF operators are flex delegated.
  return op->getDialect()->getNamespace() == "tf";
}

// This is a variation of the algorithm from
// llvm/llvm-project/mlir/lib/Transforms/Utils/TopologicalSortUtils.cpp.
//
// Takes a function object `partition` that maps operations to one of two types
// (for the current use case, flex delegate or not.)
//
// Returns true iff the entire block could be brought in topological order.
bool PartitionedTopologicalSort(
    func::FuncOp func, Block *block,
    const std::function<int(Operation *)> &partition) {
  llvm::iterator_range<Block::iterator> ops = block->without_terminator();
  if (ops.empty()) {
    return true;
  }

  int num_all_ops = 0;
  int num_part_ops = 0;
  int num_partitions_after = 0;
  int num_partitions_before = 0;
  int num_unscheduled_ops = 0;

  bool scheduled_everything = true;
  // Unscheduled operations and their respective partitions.
  llvm::DenseMap<Operation *, int> unscheduled_ops;

  // Mark all operations as unscheduled and precompute the partition
  // to which they belong.
  bool prev_part = false;
  for (Operation &op : ops) {
    ++num_all_ops;
    const bool part = partition(&op);
    num_part_ops += part;
    num_partitions_before += part && !prev_part;
    unscheduled_ops.try_emplace(&op, part);
    prev_part = part;
  }

  Block::iterator next_unscheduled_op = ops.begin();
  const Block::iterator end = ops.end();

  // An operation is ready to be scheduled if all its operands are ready. An
  // operand is ready if:
  const auto is_ready = [&](Value value, Operation *top) {
    Operation *parent = value.getDefiningOp();
    // - it is a block argument,
    if (parent == nullptr) return true;
    Operation *ancestor = block->findAncestorOpInBlock(*parent);
    // - it is an implicit capture,
    if (ancestor == nullptr) return true;
    // - it is defined in a nested region, or
    if (ancestor == top) return true;
    // - its ancestor in the block has already been scheduled.
    return unscheduled_ops.find(ancestor) == unscheduled_ops.end();
  };

  while (!unscheduled_ops.empty()) {
    bool scheduled_something = false;

    // This round, we will only schedule operations that belong
    // to this_partition. this_partition is the partition to
    // which the next schedulable operations belongs.
    std::optional<bool> this_partition;

    for (Operation &unscheduled_op : llvm::make_early_inc_range(
             llvm::make_range(next_unscheduled_op, end))) {
      const int op_partition = unscheduled_ops.lookup(&unscheduled_op);
      if (!this_partition) {
        this_partition = op_partition;
      }

      // Find the next schedulable operation of the same partition. An operation
      // is ready to be scheduled if it and its nested operations are ready.
      if (*this_partition == op_partition &&
          !unscheduled_op
               .walk([&](Operation *op) {
                 return llvm::all_of(op->getOperands(),
                                     [&](Value operand) {
                                       return is_ready(operand,
                                                       &unscheduled_op);
                                     })
                            ? WalkResult::advance()
                            : WalkResult::interrupt();
               })
               .wasInterrupted()) {
        unscheduled_ops.erase(&unscheduled_op);
        num_partitions_after += op_partition && (scheduled_something == false);
        scheduled_something = true;
        // Schedule the operation by moving it to the start.
        unscheduled_op.moveBefore(block, next_unscheduled_op);
        if (&unscheduled_op == &*next_unscheduled_op) {
          ++next_unscheduled_op;
        }
      }
    }
    // If no operations were scheduled, give up and advance the iterator.
    if (!scheduled_something) {
      scheduled_everything = false;
      unscheduled_ops.erase(&*next_unscheduled_op);
      ++next_unscheduled_op;
      ++num_unscheduled_ops;
    }
  }
  emitRemark(func.getLoc(), func.getName())
      << ": " << num_part_ops << " ops delegated out of " << num_all_ops
      << " ops with " << num_partitions_after
      << " partitions (originally: " << num_partitions_before << ")";
  if (!scheduled_everything) {
    emitError(func.getLoc(), func.getName())
        << ": " << num_unscheduled_ops << " operations couldn't be scheduled";
  }
  return scheduled_everything;
}

class PartitionedTopologicalSortPass
    : public impl::PartitionedTopologicalSortPassBase<
          PartitionedTopologicalSortPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PartitionedTopologicalSortPass);

  explicit PartitionedTopologicalSortPass(
      const std::function<bool(Operation *)> &partition = IsFlexDelegate)
      : partition_(partition) {}

  void runOnOperation() override;

 private:
  const std::function<bool(Operation *)> partition_;
};

void PartitionedTopologicalSortPass::runOnOperation() {
  func::FuncOp func = getOperation();
  if (!PartitionedTopologicalSort(func, &func.getBody().front(), partition_)) {
    return signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
CreatePartitionedTopologicalSortPass() {
  return std::make_unique<PartitionedTopologicalSortPass>();
}

static PassRegistration<PartitionedTopologicalSortPass> pass;

}  // namespace TFL
}  // namespace mlir
