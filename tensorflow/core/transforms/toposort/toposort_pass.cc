/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/transforms/toposort/toposort_pass.h"

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/RegionKindInterface.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/core//ir/ops.h"

namespace mlir {
namespace tfg {

namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/core/transforms/passes.h.inc"

}  // end namespace

void SortTopologically(Block *block) {
  if (block->empty() || llvm::hasSingleElement(*block)) return;

  Block::iterator next_scheduled_op(&block->front());
  Block::iterator end = block->end();
  if (!block->getParentOp()->hasTrait<OpTrait::NoTerminator>()) --end;

  // Track the ops that still need to be scheduled in a set.
  SmallPtrSet<Operation *, 16> unscheduled_ops;
  for (Operation &op : llvm::make_range(next_scheduled_op, end))
    unscheduled_ops.insert(&op);

  while (!unscheduled_ops.empty()) {
    bool scheduled_at_least_once = false;
    // Loop over the ops that are not sorted yet, try to find the ones "ready",
    // i.e. the ones for which there aren't any operand produced by an op in the
    // set, and "schedule" it (move it before the last_scheduled_op).
    if (end != block->end())
      assert(!block->getParentOp()->hasTrait<OpTrait::NoTerminator>());

    for (Operation &op :
         llvm::make_early_inc_range(llvm::make_range(next_scheduled_op, end))) {
      WalkResult ready_to_schedule = op.walk([&](Operation *nested_op) {
        if (llvm::all_of(nested_op->getOperands(), [&](Value operand) {
              Operation *defining_op = operand.getDefiningOp();
              if (!defining_op) return true;
              Operation *producer_in_block =
                  block->findAncestorOpInBlock(*defining_op);
              if (producer_in_block && producer_in_block != &op &&
                  unscheduled_ops.count(producer_in_block))
                return false;
              return true;
            }))
          return WalkResult::advance();
        return WalkResult::interrupt();
      });
      if (ready_to_schedule.wasInterrupted()) {
        continue;
      }
      unscheduled_ops.erase(&op);
      op.moveBefore(block, next_scheduled_op);
      scheduled_at_least_once = true;
      if (&op == &*next_scheduled_op) ++next_scheduled_op;
    }
    if (!scheduled_at_least_once) {
      unscheduled_ops.erase(&*next_scheduled_op);
      ++next_scheduled_op;
    }
  }
}

// A pass that topologically sort Graph regions.
struct TopoSortPass : TopoSortBase<TopoSortPass> {
  void runOnOperation() override {
    Operation *op = getOperation();
    auto region_op = dyn_cast<RegionKindInterface>(*op);
    if (!region_op) return;
    for (int region : llvm::seq<int>(0, op->getNumRegions())) {
      if (!region_op.hasSSADominance(region) && !op->getRegion(region).empty())
        (void)SortTopologically(&op->getRegion(region).front());
    }
  }
};

}  // namespace tfg
}  // namespace mlir

std::unique_ptr<mlir::Pass> mlir::tfg::CreateTopoSortPass() {
  return std::make_unique<mlir::tfg::TopoSortPass>();
}
