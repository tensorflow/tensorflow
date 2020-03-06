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

// This file implements a pass to remove redundant LHLO copy operations.

#include "absl/memory/memory.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"  // TF:llvm-project
#include "mlir/Pass/Pass.h"  // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace xla_lhlo {
namespace {

// Removes LHLO copy operations that copy from allocated buffers to block
// arguments. All uses of each buffer are replaced with the corresponding block
// argument and the buffer is freed. Note that this pass only works in regions
// with a single block.
struct LhloCopyRemoval : mlir::OperationPass<LhloCopyRemoval> {
  void runOnOperation() override {
    llvm::SmallVector<mlir::Operation*, 2> eraseList;
    auto operation = getOperation();
    operation->walk([&](mlir::xla_lhlo::CopyOp copyOp) {
      // If this region contains more than one block, then ignore this copy
      // operation.
      if (copyOp.getParentRegion()->getBlocks().size() > 1) {
        return;
      }

      mlir::Value fromOperand = copyOp.operand();
      mlir::Value toOperand = copyOp.output();

      // If the fromOperand value is a block argument or the toOperand
      // value is not a block argument, then ignore this copy operation.
      if (!fromOperand.getDefiningOp() || toOperand.getDefiningOp()) {
        return;
      }

      // The copy operation removal is illegal if there is at least a single use
      // of toOperand value that lies between the first use of fromOperand value
      // and the copy operation.
      auto fromOperandUsers = fromOperand.getUsers();
      auto firstUser = *fromOperandUsers.begin();
      for (auto op : fromOperandUsers) {
        if (op->isBeforeInBlock(firstUser)) firstUser = op;
      }
      for (auto op : toOperand.getUsers()) {
        if (op->isBeforeInBlock(copyOp) && firstUser->isBeforeInBlock(op)) {
          return;
        }
      }

      // TODO(DFKI): Use live variable analysis to solve aliasing issues among
      // block arguments.

      // Remove the associated alloc operation.
      auto allocOp = fromOperand.getDefiningOp();
      eraseList.push_back(allocOp);

      // Iterate over all uses of the fromOperand to find the associated
      // deallocOp (if any).
      for (auto op : fromOperandUsers) {
        if (isa<mlir::DeallocOp>(op)) {
          eraseList.push_back(op);
          break;
        }
      }

      // Replace all uses of the fromOperand with the toOperand. This rewires
      // all references pointing to the original alloc operation to the new
      // target operation in order to safely remove the copy op.
      fromOperand.replaceAllUsesWith(toOperand);
      copyOp.erase();
    });
    for (auto op : eraseList) {
      op->erase();
    }
  };
};

}  // namespace

std::unique_ptr<Pass> createLhloCopyRemovalPass() {
  return absl::make_unique<LhloCopyRemoval>();
}

static PassRegistration<LhloCopyRemoval> copy_removal_pass(
    "lhlo-copy-removal", "Removes redundant LHLO copy operations");

}  // namespace xla_lhlo
}  // namespace mlir
