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

// This file implements a pass to remove redundant LHLO copy operations.

#include "absl/memory/memory.h"
#include "mlir/Dialect/StandardOps/Ops.h"  // TF:llvm-project
#include "mlir/IR/Operation.h"             // TF:llvm-project
#include "mlir/Pass/Pass.h"                // TF:llvm-project
#include "tensorflow/compiler/mlir/xla/ir/lhlo_ops.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace xla_hlo {
namespace {

/// Removes Lhlo.CopyOp that copies from an allocated buffer to the block
/// argument. All uses of the buffer are replaced with the block argument.
struct LhloCopyRemoval : mlir::FunctionPass<LhloCopyRemoval> {
  void runOnFunction() override {
    llvm::SmallVector<mlir::Operation*, 2> eraseList;
    getFunction().walk([&](mlir::xla_lhlo::CopyOp copyOp) {
      mlir::Value fromOperand = copyOp.operand();
      mlir::Value toOperand = copyOp.output();

      // If the fromOperand value is a block argument or the toOperand
      // value is not a block argument, then ignore this copy operation.
      if (!fromOperand.getDefiningOp() || toOperand.getDefiningOp()) {
        return;
      }

      // Remove the associated alloc operation.
      auto allocOp = fromOperand.getDefiningOp();
      eraseList.push_back(allocOp);

      // Iterate over all uses of the fromOperand to find the associated
      // deallocOp (if any).
      for (auto op : fromOperand.getUsers()) {
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

std::unique_ptr<OpPassBase<FuncOp>> createLhloCopyRemovalPass() {
  return absl::make_unique<LhloCopyRemoval>();
}

static PassRegistration<LhloCopyRemoval> copy_removal_pass(
    "lhlo-copy-removal", "Removes redundant LHLO copy operations");

}  // namespace
}  // namespace xla_hlo
}  // namespace mlir
