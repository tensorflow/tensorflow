/* Copyright 2026 The OpenXLA Authors.

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

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h"

namespace xla::cpu {

#define GEN_PASS_DEF_HOISTALLOCAPASS
#include "xla/backends/cpu/codegen/tiled/transforms/passes.h.inc"

namespace {

class HoistAllocaPass : public impl::HoistAllocaPassBase<HoistAllocaPass> {
 public:
  using HoistAllocaPassBase::HoistAllocaPassBase;

  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    if (funcOp.getBody().empty()) {
      return;
    }
    mlir::Block& entryBlock = funcOp.getBody().front();

    // Find the first non-alloca operation in the entry block.
    // This will be our insertion point.
    mlir::Operation* insertionPoint = nullptr;
    for (mlir::Operation& op : entryBlock) {
      if (!mlir::isa<mlir::memref::AllocaOp>(op)) {
        insertionPoint = &op;
        break;
      }
    }

    if (!insertionPoint) {
      // The block only contains allocas (or is empty, which is invalid).
      // We cannot find a safe insertion point before non-allocas.
      return;
    }

    auto dominatesInsertionPoint = [&](mlir::Value value) {
      if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(value)) {
        return blockArg.getOwner() == &entryBlock;
      }
      return false;
    };

    llvm::SmallVector<mlir::memref::AllocaOp> allocas;
    funcOp.walk([&](mlir::memref::AllocaOp alloca) {
      // Only hoist if all operands dominate the insertion point.
      if (llvm::all_of(alloca.getOperands(), dominatesInsertionPoint)) {
        // Don't hoist if it is already before the insertion point.
        if (alloca->getBlock() == &entryBlock &&
            alloca->isBeforeInBlock(insertionPoint)) {
          return;
        }
        allocas.push_back(alloca);
      }
    });

    for (auto alloca : allocas) {
      alloca->moveBefore(insertionPoint);
    }
  }
};

}  // namespace
}  // namespace xla::cpu
