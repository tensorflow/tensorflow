/* Copyright 2023 The OpenXLA Authors.

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

#include "absl/log/check.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/MemRef/IR/MemRef.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/ImplicitLocOpBuilder.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "xla/mlir/runtime/transforms/passes.h"

namespace xla {
namespace runtime {

using namespace mlir;  // NOLINT

#define GEN_PASS_DEF_MOVEALLOCASTOENTRYBLOCK
#include "xla/mlir/runtime/transforms/passes.h.inc"

class MoveAllocasToEntryBlockPass
    : public impl::MoveAllocasToEntryBlockBase<MoveAllocasToEntryBlockPass> {
  void runOnOperation() override;
};

//===----------------------------------------------------------------------====/

void MoveAllocasToEntryBlockPass::runOnOperation() {
  ModuleOp module = getOperation();
  module.walk([](mlir::func::FuncOp func) {
    CHECK(!func.getBlocks().empty());
    Block* entryBlock = &func.getBlocks().front();
    llvm::SmallVector<memref::AllocaOp> allocas;
    for (auto op : func.getOps<memref::AllocaOp>()) {
      if (op->getBlock() != entryBlock) {
        allocas.push_back(op);
      }
    }

    auto builder =
        ImplicitLocOpBuilder::atBlockBegin(func->getLoc(), entryBlock);
    builder.setInsertionPointToStart(entryBlock);
    for (auto op : allocas) {
      op->moveBefore(builder.getInsertionBlock(), builder.getInsertionPoint());
    }
  });
}

std::unique_ptr<OperationPass<ModuleOp>> CreateMoveAllocasToEntryBlockPass() {
  return std::make_unique<MoveAllocasToEntryBlockPass>();
}

}  // namespace runtime
}  // namespace xla
