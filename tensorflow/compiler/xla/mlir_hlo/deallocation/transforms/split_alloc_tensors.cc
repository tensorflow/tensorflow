/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <iterator>
#include <memory>

#include "deallocation/transforms/passes.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace deallocation {
namespace {

void splitAllocTensors(Block& block) {
  for (auto& op : block) {
    for (auto [index, operand] : llvm::enumerate(op.getOperands())) {
      auto* definingOp = operand.getDefiningOp();
      if (llvm::isa_and_nonnull<bufferization::AllocTensorOp>(definingOp)) {
        op.setOperand(index, OpBuilder(&op).clone(*definingOp)->getResult(0));
      }
    }

    for (auto& region : op.getRegions()) {
      for (auto& block : region.getBlocks()) {
        splitAllocTensors(block);
      }
    }
  }

  for (auto& op : llvm::make_early_inc_range(block)) {
    if (llvm::isa<bufferization::AllocTensorOp>(op) && op.use_empty()) {
      op.erase();
    }
  }
}

#define GEN_PASS_DEF_SPLITALLOCTENSORSPASS
#include "deallocation/transforms/passes.h.inc"

struct SplitAllocTensorsPass
    : public impl::SplitAllocTensorsPassBase<SplitAllocTensorsPass> {
  void runOnOperation() override {
    splitAllocTensors(getOperation().getBody().front());
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createSplitAllocTensorsPass() {
  return std::make_unique<SplitAllocTensorsPass>();
}

}  // namespace deallocation
}  // namespace mlir