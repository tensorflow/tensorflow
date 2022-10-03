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

// This files implements a pass that partially bufferized IR.

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"

namespace mlir {

#define GEN_PASS_DEF_ALLOCTOARGPASS
#include "mlir-hlo/Transforms/passes.h.inc"

using ::mlir::func::FuncOp;

namespace {
class AllocToArgPass : public impl::AllocToArgPassBase<AllocToArgPass> {
 public:
  using AllocToArgPassBase<AllocToArgPass>::AllocToArgPassBase;

 private:
  void runOnOperation() override;
};
}  // namespace

void AllocToArgPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());
  BitVector resultsToErase(funcOp.getNumResults());
  Operation *terminator = funcOp.getBody().back().getTerminator();
  for (OpOperand &result : terminator->getOpOperands()) {
    Operation *allocOp = result.get().getDefiningOp();
    if (!allocOp || !isa<memref::AllocOp>(allocOp)) {
      terminator->emitOpError("expected operand #")
          << result.getOperandNumber() << " to be defined by an memref.alloc";
      return signalPassFailure();
    }
    resultsToErase.set(result.getOperandNumber());
    auto attrs = funcOp.getResultAttrDict(result.getOperandNumber());
    funcOp.insertArgument(funcOp.getNumArguments(), result.get().getType(),
                          attrs, result.get().getLoc());
    rewriter.replaceOp(allocOp, funcOp.getArguments().back());
  }
  funcOp.eraseResults(resultsToErase);
  terminator->eraseOperands(resultsToErase);
}

std::unique_ptr<OperationPass<func::FuncOp>> hlo::createAllocToArgPass() {
  return std::make_unique<AllocToArgPass>();
}

}  // namespace mlir
