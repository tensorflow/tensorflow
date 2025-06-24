/* Copyright 2022 The OpenXLA Authors.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Value.h"
#include "transforms/passes.h"

namespace mlir {

#define GEN_PASS_DEF_ALLOCTOARGPASS
#include "transforms/passes.h.inc"

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
  // Find unique block and return op.
  FuncOp funcOp = getOperation();
  auto &blocks = funcOp.getFunctionBody().getBlocks();
  if (blocks.size() != 1) {
    funcOp.emitError("expect function with single-block body");
    return signalPassFailure();
  }
  Block &bodyBlock = blocks.front();
  auto returnOp = llvm::cast<func::ReturnOp>(bodyBlock.getTerminator());

  IRRewriter rewriter(&getContext());
  BitVector resultsToErase(funcOp.getNumResults());
  Location loc = returnOp.getLoc();

  for (auto [i, result] : llvm::enumerate(returnOp.getOperands())) {
    Operation *resultDef = result.getDefiningOp();
    Type resultTy = result.getType();

    // Case: plain alloc.
    if (auto allocOp = llvm::dyn_cast_or_null<memref::AllocOp>(resultDef)) {
      resultsToErase.set(i);
      auto attrs = funcOp.getResultAttrDict(i);

      if (failed(funcOp.insertArgument(funcOp.getNumArguments(), resultTy,
                                       attrs, loc))) {
        return signalPassFailure();
      }
      rewriter.replaceOp(allocOp, funcOp.getArguments().back());
      continue;
    }

    // Case: shape-expanded alloc.
    if (auto expandOp =
            llvm::dyn_cast_or_null<memref::ExpandShapeOp>(resultDef)) {
      Operation *expandDef = expandOp.getOperand(0).getDefiningOp();
      if (auto allocOp = llvm::dyn_cast_or_null<memref::AllocOp>(expandDef)) {
        resultsToErase.set(i);
        auto attrs = funcOp.getResultAttrDict(i);
        if (failed(funcOp.insertArgument(funcOp.getNumArguments(), resultTy,
                                         attrs, loc))) {
          return signalPassFailure();
        }

        // Collapse buffer argument to replace possible uses of the unexpanded
        // buffer.
        rewriter.setInsertionPoint(allocOp);
        Value arg = funcOp.getArguments().back();
        Value collapsedArg = rewriter.create<memref::CollapseShapeOp>(
            loc, arg, expandOp.getReassociationIndices());

        // Replace alloc and its expansion.
        rewriter.replaceOp(allocOp, collapsedArg);
        rewriter.replaceOp(expandOp, arg);
        continue;
      }
    }

    returnOp.emitOpError("expected operand #")
        << i << " to be defined by (shape-expanded) memref.alloc";
    return signalPassFailure();
  }

  if (failed(funcOp.eraseResults(resultsToErase))) {
    return signalPassFailure();
  }
  returnOp->eraseOperands(resultsToErase);
}

std::unique_ptr<OperationPass<func::FuncOp>> hlo::createAllocToArgPass() {
  return std::make_unique<AllocToArgPass>();
}

}  // namespace mlir
