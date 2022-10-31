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

// This files implements a pass that inlines all mlho.fusion op regions.

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

namespace mlir {

#define GEN_PASS_DEF_INLINEFUSIONPASS
#include "mlir-hlo/Transforms/passes.h.inc"

using ::mlir::func::FuncOp;

namespace {
// Inlines all mhlo.fusion op regions.
class InlineFusionPass : public impl::InlineFusionPassBase<InlineFusionPass> {
 public:
  using InlineFusionPassBase<InlineFusionPass>::InlineFusionPassBase;

 private:
  void runOnOperation() override;
};
}  // namespace

void InlineFusionPass::runOnOperation() {
  FuncOp funcOp = getOperation();
  IRRewriter rewriter(funcOp.getContext());
  funcOp->walk([&](mhlo::FusionOp fusionOp) {
    assert(fusionOp.getFusedComputation().hasOneBlock());
    rewriter.setInsertionPoint(fusionOp);
    BlockAndValueMapping bvm;
    Block& body = *fusionOp.getFusedComputation().begin();
    bvm.map(body.getArguments(), fusionOp.getInputs());
    for (auto& op : body.without_terminator()) rewriter.clone(op, bvm);
    auto results = llvm::map_range(
        body.getTerminator()->getOperands(),
        [&](Value operand) { return bvm.lookupOrDefault(operand); });
    rewriter.replaceOp(fusionOp, llvm::to_vector(results));
  });
}

std::unique_ptr<OperationPass<func::FuncOp>> hlo::createInlineFusionPass() {
  return std::make_unique<InlineFusionPass>();
}

}  // namespace mlir
