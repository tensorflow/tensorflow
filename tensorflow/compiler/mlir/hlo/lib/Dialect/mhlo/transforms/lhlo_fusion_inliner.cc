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

// This file implements LhloFusionInlinerPass, which inline the body
// contents of lmhlo.fusion_op after its body is fully lowered.
//
#include "mlir-hlo/Dialect/mhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace lmhlo {

struct LhloFusionInlinerPass
    : public LhloFusionInlinerPassBase<LhloFusionInlinerPass> {
 public:
  void runOnFunction() override {
    auto func = getFunction();
    SmallVector<FusionOp> worklist;
    func.walk([&](FusionOp fusion) { worklist.push_back(fusion); });
    for (FusionOp fusion : worklist) {
      InlineFusion(fusion);
      fusion.erase();
    }
  }

 private:
  void InlineFusion(FusionOp fusion) {
    Block& block = fusion.region().front();
    assert(block.getNumArguments() == 0);
    for (Operation& op : llvm::make_early_inc_range(block.getOperations())) {
      if (!isa<TerminatorOp>(&op)) {
        op.moveBefore(fusion.getOperation());
      }
    }
  }
};

std::unique_ptr<OperationPass<FuncOp>> createLhloFusionInlinerPass() {
  return std::make_unique<LhloFusionInlinerPass>();
}

}  // namespace lmhlo
}  // namespace mlir
