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

#include "lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Analysis/userange_analysis.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DEF_TESTUSERANGE
#include "mlir-hlo/Transforms/passes.h.inc"

namespace {

struct TestUserangePass : public impl::TestUserangeBase<TestUserangePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::lmhlo::LmhloDialect>();
  }

  void runOnOperation() override {
    llvm::outs() << "Testing : " << getOperation().getName() << "\n";
    UserangeAnalysis(getOperation(),
                     bufferization::BufferPlacementAllocs(getOperation()),
                     BufferViewFlowAnalysis(getOperation()))
        .dump(llvm::outs());
  }
};

}  // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTestUserangePass() {
  return std::make_unique<TestUserangePass>();
}

}  // namespace mlir
