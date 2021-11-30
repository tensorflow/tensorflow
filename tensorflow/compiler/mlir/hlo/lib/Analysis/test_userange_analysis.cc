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

#include "mlir-hlo/Analysis/userange_analysis.h"
#include "mlir-hlo/Dialect/lhlo/IR/lhlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Analysis/BufferViewFlowAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

namespace mlir {

namespace {

struct TestUserangePass : public TestUserangeBase<TestUserangePass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::lmhlo::LmhloDialect>();
  }

  void runOnFunction() override {
    llvm::outs() << "Testing : " << getFunction().getName() << "\n";
    UserangeAnalysis(getFunction(), BufferPlacementAllocs(getFunction()),
                     BufferViewFlowAnalysis(getFunction()))
        .dump(llvm::outs());
  }
};

}  // end anonymous namespace

std::unique_ptr<FunctionPass> createTestUserangePass() {
  return std::make_unique<TestUserangePass>();
}

}  // namespace mlir
