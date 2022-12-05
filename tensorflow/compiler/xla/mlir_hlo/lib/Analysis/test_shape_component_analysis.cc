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

#include "mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DEF_TESTSHAPECOMPONENTANALYSIS
#include "mlir-hlo/Transforms/passes.h.inc"

using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;

namespace {

struct TestShapeComponentAnalysisPass
    : public impl::TestShapeComponentAnalysisBase<
          TestShapeComponentAnalysisPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnOperation() override {
    ShapeComponentAnalysis shapeComponent;
    llvm::outs() << "Testing : " << getOperation().getName() << '\n';
    // Analyze anything that looks like a shape tensor.
    getOperation().walk([&](Operation* op) {
      // Skip ops with more than one result.
      if (op->getNumResults() != 1) return;
      Value result = op->getResults().front();

      // Dump shape info if any.
      if (auto shapeInfo = shapeComponent.GetShapeInfo(result)) {
        llvm::outs() << "Shape info for " << result << ":\n";
        for (const SymbolicExpr& d : *shapeInfo) {
          llvm::outs().indent(2);
          d.dump(llvm::outs());
        }
      }

      // Dump value info if any.
      if (auto valueInfo = shapeComponent.GetValueInfo(result)) {
        llvm::outs() << "Value info for " << result << ":\n";
        for (const SymbolicExpr& d : *valueInfo) {
          llvm::outs().indent(2);
          d.dump(llvm::outs());
        }
      }
    });
  }
};

}  // end anonymous namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createTestShapeComponentAnalysisPass() {
  return std::make_unique<TestShapeComponentAnalysisPass>();
}

}  // namespace mlir
