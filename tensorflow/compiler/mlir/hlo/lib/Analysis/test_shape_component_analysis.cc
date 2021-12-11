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

#include "mlir-hlo/Analysis/shape_component_analysis.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Transforms/PassDetail.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

using SymbolicExpr = ShapeComponentAnalysis::SymbolicExpr;

namespace {

struct TestShapeComponentAnalysisPass
    : public TestShapeComponentAnalysisBase<TestShapeComponentAnalysisPass> {
  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<mlir::mhlo::MhloDialect>();
  }

  void runOnFunction() override {
    ShapeComponentAnalysis shape_component;
    llvm::outs() << "Testing : " << getFunction().getName() << '\n';
    // Analyze anything that looks like a shape tensor.
    getFunction().walk([&](Operation* op) {
      // Skip ops with more than one result.
      if (op->getNumResults() != 1) return;
      Value result = op->getResults().front();

      // Dump shape info if any.
      if (auto shapeInfo = shape_component.GetShapeInfo(result)) {
        llvm::outs() << "Shape info for " << result << ":\n";
        for (const SymbolicExpr& d : *shapeInfo) {
          llvm::outs().indent(2);
          d.dump(llvm::outs());
        }
      }

      // Dump value info if any.
      if (auto valueInfo = shape_component.GetValueInfo(result)) {
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

std::unique_ptr<FunctionPass> createTestShapeComponentAnalysisPass() {
  return std::make_unique<TestShapeComponentAnalysisPass>();
}

}  // namespace mlir
