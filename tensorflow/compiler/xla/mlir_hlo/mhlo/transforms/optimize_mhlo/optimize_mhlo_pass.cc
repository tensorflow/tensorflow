/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "mhlo/transforms/passes.h"
#include "mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_OPTIMIZEMHLOPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {
class OptimizeMhloPass : public impl::OptimizeMhloPassBase<OptimizeMhloPass> {
 public:
  /// Performs the lowering to MHLO dialect.
  void runOnOperation() override;
};
// Lowers the complex operations that can be represented using other operations.
void OptimizeMhloPass::runOnOperation() {
  // Add lowering patterns to the list.
  RewritePatternSet patterns(&getContext());
  populateOptimizeMhloPatterns(&getContext(), &patterns);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}
}  // end anonymous namespace
}  // namespace mhlo
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::mhlo::createOptimizeMhloPass() {
  return std::make_unique<mlir::mhlo::OptimizeMhloPass>();
}
