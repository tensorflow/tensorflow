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

// Thsi file implements passes to convert complex operations to equivalent real
// value operations. This does not include removing complex values from function
// argument or return types.

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir-hlo/utils/hlo_utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LOWERCOMPLEXPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

namespace {
class LowerComplexPass : public impl::LowerComplexPassBase<LowerComplexPass> {
 public:
  /// Performs the lowering to MHLO dialect.
  void runOnOperation() override;
};

#include "generated_lower_complex.inc"

// Lowers the complex operations that can be represented using other operations.
void LowerComplexPass::runOnOperation() {
  // Add lowering patterns to the list.
  RewritePatternSet patterns(&getContext());
  mlir::mhlo::populateComplexLoweringPatterns(&getContext(), &patterns);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    return signalPassFailure();
}

}  // end anonymous namespace
}  // end namespace mhlo
}  // end namespace mlir

void mlir::mhlo::populateComplexLoweringPatterns(MLIRContext* /*context*/,
                                                 RewritePatternSet* patterns) {
  populateWithGenerated(*patterns);
}

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::mhlo::createLowerComplexPass() {
  return std::make_unique<mlir::mhlo::LowerComplexPass>();
}
