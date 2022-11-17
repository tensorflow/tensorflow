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

#include <memory>
#include <utility>

#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/stablehlo/transforms/passes.h"
#include "tensorflow/compiler/xla/mlir_hlo/mhlo/IR/hlo_ops.h"

namespace mlir {
namespace odml {

// Convert mhlo.dot to mhlo.dot_general.
LogicalResult ConvertDotToDotGeneral(mhlo::DotOp op,
                                     PatternRewriter &rewriter) {
  auto lhs_type = op.getLhs().getType().cast<ShapedType>();
  auto rhs_type = op.getRhs().getType().cast<ShapedType>();
  if (!lhs_type.hasRank() || !rhs_type.hasRank()) {
    return rewriter.notifyMatchFailure(op, "unsupported unranked input type");
  }
  if (lhs_type.getRank() < 1 || 2 < lhs_type.getRank() ||
      rhs_type.getRank() < 1 || 2 < rhs_type.getRank()) {
    return rewriter.notifyMatchFailure(
        op,
        "unsupported dot operation type; operands must be vectors or "
        "matrices");
  }
  rewriter.replaceOpWithNewOp<mhlo::DotGeneralOp>(
      op, op.getType(), op.getLhs(), op.getRhs(),
      mhlo::DotDimensionNumbersAttr::get(
          op.getContext(),
          /*lhsBatchingDimensions=*/{},
          /*rhsBatchingDimensions=*/{},
          /*lhsContractingDimensions=*/{lhs_type.getRank() - 1},
          /*rhsContractingDimensions=*/{0}),
      op.getPrecisionConfigAttr());
  return success();
}

class OptimizePass
    : public PassWrapper<OptimizePass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final { return "mhlo-optimize"; }
  StringRef getDescription() const final {
    return "Applies various optimizations on MHLO IR";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add(ConvertDotToDotGeneral);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

std::unique_ptr<Pass> createOptimizePass() {
  return std::make_unique<OptimizePass>();
}

static PassRegistration<OptimizePass> pass;

}  // namespace odml
}  // namespace mlir
