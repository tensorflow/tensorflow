/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {
#define GEN_PASS_DEF_OPTIMIZEBATCHMATMULPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

// Checks whether the producer of `value` is TFL_DequantizeOp. This function
// iteratively finds the defining op if the direct defining op is TFL_SplitOp.
bool NotFromDequant(mlir::Value value) {
  auto dequant_op = value.getDefiningOp<DequantizeOp>();
  if (dequant_op) {
    return false;
  }
  auto split_op = value.getDefiningOp<SplitOp>();
  if (!split_op) {
    return true;
  }
  return !split_op.getValue().getDefiningOp<DequantizeOp>();
}

// Optimize TFLite operations in functions.
class OptimizeBatchMatmulPass
    : public impl::OptimizeBatchMatmulPassBase<OptimizeBatchMatmulPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeBatchMatmulPass)

  OptimizeBatchMatmulPass() = default;
  OptimizeBatchMatmulPass(const OptimizeBatchMatmulPass &) {}

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize_batch_matmul.inc"

void OptimizeBatchMatmulPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();
  TFL::populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeBatchMatmulPass() {
  return std::make_unique<OptimizeBatchMatmulPass>();
}

static PassRegistration<OptimizeBatchMatmulPass> pass;

}  // namespace TFL
}  // namespace mlir
