/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project  // IWYU pragma: keep
#include "mlir/Rewrite/FrozenRewritePatternSet.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_REMOVESHARDINGCUSTOMCALLPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

// Include patterns generated from `remove_sharding_custom_call.td`.
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/remove_sharding_custom_call.inc"

class RemoveShardingCustomCallPass
    : public impl::RemoveShardingCustomCallPassBase<
          RemoveShardingCustomCallPass> {
 public:
  using impl::RemoveShardingCustomCallPassBase<
      RemoveShardingCustomCallPass>::RemoveShardingCustomCallPassBase;

 private:
  void runOnOperation() override;
};

void RemoveShardingCustomCallPass::runOnOperation() {
  func::FuncOp func_op = getOperation();
  MLIRContext& ctx = getContext();

  RewritePatternSet patterns(&ctx);
  populateWithGenerated(patterns);

  FrozenRewritePatternSet frozen_patterns(std::move(patterns));
  if (failed(applyPatternsGreedily(func_op, frozen_patterns))) {
    func_op.emitWarning() << "Failed to converge "
                          << RemoveShardingCustomCallPass::getArgumentName();
  }
}

}  // namespace mlir::quant::stablehlo
