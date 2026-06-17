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

#include <utility>

#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo {

#define GEN_PASS_DEF_OPTIMIZEGRAPHPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/passes.h.inc"

namespace {

class OptimizeGraphPass
    : public impl::OptimizeGraphPassBase<OptimizeGraphPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OptimizeGraphPass)

  explicit OptimizeGraphPass() = default;

 private:
  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/optimize_graph.inc"

void OptimizeGraphPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  auto func = getOperation();
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}
}  // namespace

}  // namespace mlir::quant::stablehlo
