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
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/ChloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"  // IWYU pragma: keep

namespace mlir::quant::stablehlo {
namespace {

#define GEN_PASS_DEF_OPTIMIZEINTGRAPH
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/passes.h.inc"

class OptimizeIntGraph : public impl::OptimizeIntGraphBase<OptimizeIntGraph> {
 public:
  OptimizeIntGraph() = default;
  OptimizeIntGraph(const OptimizeIntGraph &) = default;
  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/bridge/optimize.inc"

void OptimizeIntGraph::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  auto func = getOperation();
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateOptimizeIntGraphPass() {
  return std::make_unique<OptimizeIntGraph>();
}

}  // namespace mlir::quant::stablehlo
