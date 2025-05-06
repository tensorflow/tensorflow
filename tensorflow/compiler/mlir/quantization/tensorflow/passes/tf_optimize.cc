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

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/common/tf_attrs_and_constraints.h"  // IWYU pragma: keep - required to use `IsSplatValueEqual`.
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_passes.h"

namespace mlir::tf_quant {
namespace {

// Applies optimization after quantization.
class TFOptimizePass
    : public PassWrapper<TFOptimizePass, OperationPass<func::FuncOp>> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TFOptimizePass)

  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tf-quant-optimize";
  }
  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Applies optimization after quantization";
  }

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/tf_optimize.inc"

void TFOptimizePass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  auto func = getOperation();
  if (failed(applyPatternsGreedily(func, std::move(patterns)))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateTFOptimizePass() {
  return std::make_unique<TFOptimizePass>();
}

static PassRegistration<TFOptimizePass> pass;

}  // namespace mlir::tf_quant
