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

#include <utility>

#include "llvm/ADT/StringRef.h"
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

class PrepareLiftingPass
    : public PassWrapper<PrepareLiftingPass, OperationPass<func::FuncOp>> {
 public:
  StringRef getArgument() const final {
    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-prepare-lifting";
  }

  StringRef getDescription() const final {
    // This is a brief description of the pass.
    return "Apply graph optimizations such as fusing and constant folding to "
           "prepare lifting.";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<TF::TensorFlowDialect>();
  }

  void runOnOperation() override;
};

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/prepare_lifting.inc"

void PrepareLiftingPass::runOnOperation() {
  MLIRContext *ctx = &getContext();
  auto func = getOperation();

  // Decompose batch normalization ops.
  RewritePatternSet patterns(ctx);
  populateWithGenerated(patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreatePrepareLiftingPass() {
  return std::make_unique<PrepareLiftingPass>();
}

static PassRegistration<PrepareLiftingPass> pass;

}  // namespace quant
}  // namespace mlir
