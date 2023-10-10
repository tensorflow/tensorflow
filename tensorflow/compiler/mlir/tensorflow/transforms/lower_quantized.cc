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

// Rewrites ops that require quantized inputs or outputs to ops that allow
// non-quantized inputs and outputs.

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"

#define DEBUG_TYPE "tf-lower-quantized"

namespace mlir {
namespace TF {
namespace {

#define GEN_PASS_DEF_LOWERQUANTIZEDPASS
#include "tensorflow/compiler/mlir/tensorflow/transforms/tf_passes.h.inc"

class LowerQuantizedPass
    : public impl::LowerQuantizedPassBase<LowerQuantizedPass> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    mlir::TF::PopulateLoweringQuantizedPatterns(&getContext(), &patterns);
    (void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLowerQuantizedPass() {
  return std::make_unique<LowerQuantizedPass>();
}

}  // namespace TF
}  // namespace mlir
