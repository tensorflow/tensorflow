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

// Legalize TensorFlow and TensorFlow Lite to TOSA

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Tosa/IR/TosaOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_common.h"
#include "tensorflow/compiler/mlir/tosa/transforms/legalize_utils.h"
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h"

namespace mlir {
namespace tosa {

namespace {

#define GEN_PASS_DEF_TOSALEGALIZETFTFLPASS
#include "tensorflow/compiler/mlir/tosa/transforms/passes.h.inc"

// Performs lowering to TOSA dialect
class LegalizeTFTFL : public impl::TosaLegalizeTFTFLPassBase<LegalizeTFTFL> {
 public:
  explicit LegalizeTFTFL() {}
  void runOnOperation() override;
};

void LegalizeTFTFL::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  populateLegalizeTFPatterns(ctx, patterns);
  populateLegalizeTFLPatterns(ctx, patterns);

  func::FuncOp func = getOperation();
  if (ApplyPatternsWithShapeResolution(func, std::move(patterns)).failed()) {
    signalPassFailure();
  }
}

}  // anonymous namespace

// Creates an instance of the TensorFlow Lite dialect LegalizeTFL pass.
std::unique_ptr<OperationPass<func::FuncOp>> createLegalizeTFTFLPass() {
  return std::make_unique<LegalizeTFTFL>();
}

}  // namespace tosa
}  // namespace mlir
