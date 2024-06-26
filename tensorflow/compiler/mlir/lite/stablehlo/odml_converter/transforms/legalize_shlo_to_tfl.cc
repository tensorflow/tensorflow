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
#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo  // IWYU pragma: keep
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"  // IWYU pragma: keep

namespace mlir {
namespace odml {
namespace {

#define GEN_PASS_DEF_LEGALIZESHLOTOTFLPASS
#include "tensorflow/compiler/mlir/lite/stablehlo/odml_converter/passes.h.inc"
#include "tensorflow/compiler/mlir/lite/stablehlo/odml_converter/transforms/generated_legalize_shlo_to_tfl.inc"

class LegalizeShloToTflPass
    : public impl::LegalizeShloToTflPassBase<LegalizeShloToTflPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeShloToTflPass)

  void runOnOperation() override {
    auto func = getOperation();
    RewritePatternSet patterns(&getContext());
    populateWithGenerated(patterns);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> CreateLegalizeShloToTflPass() {
  return std::make_unique<LegalizeShloToTflPass>();
}

}  // namespace odml
}  // namespace mlir
