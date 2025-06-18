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

#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Support/TypeID.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h"  // IWYU pragma: keep
#include "xla/mlir_hlo/mhlo/transforms/rewriters.h"

//===----------------------------------------------------------------------===//
// The unfuse-mhlo-batch-norm Pass.
//===----------------------------------------------------------------------===//

namespace mlir::tf_quant::stablehlo {

#define GEN_PASS_DEF_UNFUSEMHLOBATCHNORMPASS
#include "tensorflow/compiler/mlir/quantization/stablehlo/passes/tf_passes.h.inc"

namespace {

class UnfuseMhloBatchNormPass
    : public impl::UnfuseMhloBatchNormPassBase<UnfuseMhloBatchNormPass> {
 public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UnfuseMhloBatchNormPass)

  explicit UnfuseMhloBatchNormPass() = default;

 private:
  void runOnOperation() override;
};

void UnfuseMhloBatchNormPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  RewritePatternSet patterns(ctx);
  mhlo::populateUnfuseBatchNormPatterns(ctx, &patterns);

  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}
}  // namespace

}  // namespace mlir::tf_quant::stablehlo
