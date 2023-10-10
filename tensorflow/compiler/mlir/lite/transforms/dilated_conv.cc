/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/transforms/dilated_conv.h"

#include <utility>

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"

namespace mlir {
namespace TFL {
namespace {

#define GEN_PASS_DEF_IDENTIFYDILATEDCONVPASS
#include "tensorflow/compiler/mlir/lite/transforms/passes.h.inc"

struct IdentifyDilatedConvPass
    : public impl::IdentifyDilatedConvPassBase<IdentifyDilatedConvPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IdentifyDilatedConvPass)
  void runOnOperation() override;
};

void IdentifyDilatedConvPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  auto func = getOperation();

  patterns.add<ConvertTFDilatedConvOp<TF::Conv2DOp>,
               ConvertTFDilatedConvOp<TF::DepthwiseConv2dNativeOp>>(
      &getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>> CreateIdentifyDilatedConvPass() {
  return std::make_unique<IdentifyDilatedConvPass>();
}

}  // namespace TFL
}  // namespace mlir
