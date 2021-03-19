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

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project

namespace mlir {
namespace TFL {
namespace {

struct IdentifyDilatedConvPass
    : public PassWrapper<IdentifyDilatedConvPass, FunctionPass> {
  void runOnFunction() override;
};

void IdentifyDilatedConvPass::runOnFunction() {
  OwningRewritePatternList patterns;
  auto func = getFunction();

  patterns.insert<ConvertTFDilatedConvOp<TF::Conv2DOp>,
                  ConvertTFDilatedConvOp<TF::DepthwiseConv2dNativeOp>>(
      &getContext());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
}  // namespace

static PassRegistration<IdentifyDilatedConvPass> pass(
    "tfl-identify-dilated-conv",
    "Identify and replace patterns for dilated convolution.");

}  // namespace TFL
}  // namespace mlir
