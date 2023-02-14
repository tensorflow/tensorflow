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

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_REWRITEVECTORTRANSPOSEPASS
#include "gml_st/transforms/passes.h.inc"

struct RewriteVectorTransposePass
    : public impl::RewriteVectorTransposePassBase<RewriteVectorTransposePass> {
  void runOnOperation() override {
    auto avxLoweringOptions =
        x86vector::avx2::LoweringOptions().setTransposeOptions(
            x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32()
                .lower8x8xf32());

    func::FuncOp funcOp = getOperation();
    MLIRContext *context = funcOp.getContext();
    RewritePatternSet patterns(context);
    vector::VectorTransformsOptions vectorTransformOptions;
    vectorTransformOptions = vectorTransformOptions.setVectorTransposeLowering(
        vector::VectorTransposeLowering::EltWise);
    vector::populateVectorTransposeLoweringPatterns(patterns,
                                                    vectorTransformOptions);
    x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avxLoweringOptions, /*benefit=*/10);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createRewriteVectorTransposePass() {
  return std::make_unique<RewriteVectorTransposePass>();
}

}  // namespace gml_st
}  // namespace mlir
