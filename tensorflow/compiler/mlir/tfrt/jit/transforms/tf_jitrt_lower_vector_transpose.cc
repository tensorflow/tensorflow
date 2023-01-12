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

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_DEF_LOWERTRANSPOSE
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

struct LowerTransposePass
    : public impl::LowerTransposeBase<LowerTransposePass> {
  void runOnOperation() override {
    auto avx_lowering_options =
        mlir::x86vector::avx2::LoweringOptions().setTransposeOptions(
            mlir::x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32()
                .lower8x8xf32());

    mlir::func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = funcOp.getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::vector::VectorTransformsOptions vectorTransformOptions;
    vectorTransformOptions = vectorTransformOptions.setVectorTransposeLowering(
        mlir::vector::VectorTransposeLowering::EltWise);
    mlir::vector::populateVectorTransposeLoweringPatterns(
        patterns, vectorTransformOptions);
    mlir::x86vector::avx2::populateSpecializedTransposeLoweringPatterns(
        patterns, avx_lowering_options, /*benefit=*/10);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateLowerVectorTransposePass() {
  return std::make_unique<LowerTransposePass>();
}

}  // namespace tensorflow
