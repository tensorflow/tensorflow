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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_jitrt_passes.h.inc"

using mlir::linalg::CodegenStrategy;

struct LowerTransposePass : public LowerTransposeBase<LowerTransposePass> {
  void runOnOperation() override {
    mlir::OpPassManager dynamic_pm("func.func");
    auto avx_lowering_options =
        mlir::x86vector::avx2::LoweringOptions().setTransposeOptions(
            mlir::x86vector::avx2::TransposeLoweringOptions()
                .lower4x8xf32()
                .lower8x8xf32());

    CodegenStrategy strategy;
    strategy.vectorLowering(mlir::linalg::LinalgVectorLoweringOptions()
                                .enableShapeCastLowering(false)
                                .enableVectorTransposeLowering()
                                .enableAVX2Lowering()
                                .setAVX2LoweringOptions(avx_lowering_options));

    strategy.configurePassPipeline(dynamic_pm, &getContext());
    if (failed(runPipeline(dynamic_pm, getOperation()))) {
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
