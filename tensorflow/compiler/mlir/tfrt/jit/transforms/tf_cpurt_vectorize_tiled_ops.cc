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

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h"

namespace tensorflow {
namespace {

#define GEN_PASS_CLASSES
#include "tensorflow/compiler/mlir/tfrt/jit/transforms/tf_cpurt_passes.h.inc"

struct VectorizeTiledOpsPass
    : public VectorizeTiledOpsBase<VectorizeTiledOpsPass> {
  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnFunction() override {
    auto funcOp = getFunction();

    // Vector transfer options.
    mlir::VectorTransferToSCFOptions vector_transfer_opts;

    // Vectorize linalg.fill and linalg.generic operations.
    mlir::OpPassManager dynamicPM("builtin.func");
    mlir::linalg::CodegenStrategy strategy;
    strategy.vectorize(mlir::linalg::FillOp::getOperationName())
        .vectorLowering(
            mlir::linalg::LinalgVectorLoweringOptions()
                .setVectorTransferToSCFOptions(vector_transfer_opts));
    strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
    if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();

    mlir::OpPassManager dynamicPM2("builtin.func");
    mlir::linalg::CodegenStrategy strategy2;
    strategy2.vectorize(mlir::linalg::GenericOp::getOperationName())
        .vectorLowering(
            mlir::linalg::LinalgVectorLoweringOptions()
                .setVectorTransferToSCFOptions(vector_transfer_opts));
    strategy2.configurePassPipeline(dynamicPM2, funcOp.getContext());
    if (failed(runPipeline(dynamicPM2, funcOp))) return signalPassFailure();

    // Vectorize padding.
    mlir::OwningRewritePatternList patterns(funcOp.getContext());
    mlir::linalg::populatePadTensorOpVectorizationPatterns(patterns);
    mlir::vector::populateVectorTransferPermutationMapLoweringPatterns(
        patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<mlir::FunctionPass> CreateVectorizeTiledOpsPass() {
  return std::make_unique<VectorizeTiledOpsPass>();
}

}  // namespace tensorflow
