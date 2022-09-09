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

#include <memory>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace gml_st {

#define GEN_PASS_DEF_VECTORIZEGMLSTLOOPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

struct VectorizeGmlStLoopsPass
    : public impl::VectorizeGmlStLoopsPassBase<VectorizeGmlStLoopsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    // Vectorize linalg.generic operations inside gml_st.for and gml_st.parallel
    // loops.
    OpPassManager dynamicPM("func.func");
    linalg::CodegenStrategy strategy;
    strategy.vectorize(linalg::GenericOp::getOperationName(),
                       [](mlir::Operation *op) {
                         auto generic = mlir::dyn_cast<linalg::GenericOp>(op);
                         if (!generic) return failure();
                         if (op->getParentOfType<ForOp>() ||
                             op->getParentOfType<ParallelOp>()) {
                           return success();
                         }
                         return failure();
                       });
    strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
    if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeGmlStLoopsPass() {
  return std::make_unique<VectorizeGmlStLoopsPass>();
}

}  // namespace gml_st
}  // namespace mlir
