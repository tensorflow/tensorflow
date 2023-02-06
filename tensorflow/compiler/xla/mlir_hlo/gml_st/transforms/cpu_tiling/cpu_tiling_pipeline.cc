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

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace gml_st {

void addCPUTilingPipeline(OpPassManager& pm,
                          const GmlStCPUPipelineOptions& options) {
  using func::FuncOp;

  pm.addNestedPass<FuncOp>(createTransformScatterForCpuPass());
  pm.addNestedPass<FuncOp>(createTransformReduceForCpuPass(
      options.vectorSize, options.reduction1DTileSize,
      options.reduction2DTileSizes));
  pm.addNestedPass<FuncOp>(createTransformMatmulForCpuPass(
      options.matmulTileSizes, options.lowerToMmt4d));
  pm.addNestedPass<FuncOp>(createTransformTransposeForCpuPass());
  pm.addNestedPass<FuncOp>(createTransformMapForCpuPass(options.vectorSize));
  pm.addNestedPass<FuncOp>(createTransformSortForCpuPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::gml_st::createTransformReverseForCpuPass());

  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(createComposeExtractInsertSlicePass());
  pm.addNestedPass<FuncOp>(createVectorizeForCPUPass());
  pm.addNestedPass<FuncOp>(createScalarizationPass());
  pm.addNestedPass<FuncOp>(createRewriteVectorContractPass());
}

}  // namespace gml_st
}  // namespace mlir
