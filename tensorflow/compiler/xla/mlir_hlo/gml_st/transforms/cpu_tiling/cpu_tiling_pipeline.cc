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
#include <functional>

#include "gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace gml_st {

GmlStCPUTilingOptions getDefaultCPUPipelineOptions(StringRef cpuName,
                                                   int64_t statsDetailLevel) {
  GmlStCPUTilingOptions opts;
  opts.vectorSize = 8;
  opts.reductionEnableHeuristic = false;
  opts.reduction1DSplitRatio = 8;
  opts.reduction1DTileSize = 8;
  opts.reduction2DParallelDimTileSize = 4;
  opts.reduction2DReductionDimTileSize = 4;
  opts.matmulTileSizes = {};
  // TODO(vuson): Re-enable or remove this:
  opts.vectorizationSizeThreshold = 0;
  opts.vectorizationTiledSizeThreshold = 1024;
  opts.lowerToMmt4d = false;
  opts.cpuName = cpuName;
  opts.statsDetailLevel = statsDetailLevel;
  opts.fuseDegenerateReshapes = false;
  opts.inlineFusionClusters = true;
  return opts;
}

void addCPUTilingPipeline(OpPassManager& pm,
                          const GmlStCPUTilingOptions& options) {
  using func::FuncOp;

  pm.addNestedPass<FuncOp>(createCollectStatsPass(options.statsDetailLevel));
  pm.addNestedPass<FuncOp>(createScalarizationPass(false));
  pm.addNestedPass<FuncOp>(
      createVectorizeForCPUPass(options.vectorizationSizeThreshold));

  if (options.lowerToMmt4d) pm.addNestedPass<FuncOp>(createPackMatmulPass());

  pm.addNestedPass<FuncOp>(createTransformScatterForCpuPass());

  pm.addNestedPass<FuncOp>(
      createTransformDotForCpuPass(options.matmulTileSizes, options.cpuName));
  TransformReduceForCpuPassOptions reductionOpts;
  reductionOpts.enableHeuristic = options.reductionEnableHeuristic;
  reductionOpts.tileSize1D = options.reduction1DTileSize;
  reductionOpts.splitRatio1D = options.reduction1DSplitRatio;
  reductionOpts.parallelDimTileSize2D = options.reduction2DParallelDimTileSize;
  reductionOpts.reductionDimTileSize2D =
      options.reduction2DReductionDimTileSize;
  pm.addNestedPass<FuncOp>(createTransformReduceForCpuPass(reductionOpts));

  // Upstream generalization of tensor.pack/unpack (i.e. tensor.pack/unpack ->
  // tensor.pad + linalg.transpose + tensor.insert_slice) does not transfer
  // transformed labels from tensor.pack/unpack to linalg.transpose and thus
  // makes the latter being tiled again.
  // Hence, elementwise ops transformation needs to be run before pack/unpack
  // transformation.
  pm.addNestedPass<FuncOp>(createTransformElementwiseForCpuPass(
      options.vectorSize, options.fuseDegenerateReshapes));
  pm.addNestedPass<FuncOp>(createTransformMmt4DForCpuPass());
  pm.addNestedPass<FuncOp>(createTransformPackForCpuPass());

  if (options.inlineFusionClusters)
    pm.addNestedPass<FuncOp>(createInlineFusionClustersPass());

  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  pm.addNestedPass<FuncOp>(createRewriteForallOpPass());
  pm.addNestedPass<FuncOp>(createComposeExtractInsertSlicePass());
  pm.addNestedPass<FuncOp>(
      createVectorizeForCPUPass(options.vectorizationTiledSizeThreshold));

  // Tile remaining ops by size one and scalarize what we can.
  pm.addNestedPass<FuncOp>(createTileByOnePass());
  pm.addNestedPass<FuncOp>(createScalarizationPass());
  pm.addNestedPass<FuncOp>(createComposeExtractInsertSlicePass());

  pm.addPass(createCanonicalizerPass());

  // Remove transformed labels after tiling all ops.
  pm.addNestedPass<FuncOp>(createRemoveLabelPass());
}

void addDefaultCPUTilingPipeline(OpPassManager& pm, StringRef cpuName,
                                 int64_t statsDetailLevel) {
  addCPUTilingPipeline(pm,
                       getDefaultCPUPipelineOptions(cpuName, statsDetailLevel));
}

}  // namespace gml_st
}  // namespace mlir
