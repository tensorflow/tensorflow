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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_PASSES_H
#define MLIR_HLO_GML_ST_TRANSFORMS_PASSES_H

#include <functional>
#include <memory>
#include <optional>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"  // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::gml_st {

#define GEN_PASS_DECL
#include "gml_st/transforms/passes.h.inc"

/// Pass to fuse producers into a tiled consumer.
std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass(
    StringRef producer = "", StringRef consumer = "");

/// Pass to match, tile, and fuse softmax implementations.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass(
    ArrayRef<int64_t> tileSizes);
std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass();

/// Pass to tile the root operation and to greedily fuse producers into it.
std::unique_ptr<OperationPass<func::FuncOp>> createGreedyFusionPass(
    ArrayRef<int64_t> tileSizes);
std::unique_ptr<OperationPass<func::FuncOp>> createGreedyFusionPass();

// Pass to collapse dimensions of bcasts, reductions, and cwise ops.
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseShapePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseShapePass(
    const CollapseShapePassOptions &options);

// Pass to tile all tileable ops to size 1.
std::unique_ptr<OperationPass<func::FuncOp>> createTileByOnePass();

/// Pass to compose tensor.extract_slice/insert_slice ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createComposeExtractInsertSlicePass();

/// Pass to vectorize compute ops and scf.for loops that are tiled perfectly.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeForCPUPass(
    int64_t numElementsThreshold = 1024);

/// Pass to vectorize `memref.copy`.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeCopyPass(
    int64_t numElementsThreshold = 8);

/// Pass to remove redundant `memref.copy` ops.
std::unique_ptr<OperationPass<func::FuncOp>> createNaiveCopyRemovalPass();

/// Pass to gradually lower vector ops to SCF.
std::unique_ptr<OperationPass<func::FuncOp>> createLowerVectorsPass(
    bool enableAVX2 = true, bool flatten = false);

/// Pass to pack linalg.matmul as linalg.mmt4d.
std::unique_ptr<OperationPass<func::FuncOp>> createPackMatmulPass();

/// Pass to transform a thlo.scatter op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformScatterForCpuPass();

/// Pass to transform a dot operation for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformDotForCpuPass(
    ArrayRef<int64_t> tileSizes = {}, StringRef cpuName = "");

/// Pass to transform tensor.pack/unpack ops for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformPackForCpuPass();

/// Pass to transform a linalg.mmt4d op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformMmt4DForCpuPass();

/// Pass to fuse linalg on tensor operations.
std::unique_ptr<OperationPass<func::FuncOp>> createFusionOfTensorOpsPass();

/// Pass to convert ops on tensors with 1 element to scalar ops.
std::unique_ptr<OperationPass<func::FuncOp>> createScalarizationPass(
    bool scalarizeAllThlo = true);

/// Pass to transform elementwise ops for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformElementwiseForCpuPass(int64_t vectorSize = 8,
                                     bool fuseDegenerateReshapes = false);

/// Pass to transform a linalg.reduce op for CPU backend.
std::unique_ptr<Pass> createTransformReduceForCpuPass(
    const TransformReduceForCpuPassOptions &option = {});

/// Pass to create fusion clusters.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFusionPlanningForCpuPass(int64_t vectorSize = 8);

/// Pass to outline fusion regions into functions.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFusionOutliningPass();

/// Pass to inline fusion clusters.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createInlineFusionClustersPass();

/// Pass with canonicalization patterns for linalg ops.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createOptimizeLinalgOpsPass();

/// Pass to rewrite tensor.from_elements into tensor.insert.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRewriteFromElementsOpPass();

/// Pass to rewrite scf.forall to scf.for.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRewriteForallOpPass();

/// Pass to add debug info to be propagated into LLVM backend.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddDebugInfoPass();

/// Pass to print stats about tileable ops.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createCollectStatsPass(
    int64_t level = 0);

/// Pass to remove all transformed labels from tiled ops.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRemoveLabelPass();

/// Populate pattern to remove single/zero iteration scf.forall dimensions.
void populateCollapseForallOpDimensionsPattern(RewritePatternSet &patterns);

struct GmlStCPUTilingOptions
    : public mlir::PassPipelineOptions<GmlStCPUTilingOptions> {
  GmlStCPUTilingOptions() = default;
  GmlStCPUTilingOptions(const GmlStCPUTilingOptions &opts) {
    this->lowerToMmt4d = opts.lowerToMmt4d;
    this->matmulTileSizes = opts.matmulTileSizes;
    this->reduction1DTileSize = opts.reduction1DTileSize;
    this->reduction1DSplitRatio = opts.reduction1DSplitRatio;
    this->reduction2DParallelDimTileSize = opts.reduction2DParallelDimTileSize;
    this->reduction2DReductionDimTileSize =
        opts.reduction2DReductionDimTileSize;
    this->vectorSize = opts.vectorSize;
    this->statsDetailLevel = opts.statsDetailLevel;
    this->cpuName = opts.cpuName;
    this->inlineFusionClusters = opts.inlineFusionClusters;
  }

  Option<int64_t> vectorSize{*this, "vector-size",
                             llvm::cl::desc("Vector size for a 1D reduction."),
                             llvm::cl::init(8)};

  Option<bool> reductionEnableHeuristic{
      *this, "reduction-enable-heuristic",
      llvm::cl::desc("Enable tiling parameters heuristic for reductions."),
      llvm::cl::init(false)};

  Option<int64_t> reduction1DTileSize{
      *this, "reduction-1d-tile-size",
      llvm::cl::desc("Tile size for a 1D reduction."), llvm::cl::init(32)};

  Option<int64_t> reduction1DSplitRatio{
      *this, "reduction-1d-split-ratio",
      llvm::cl::desc("Ratio used to split the reduction dimension"),
      llvm::cl::init(8)};

  Option<int64_t> reduction2DParallelDimTileSize{
      *this, "reduction-2d-parallel-dim-tile-size",
      llvm::cl::desc("Tile size for the parallel dimension of a 2D reduction."),
      llvm::cl::init(4)};

  Option<int64_t> reduction2DReductionDimTileSize{
      *this, "reduction-2d-reduction-dim-tile-size",
      llvm::cl::desc(
          "Tile size for the reduction dimension of a 2D reduction."),
      llvm::cl::init(4)};

  ListOption<int64_t> matmulTileSizes{
      *this, "matmul-tile-sizes",
      llvm::cl::desc("Tile sizes for `linalg.matmul`. Leave empty to determine "
                     "sizes automatically."),
      llvm::cl::list_init<int64_t>({}), llvm::cl::ZeroOrMore};

  Option<int64_t> vectorizationSizeThreshold{
      *this, "vectorization-size-threshold",
      llvm::cl::desc("Threshold size for vectorization."), llvm::cl::init(128)};

  Option<int64_t> vectorizationTiledSizeThreshold{
      *this, "vectorization-tiled-size-threshold",
      llvm::cl::desc("Threshold size for vectorization after tiling."),
      llvm::cl::init(1024)};

  Option<bool> lowerToMmt4d{
      *this, "lower-to-mmt4d",
      llvm::cl::desc("Enable the specific code generation (packing) for matmul "
                     "operations."),
      llvm::cl::init(false)};

  Option<StringRef> cpuName{
      *this, "cpu",
      llvm::cl::desc("CPU name, similar to llc's -mcpu flag. e.g. 'znver2', "
                     "'skylake-avx512'."),
      llvm::cl::init("")};

  Option<int64_t> statsDetailLevel{
      *this, "stats-detail-level",
      llvm::cl::desc("Detail level for collecting IR statistics."),
      llvm::cl::init(0)};

  Option<bool> fuseDegenerateReshapes{
      *this, "fuse-degenerate-reshapes",
      llvm::cl::desc("Fuse through tensor.expand/collapse_shape"),
      llvm::cl::init(false)};

  Option<bool> inlineFusionClusters{
      *this, "inline-fusion-clusters",
      llvm::cl::desc("Inline fusion clusters at the end of the pipeline."),
      llvm::cl::init(true)};
};

// Returns default "optimized" tiling parameters.
GmlStCPUTilingOptions getDefaultCPUPipelineOptions(
    StringRef cpuName, int64_t statsDetailLevel = 0);

// Adds tiling-fusion-vectorization passes for tHLO/Linalg ops mix.
void addCPUTilingPipeline(OpPassManager &pm,
                          const GmlStCPUTilingOptions &options);

// Adds tiling-fusion-vectorization passes for tHLO/Linalg ops mix with the
// "optimized" tiling parameters.
void addDefaultCPUTilingPipeline(OpPassManager &pm, StringRef cpuName,
                                 int64_t statsDetailLevel = 0);

#define GEN_PASS_REGISTRATION
#include "gml_st/transforms/passes.h.inc"

}  // namespace mlir::gml_st

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_PASSES_H
