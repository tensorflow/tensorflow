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

#define GEN_PASS_DECL
#include "gml_st/transforms/passes.h.inc"

namespace mlir {
namespace gml_st {

struct MatmulSizes {
  // [m, k] x [k, n]
  int64_t m;
  int64_t n;
  int64_t k;
};

using MatmulTileSizeComputationFn = std::function<MatmulSizes(MatmulSizes)>;

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
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeForCPUPass();

/// Pass to vectorize `memref.copy`.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeCopyPass();

/// Pass to remove redundant `memref.copy` ops.
std::unique_ptr<OperationPass<func::FuncOp>> createNaiveCopyRemovalPass();

/// Pass to gradually lower vector ops to SCF.
std::unique_ptr<OperationPass<func::FuncOp>> createLowerVectorsPass(
    bool enableAVX2 = true);

/// Pass to transform a conv op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformConvForCpuPass();

/// Pass to transform a thlo.scatter op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformScatterForCpuPass();

/// Pass to transform a dot operation for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformDotForCpuPass(
    MatmulTileSizeComputationFn tileSizeFn = nullptr);

/// Pass to transform a linalg.matmul op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformMatmulForCpuPass(
    MatmulTileSizeComputationFn tileSizeFn = nullptr,
    bool lowerToMmt4DOp = false);

/// Pass to fuse linalg on tensor operations.
std::unique_ptr<OperationPass<func::FuncOp>> createFusionOfTensorOpsPass();

/// Pass to convert ops on tensors with 1 element to scalar ops.
std::unique_ptr<OperationPass<func::FuncOp>> createScalarizationPass();

/// Pass to transform a linalg.map op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMapForCpuPass(int64_t tileSize = 1);

/// Pass to transform a linalg.reduce op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformReduceForCpuPass(int64_t vectorSize = 8, int64_t tileSize1D = 32,
                                ArrayRef<int64_t> tileSizes2D = {});

/// Pass to transform a thlo.reverse op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformReverseForCpuPass(int64_t vectorSize = 8);

/// Pass to transform a linalg.transpose op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformTransposeForCpuPass(ArrayRef<int64_t> tileSizes = std::nullopt);

/// Pass to transform a thlo.sort op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformSortForCpuPass();

/// Pass to transform a linalg.generic op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformGenericForCpuPass();

/// Pass to create fusion clusters.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createFusionPlanningForCpuPass(int64_t vectorSize = 8);

/// Pass to outline fusion regions into functions.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFusionOutliningPass();

/// Pass to inline fusion clusters.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createInlineFusionClustersPass();

/// Pass to rewrite tensor.from_elements into tensor.insert.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRewriteFromElementsOpPass();

/// Pass to rewrite scf.forall to scf.for.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createRewriteForallOpPass();

/// Pass to add debug info to be propagated into LLVM backend.
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createAddDebugInfoPass();

/// Populate pattern to remove single/zero iteration scf.forall dimensions.
void populateCollapseForallOpDimensionsPattern(RewritePatternSet &patterns);

struct GmlStCPUTilingOptions
    : public mlir::PassPipelineOptions<GmlStCPUTilingOptions> {
  GmlStCPUTilingOptions() = default;
  GmlStCPUTilingOptions(const GmlStCPUTilingOptions &opts) {
    this->lowerToMmt4d = opts.lowerToMmt4d;
    this->matmulTileSizes = opts.matmulTileSizes;
    this->reduction1DTileSize = opts.reduction1DTileSize;
    this->reduction2DTileSizes = opts.reduction2DTileSizes;
    this->vectorSize = opts.vectorSize;
    this->enableFusionClusters = opts.enableFusionClusters;
  }

  Option<int64_t> vectorSize{*this, "vector-size",
                             llvm::cl::desc("Vector size for a 1D reduction."),
                             llvm::cl::init(8)};

  Option<int64_t> reduction1DTileSize{
      *this, "reduction-1d-tile-size",
      llvm::cl::desc("Tile size for a 1D reduction."), llvm::cl::init(32)};

  ListOption<int64_t> reduction2DTileSizes{
      *this, "reduction-2d-tile-sizes",
      llvm::cl::desc("Tile sizes for a 2D reduction."),
      llvm::cl::list_init<int64_t>({4, 4}), llvm::cl::ZeroOrMore};

  ListOption<int64_t> matmulTileSizes{
      *this, "matmul-tile-sizes",
      llvm::cl::desc("Tile sizes for `linalg.matmul`. Leave empty to determine "
                     "sizes automatically."),
      llvm::cl::list_init<int64_t>({}), llvm::cl::ZeroOrMore};

  Option<bool> lowerToMmt4d{
      *this, "lower-to-mmt4d",
      llvm::cl::desc("Enable the specific code generation (packing) for matmul "
                     "operations."),
      llvm::cl::init(false)};

  Option<bool> enableFusionClusters{
      *this, "enable-fusion-clusters",
      llvm::cl::desc("Enable the pass to create gml_st.fusion clusters."),
      llvm::cl::init(true)};

  Option<bool> enableFusionClusterOutlining{
      *this, "enable-fusion-cluster-outlining",
      llvm::cl::desc(
          "Enable passes to outline and deduplicate gml_st.fusion clusters."),
      llvm::cl::init(false)};

  Option<StringRef> cpuName{
      *this, "cpu",
      llvm::cl::desc("CPU name, similar to llc's -mcpu flag. e.g. 'znver2', "
                     "'skylake-avx512'."),
      llvm::cl::init("")};
};

// Returns default "optimized" tiling parameters.
GmlStCPUTilingOptions getDefaultCPUPipelineOptions(StringRef cpuName);

// Adds tiling-fusion-vectorization passes for tHLO/Linalg ops mix.
void addCPUTilingPipeline(OpPassManager &pm,
                          const GmlStCPUTilingOptions &options);

// Adds tiling-fusion-vectorization passes for tHLO/Linalg ops mix with the
// "optimized" tiling parameters.
void addDefaultCPUTilingPipeline(OpPassManager &pm, StringRef cpuName);

#define GEN_PASS_REGISTRATION
#include "gml_st/transforms/passes.h.inc"

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_PASSES_H
