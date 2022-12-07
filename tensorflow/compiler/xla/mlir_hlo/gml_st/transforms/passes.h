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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASSES_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASSES_H

#include <memory>
#include <string>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_DECL
#include "gml_st/transforms/passes.h.inc"

namespace mlir {
namespace gml_st {

/// The key to the attribute corresponding to the distribution type of
/// operations that have been SIMTfied.
inline constexpr const char kDistributionLabelKey[] =
    "gml-st-distribution-label";

/// Pass to tile ops using TilingInterface.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingPass(
    StringRef opName = "", StringRef opLabel = "", bool distribute = true,
    ArrayRef<int64_t> tileSizes = {});

/// Pass to fuse producers into a tiled consumer.
std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass(
    StringRef producer = "", StringRef consumer = "");

/// Pass to tile and fuse all cwise ops.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingCwisePass(
    bool distribute, ArrayRef<int64_t> tileSizes,
    StringRef distributionLabel = "");
std::unique_ptr<OperationPass<func::FuncOp>> createTilingCwisePass();

/// Pass to tile warp-level ops on GPU.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingGpuWarpPass();

/// Pass to match, tile, and fuse softmax implementations.
std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass(
    bool distribute, ArrayRef<int64_t> tileSizes,
    StringRef distributionLabel = "");
std::unique_ptr<OperationPass<func::FuncOp>> createTilingSoftmaxPass();

/// Pass to tile the root operation and to greedily fuse producers into it.
std::unique_ptr<OperationPass<func::FuncOp>> createGreedyTilingAndFusionPass(
    bool distribute, ArrayRef<int64_t> tileSizes, StringRef distributionLabel);
std::unique_ptr<OperationPass<func::FuncOp>> createGreedyTilingAndFusionPass();

// Pass to collapse dimensions of bcasts, reductions, and cwise ops.
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseShapePass();
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseShapePass(
    const CollapseShapePassOptions &options);

/// Pass to collapse (or uncollapse) materialize operations.
std::unique_ptr<OperationPass<func::FuncOp>> createCollapseMaterializeOpsPass();

/// Pass to lower `gml_st.parallel` to `gpu.launch`, transforming the code into
/// its SIMT interpretation.
std::unique_ptr<OperationPass<func::FuncOp>> createGmlStSimtfyPass(
    StringRef blockDistributionLabel = "block");

/// Pass to eliminate the remaining `gml_st` ops after SIMTfication.
std::unique_ptr<OperationPass<func::FuncOp>> createGmlStToGpuPass(
    StringRef warpDistributionLabel = "warp");

/// Create a pass to convert `gml_st.loop` to `scf.for` and `scf.parallel`
/// loops and memref.load/memref.store accesses.
std::unique_ptr<OperationPass<func::FuncOp>> createGmlStToScfPass();

/// Pass to vectorize linalg.generic ops tiled to gml_st.parallel and gml_st.for
/// loops.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizeGmlStLoopsPass(
    bool vectorizeGmlStOps = false,
    ArrayRef<StringRef> distributionLabels = {});

/// Pass to vectorize gml_st.for loops that are tiled perfectly.
std::unique_ptr<OperationPass<func::FuncOp>>
createVectorizePerfectlyTiledLoopsPass();

/// Pass to lower vector.contract.
std::unique_ptr<OperationPass<func::FuncOp>> createLowerVectorContractPass();

/// Pass to transform a thlo.scatter op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformScatterForCpuPass();

/// Pass to transform a linalg.matmul op for CPU backend.
std::unique_ptr<OperationPass<func::FuncOp>> createTransformMatmulForCpuPass(
    ArrayRef<int64_t> matmulTileSizes = llvm::None,
    bool lowerToMmt4DOp = false);

/// Pass to transform a linalg.map op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformMapForCpuPass(int64_t tileSize = 1);

/// Pass to transform a linalg.reduce op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformReduceForCpuPass(int64_t vectorSize = 8, int64_t tileSize1D = 32,
                                ArrayRef<int64_t> tileSizes2D = {});

/// Pass to transform a linalg.transpose op for CPU backend.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformTransposeForCpuPass(ArrayRef<int64_t> tileSizes = llvm::None);

#define GEN_PASS_REGISTRATION
#include "gml_st/transforms/passes.h.inc"

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PASSES_H
