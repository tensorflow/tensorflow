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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_FUSION_FUSION_H
#define MLIR_HLO_GML_ST_TRANSFORMS_FUSION_FUSION_H

#include <utility>

#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/tiling/tiling.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::gml_st {

struct FusionCluster {
  SetVector<Operation *> operations;
  Operation *root;
  // Map from Value of the fusion cluster argument to the root dimensions.
  llvm::SmallVector<std::pair<Value, SmallVector<int64_t>>> argDimsMapping;
};
// Cluster producers and consumers around the root op.
FusionCluster getFusionCluster(
    Operation *op, llvm::function_ref<bool(Operation *)> producerFilterFn,
    llvm::function_ref<bool(Operation *)> consumerFilterFn);

// Creates gml_st.fusion op with a region with ops from the fusion cluster.
// Operands of the ops in the region are replaced with region arguments to
// isolate the fusion cluster form above. Usages of the ops are replaces with
// the fusion op results.
FailureOr<gml_st::FusionOp> wrapFusionCluster(
    PatternRewriter &rewriter, const FusionCluster &fusionCluster);

// Replaces gml_st.fusion op with ops from the region.
LogicalResult inlineFusionCluster(FusionOp fusionOp, PatternRewriter &rewriter);

// Adds patterns to duplicate linalg.fill and tensor.empty that used as init
// parameters.
void populateDuplicateInitOpsPatterns(RewritePatternSet &patterns);

// Fuses an op into `tensor.extract_slice` and performs the necessary updates to
// the surrounding loop if any.
FailureOr<Operation *> fuse(PatternRewriter &rewriter,
                            tensor::ExtractSliceOp materializeOp);

// Finds `tensor.extract_slice` ops in the block and fuses ops into them.
// Verifies that fusion candidate doesn't have any uses except the one
// `tensor.extract_slice` in the block to avoid exponential code growth.
void fuseGreedily(PatternRewriter &rewriter, ArrayRef<Block *> blocks,
                  llvm::function_ref<bool(Operation *)> filterFn = nullptr);

// Tiles the op to gml_st.parallel and fuses greedily according to the filter.
FailureOr<GMLSTTilingResult> tileUsingSCFForallOpAndFuseGreedily(
    PatternRewriter &rewriter, Operation *op, const scf::SCFTilingOptions &opts,
    llvm::function_ref<bool(Operation *)> fuseFilterFn = nullptr);

// Tiles the op to scf.for and fuses greedily according to the filter.
FailureOr<scf::SCFTilingResult> tileUsingSCFForOpAndFuseGreedily(
    PatternRewriter &rewriter, Operation *op, const scf::SCFTilingOptions &opts,
    llvm::function_ref<bool(Operation *)> fuseFilterFn = nullptr);

// Tiles the op to 1 for all dimensions and fuses greedily according to the
// filter function.
LogicalResult tilePeeledOpsToScalars(
    PatternRewriter &rewriter, const GmlStPeelingResult &peelingResult,
    llvm::function_ref<bool(Operation *)> fuseFilterFn = nullptr);

}  // namespace mlir::gml_st

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_FUSION_FUSION_H
