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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TRANSFORMS_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TRANSFORMS_H

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

/// Rewrite a gml_st::LoopOp with bounds/step that potentially do not divide
/// evenly into a gml_st::LoopOp where the step divides the iteration space
/// evenly, followed by another gml_st::LoopOp for the last (partial) iteration
/// (if any). This transformation is called "loop peeling".
///
/// This function peels the `idx`-th loop of the gml_st::LoopOp. To tile all
/// loops in the loop nest, this function must be called multiple times.
///
/// After loop peeling, this function tries to simplify/canonicalize affine.min
/// and affine.max ops in the body of the two gml_st::LoopOps. For more details,
/// refer to `mlir::scf::peelAndCanonicalizeForLoop`.
///
/// The return value indicates whether the loop was rewritten or not. Loops are
/// not rewritten if:
/// * Loop step size is 1 or
/// * Loop bounds and step size are static, and step already divides the
///   iteration space evenly.
///
/// Note: This function rewrites the given gml_st::LoopOp in-place and clones
/// the gml_st::LoopOp operation for the last iteration. It replaces all uses of
/// the unpeeled gml_st::LoopOp with the results of the newly generated
/// gml_st::LoopOp.
LogicalResult peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                           LoopOp loopOp, int64_t idx,
                                           LoopOp &result);

/// Perform standalone tiling of a single LinalgOp by `tileSizes`.
/// An empty vector is interpreted as the identity permutation and the
/// transformation returns early.
///
/// Return a struct containing the tiled loops in the specified order
/// and the cloned op if successful, llvm::None otherwise.
FailureOr<linalg::TiledLinalgOp> tileLinalgOp(
    RewriterBase &b, linalg::LinalgOp op,
    const linalg::LinalgTilingOptions &options);

struct TilingResult {
  // The outermost loop resulted from tiling.
  Operation *outerLoop;
  // The operation inside the loop that corresponds to the op before tiling.
  Operation *tiledOp;
};

/// Perform tiling that creates gml_st.parallel and gml_st.for operations with
/// gml_st.tile subsets. If the tiling is successful, returns the outer loop.
FailureOr<TilingResult> tileToTiles(RewriterBase &b, linalg::LinalgOp op,
                                    ArrayRef<int64_t> tileSizes);

/// Perform tiling that creates gml_st.parallel and gml_st.for operations with
/// gml_st.point subsets. If the tiling is successful, returns the outer loop.
FailureOr<TilingResult> tileToPoints(RewriterBase &b, linalg::LinalgOp op);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_TRANSFORMS_H
