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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PEELING_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PEELING_H

#include <functional>
#include <string>

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

constexpr llvm::StringRef kPeeledMarker = "__internal_peeled_marker__";

/// Rewrite a gml_st::LoopOp/ParallelOp/ForOp with bounds/step that potentially
/// do not divide evenly into a gml_st::LoopOp/ParallelOp/ForOp where the step
/// divides the iteration space evenly, followed by another
/// gml_st::LoopOp/ParallelOp/ForOp for the last (partial) iteration (if any).
/// This transformation is called "loop peeling".
///
/// This function peels all loops in the loop nest by calling
/// peelAndCanonicalizeGmlStLoop.
void peelAllLoops(LoopOp loop, mlir::PatternRewriter &rewriter);
void peelAllLoops(ForOp loop, mlir::PatternRewriter &rewriter);
void peelAllLoops(ParallelOp loop, mlir::PatternRewriter &rewriter);

/// This function peels the `idx`-th loop of the
/// gml_st::LoopOp/ParallelOp/ForOp. To peel all loops in the loop nest, this
/// function must be called multiple times.
///
/// After loop peeling, this function tries to simplify/canonicalize affine.min
/// and affine.max ops in the body of the two gml_st::LoopOp/ParallelOp/ForOps.
/// For more details, refer to `mlir::scf::peelAndCanonicalizeForLoop`.
///
/// The return value indicates whether the loop was rewritten or not. Loops are
/// not rewritten if:
/// * Loop step size is 1 or
/// * Loop bounds and step size are static, and step already divides the
///   iteration space evenly.
///
/// Note: This function rewrites the given gml_st::LoopOp/ParallelOp/ForOp
/// in-place and clones the gml_st::LoopOp/ParallelOp/ForOp operation for the
/// last iteration. It replaces all uses of the unpeeled
/// gml_st::LoopOp/ParallelOp/ForOp with the results of the newly generated
/// gml_st::LoopOp/ParallelOp/ForOp.
LogicalResult peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                           LoopOp loopOp, int64_t idx,
                                           LoopOp &result);
LogicalResult peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter, ForOp loopOp,
                                           int64_t idx, ForOp &result);
LogicalResult peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                           ParallelOp loopOp, int64_t idx,
                                           ParallelOp &result);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PEELING_H
