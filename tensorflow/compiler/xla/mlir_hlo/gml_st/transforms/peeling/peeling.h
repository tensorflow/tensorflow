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
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

constexpr llvm::StringRef kPeelingAppliedLabel = "__peeling_applied_label__";

using PeelingResult = SmallVector<Operation *>;

/// Rewrite a gml_st::LoopOp/ParallelOp/ForOp with bounds/step that potentially
/// do not divide evenly into a gml_st::LoopOp/ParallelOp/ForOp where the step
/// divides the iteration space evenly, followed by another
/// gml_st::LoopOp/ParallelOp/ForOp for the last (partial) iteration (if any).
/// This transformation is called "loop peeling".
///
/// These functions peel all loops in the loop nest by calling
/// peelAndCanonicalizeGmlStLoop. Additionally, they mark all loops (main and
/// remainder loops) as peeled, so the same loop is not rewritten a second time.
PeelingResult peelAllLoops(LoopOp loop, mlir::PatternRewriter &rewriter);
PeelingResult peelAllLoops(ForOp loop, mlir::PatternRewriter &rewriter);
PeelingResult peelAllLoops(ParallelOp loop, mlir::PatternRewriter &rewriter);

/// These functions peel the `idx`-th loop of the
/// gml_st::LoopOp/ParallelOp/ForOp. To peel all loops in the loop nest, these
/// functions must be called multiple times.
///
/// After loop peeling, these functions try to simplify/canonicalize affine.min
/// and affine.max ops in the body of the two gml_st::LoopOp/ParallelOp/ForOps.
/// For more details, refer to `mlir::scf::peelAndCanonicalizeForLoop`.
///
/// The return value indicates whether the loop was rewritten or not. Loops are
/// not rewritten if:
/// * Loop step size is 1 or
/// * Loop bounds and step size are static, and step already divides the
///   iteration space evenly.
///
/// Note: These functions rewrite the given gml_st::LoopOp/ParallelOp/ForOp
/// in-place and clone the gml_st::LoopOp/ParallelOp/ForOp operation for the
/// last iteration. They replace all uses of the unpeeled
/// gml_st::LoopOp/ParallelOp/ForOp with the results of the newly generated
/// gml_st::LoopOp/ParallelOp/ForOp.
///
/// Note: These functions do not mark the loops as peeled. This should be
/// handled by the caller.
FailureOr<LoopOp> peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                               LoopOp loopOp, int64_t idx);
FailureOr<ForOp> peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                              ForOp loopOp, int64_t idx);
FailureOr<ParallelOp> peelAndCanonicalizeGmlStLoop(RewriterBase &rewriter,
                                                   ParallelOp loopOp,
                                                   int64_t idx);
}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_PEELING_H
