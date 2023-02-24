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

#ifndef MLIR_HLO_GML_ST_TRANSFORMS_PEELING_PEELING_H
#define MLIR_HLO_GML_ST_TRANSFORMS_PEELING_PEELING_H

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

struct GmlStPeelingResult {
  ParallelOp mainLoop = nullptr;
  SmallVector<ParallelOp> tailLoops = {};
};

/// Rewrite a gml_st::ParallelOp with bounds/step that potentially do not divide
/// evenly into a gml_st::ParallelOp where the step divides the iteration space
/// evenly, followed by another gml_st::ParallelOp for the last (partial)
/// iteration (if any).  This transformation is called "loop peeling".
///
/// These functions peel all loops in the loop nest by calling
/// peelAndCanonicalizeGmlStLoop. Additionally, they mark all loops (main and
/// remainder loops) as peeled, so the same loop is not rewritten a second time.
GmlStPeelingResult peelAllLoops(ParallelOp loop,
                                mlir::PatternRewriter &rewriter);

struct SCFForPeelingResult {
  scf::ForOp mainLoop = nullptr;
  scf::ForOp tailLoop = nullptr;
};
SCFForPeelingResult peelSCFForOp(RewriterBase &rewriter, scf::ForOp);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_GML_ST_TRANSFORMS_PEELING_PEELING_H
