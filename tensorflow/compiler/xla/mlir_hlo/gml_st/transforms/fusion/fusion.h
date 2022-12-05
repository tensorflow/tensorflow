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

#ifndef MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_FUSION_H
#define MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_FUSION_H

#include "gml_st/IR/gml_st_ops.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace gml_st {

// Create fused operation based on the specificed subset. The result is
// equivalent to the given `materialize` op.
FailureOr<Value> createFusedOp(PatternRewriter &rewriter,
                               MaterializeOp materializeOp);

// Fuses an op into `gml_st.materialize` and performs the necessary updates to
// the surrounding loop if any.
FailureOr<Operation *> fuse(PatternRewriter &rewriter,
                            MaterializeOp materializeOp);

// Finds `gml_st.materialize` ops in the block and fuses ops into them. Verifies
// that fusion candidate doesn't have any uses except the one
// `gml_st.materialize` in the block to avoid exponential code growth.
void fuseGreedily(PatternRewriter &rewriter, Block &block,
                  llvm::function_ref<bool(Operation *)> filterFn = nullptr);

/// Populate fusion patterns.
void populateFusionPatterns(MLIRContext *ctx,
                            function_ref<LogicalResult(MaterializeOp)> filterFn,
                            RewritePatternSet *patterns);

}  // namespace gml_st
}  // namespace mlir

#endif  // MLIR_HLO_DIALECT_GML_ST_TRANSFORMS_FUSION_H
