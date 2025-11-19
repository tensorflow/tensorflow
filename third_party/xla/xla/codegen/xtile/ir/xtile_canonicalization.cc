/* Copyright 2025 The OpenXLA Authors.

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

#include "mlir/IR/PatternMatch.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "xla/codegen/xtile/ir/xtile_dialect.h"

namespace xla::xtile {

namespace {

// StableHLO doesn't provide its own canonicalization patterns, so we need to
// fold reshapes that are no-ops.
struct FoldStableHloReshape final
    : public mlir::OpRewritePattern<mlir::stablehlo::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::stablehlo::ReshapeOp op,
      mlir::PatternRewriter& rewriter) const override {
    if (op.getOperand().getType() != op.getType()) {
      return rewriter.notifyMatchFailure(op, "reshape is not a no-op");
    }

    rewriter.replaceOp(op, op.getOperand());
    return mlir::success();
  }
};

}  // namespace

void XTileDialect::getCanonicalizationPatterns(
    mlir::RewritePatternSet& results) const {
  results.add<FoldStableHloReshape>(getContext());
}

}  // namespace xla::xtile
