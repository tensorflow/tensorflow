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

#include <iterator>
#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/compose_tile_interface.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

template <typename TilingOp>
struct ComposePattern : public OpRewritePattern<TilingOp> {
  using OpRewritePattern<TilingOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TilingOp op,
                                PatternRewriter& rewriter) const override {
    auto iface = llvm::dyn_cast<ComposeTileInterface>(op.getOperation());
    if (!iface) return failure();

    Value composedTile = iface.compose(rewriter);
    if (!composedTile) return failure();

    rewriter.replaceOp(op, composedTile);
    return success();
  }
};

class ComposeSetOpsPass : public ComposeSetOpsPassBase<ComposeSetOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithmeticDialect, GmlStDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<ComposePattern<TileOp>>(ctx);
    mlir::GreedyRewriteConfig config;
    // Apply patterns from the top down. This makes sure that we have already
    // composed the operand of a tiling op.
    config.useTopDownTraversal = true;
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns),
                                            config))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createComposeSetOpsPass() {
  return std::make_unique<ComposeSetOpsPass>();
}

}  // namespace gml_st
}  // namespace mlir
