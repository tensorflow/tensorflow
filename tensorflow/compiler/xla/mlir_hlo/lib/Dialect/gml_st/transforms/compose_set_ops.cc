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
#include "mlir-hlo/Dialect/gml_st/transforms/compose_set_interface.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_COMPOSESETOPSPASS
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h.inc"

struct ComposeSetPattern
    : public OpInterfaceRewritePattern<ComposeSetInterface> {
  using OpInterfaceRewritePattern<
      ComposeSetInterface>::OpInterfaceRewritePattern;

  LogicalResult matchAndRewrite(ComposeSetInterface iface,
                                PatternRewriter& rewriter) const override {
    Value composed = iface.compose(rewriter);
    if (!composed) return failure();

    rewriter.replaceOp(iface.getOperation(), composed);
    return success();
  }
};

class ComposeSetOpsPass
    : public impl::ComposeSetOpsPassBase<ComposeSetOpsPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registry.insert<arith::ArithmeticDialect, GmlStDialect>();
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.insert<ComposeSetPattern>(ctx);

    // Apply patterns from the top down. This makes sure that we have already
    // composed the operand of a tiling op.
    mlir::GreedyRewriteConfig config;
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
