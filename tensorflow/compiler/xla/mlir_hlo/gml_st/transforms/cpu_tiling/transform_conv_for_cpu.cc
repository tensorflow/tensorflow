/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <memory>
#include <utility>

#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::gml_st {
namespace {

#define GEN_PASS_DEF_TRANSFORMCONVFORCPUPASS
#include "gml_st/transforms/passes.h.inc"

static constexpr llvm::StringRef kConvTransformedLabel =
    "__conv_transformed_label__";

struct Conv2DNhwcHwcfOpTransformPattern
    : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp convOp,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(convOp, kConvTransformedLabel))
      return rewriter.notifyMatchFailure(convOp,
                                         "has already been transformed.");
    if (isa<scf::ForallOp, scf::ForOp>(convOp->getParentOp())) {
      return rewriter.notifyMatchFailure(
          convOp, "has already been tiled by another pass.");
    }

    setLabel(convOp, kConvTransformedLabel);
    return success();
  }
};

struct TransformConvForCpuPass
    : public impl::TransformConvForCpuPassBase<TransformConvForCpuPass> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const final {
    registry
        .insert<arith::ArithDialect, tensor::TensorDialect, scf::SCFDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();

    RewritePatternSet patterns(ctx);
    patterns.add<Conv2DNhwcHwcfOpTransformPattern>(ctx);
    populateCollapseForallOpDimensionsPattern(patterns);

    if (failed(applyPatternsAndFoldGreedily(f, std::move(patterns)))) {
      return signalPassFailure();
    }

    // Ensure we drop the marker in the end.
    f.walk([](linalg::Conv2DNhwcHwcfOp convOp) {
      removeLabel(convOp, kConvTransformedLabel);
    });
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTransformConvForCpuPass() {
  return std::make_unique<mlir::gml_st::TransformConvForCpuPass>();
}

}  // namespace mlir::gml_st
