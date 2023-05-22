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

#include "gml_st/transforms/test_passes.h"

#include <memory>
#include <string>
#include <utility>

#include "gml_st/transforms/fusion/fusion.h"
#include "gml_st/transforms/peeling/peeling.h"
#include "gml_st/transforms/transforms.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_TESTGMLSTGREEDYFUSION
#include "gml_st/transforms/test_passes.h.inc"

static constexpr llvm::StringRef kTestFusionAppliedLabel =
    "__test_fusion_applied_label__";

struct GreedyFusionPattern : public OpRewritePattern<scf::ForallOp> {
  using OpRewritePattern<scf::ForallOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForallOp op,
                                PatternRewriter &rewriter) const override {
    if (hasLabel(op, kTestFusionAppliedLabel)) return failure();

    rewriter.updateRootInPlace(op, [&]() {
      fuseGreedily(rewriter, &op.getRegion().front(), [](Operation *op) {
        return isa<linalg::BroadcastOp, linalg::FillOp, linalg::MapOp,
                   tensor::CollapseShapeOp, tensor::ExpandShapeOp>(op);
      });
    });

    setLabel(op, kTestFusionAppliedLabel);
    return success();
  }
};

struct TestGmlStGreedyFusionPass
    : public impl::TestGmlStGreedyFusionBase<TestGmlStGreedyFusionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<GmlStDialect, linalg::LinalgDialect, tensor::TensorDialect>();
    linalg::registerTilingInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    patterns.add<GreedyFusionPattern>(ctx);

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns))))
      return signalPassFailure();

    funcOp.walk(
        [](scf::ForallOp op) { removeLabel(op, kTestFusionAppliedLabel); });
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createTestGmlStGreedyFusionPass() {
  return std::make_unique<TestGmlStGreedyFusionPass>();
}

}  // namespace gml_st
}  // namespace mlir
