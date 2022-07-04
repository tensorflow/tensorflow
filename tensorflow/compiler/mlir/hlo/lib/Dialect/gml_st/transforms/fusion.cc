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

#include <memory>
#include <utility>

#include "mlir-hlo/Dialect/gml_st/IR/gml_st_ops.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface.h"
#include "mlir-hlo/Dialect/gml_st/transforms/fusion_interface_impl.h"
#include "mlir-hlo/Dialect/gml_st/transforms/pass_detail.h"
#include "mlir-hlo/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

struct FusionPattern : public OpRewritePattern<MaterializeOp> {
  using OpRewritePattern<MaterializeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MaterializeOp op,
                                PatternRewriter& rewriter) const override {
    Operation* def = op.source().getDefiningOp();
    if (!def) return failure();

    auto iface = llvm::dyn_cast<FusionIterface>(def);
    if (!iface) return failure();

    Value fused = iface.fuse(op.getLoc(), op.set(), rewriter);
    if (!fused) return failure();

    rewriter.replaceOp(op, fused);
    return success();
  }
};

class FusionPass : public FusionPassBase<FusionPass> {
  void getDependentDialects(DialectRegistry& registry) const final {
    registerFusionInterfaceExternalModels(registry);
  }

  void runOnOperation() final {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<FusionPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createFusionPass() {
  return std::make_unique<FusionPass>();
}

}  // namespace gml_st
}  // namespace mlir
