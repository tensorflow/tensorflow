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
#include <string>
#include <utility>

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONOUTLININGPASS
#include "gml_st/transforms/passes.h.inc"

LogicalResult outlineFusionOp(func::FuncOp funcOp, PatternRewriter& rewriter) {
  MLIRContext* ctx = funcOp.getContext();

  // Outline fusion ops one by one.
  int64_t numOutlinedFusions = 0;
  funcOp.walk([&](gml_st::FusionOp fusionOp) {
    // Insert outlined fusion func ops right before the parent func op.
    rewriter.setInsertionPoint(funcOp);
    auto outlinedFuncTy =
        FunctionType::get(ctx, fusionOp->getOperandTypes(),
                          fusionOp.getTerminator()->getOperandTypes());
    std::string outlinedFuncName =
        llvm::formatv("{0}_fusion_{1}", funcOp.getName(), numOutlinedFusions++)
            .str();
    auto outlinedFuncOp = rewriter.create<func::FuncOp>(
        fusionOp.getLoc(), outlinedFuncName, outlinedFuncTy);

    // Move body and replace yield terminator with return terminator.
    outlinedFuncOp.getBody().takeBody(fusionOp.getRegion());
    Block& theBlock = outlinedFuncOp.getBody().getBlocks().front();
    rewriter.setInsertionPoint(theBlock.getTerminator());
    rewriter.replaceOpWithNewOp<func::ReturnOp>(
        theBlock.getTerminator(), theBlock.getTerminator()->getOperands());

    // Replace fusion op with a call to the newly outlined function.
    rewriter.setInsertionPoint(fusionOp);
    rewriter.replaceOpWithNewOp<func::CallOp>(fusionOp, outlinedFuncOp,
                                              fusionOp->getOperands());
  });

  // Successfully applied pattern if at least one fusion was outlined.
  if (numOutlinedFusions > 0) return success();
  return failure();
}

struct FusionOutliningPass
    : public impl::FusionOutliningPassBase<FusionOutliningPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext* ctx = &getContext();

    // Populate patterns.
    RewritePatternSet patterns(ctx);
    patterns.add(outlineFusionOp);

    if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> createFusionOutliningPass() {
  return std::make_unique<gml_st::FusionOutliningPass>();
}

}  // namespace gml_st
}  // namespace mlir
