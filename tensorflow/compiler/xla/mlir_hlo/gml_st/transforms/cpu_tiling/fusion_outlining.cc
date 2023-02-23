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

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONOUTLININGPASS
#include "gml_st/transforms/passes.h.inc"

void outlineFusionOp(func::FuncOp parentFuncOp, gml_st::FusionOp fusionOp,
                     int64_t localFusionId, PatternRewriter& rewriter) {
  MLIRContext* ctx = fusionOp.getContext();

  // Find implicit operands, all of which must be constant-like.
  Region& fusionBody = fusionOp.getBodyRegion();
  SetVector<Operation*> implicitConstantLikeOperands;
  visitUsedValuesDefinedAbove({fusionBody}, [&](OpOperand* operand) -> void {
    Operation* def = operand->get().getDefiningOp();
    assert(def && def->getNumOperands() == 0 && isPure(def) &&
           "expect only constant-like implicit operands");
    implicitConstantLikeOperands.insert(def);
  });

  // Insert outlined fusion func ops right before the parent func op.
  rewriter.setInsertionPoint(parentFuncOp);
  std::string outlinedFuncName =
      llvm::formatv("{0}_fusion_{1}", parentFuncOp.getName(), localFusionId)
          .str();
  auto funcTy = FunctionType::get(ctx, fusionOp->getOperandTypes(),
                                  fusionOp.getTerminator()->getOperandTypes());
  auto funcOp = rewriter.create<func::FuncOp>(fusionOp.getLoc(),
                                              outlinedFuncName, funcTy);

  // Move body and replace yield with return terminator.
  Region& funcOpBodyRegion = funcOp.getBody();
  funcOpBodyRegion.takeBody(fusionOp.getRegion());
  Block& funcOpBodyBlock = funcOpBodyRegion.getBlocks().front();
  auto yieldOp = llvm::cast<gml_st::YieldOp>(funcOpBodyBlock.getTerminator());
  rewriter.setInsertionPoint(yieldOp);
  rewriter.replaceOpWithNewOp<func::ReturnOp>(yieldOp, yieldOp.getOperands());

  // Inline constant-like implicit operands.
  rewriter.setInsertionPointToStart(&funcOpBodyBlock);
  for (Operation* constantLikeOp : implicitConstantLikeOperands) {
    Operation* clonedConstantLikeOp = rewriter.clone(*constantLikeOp);
    for (auto it : llvm::zip(constantLikeOp->getResults(),
                             clonedConstantLikeOp->getResults())) {
      replaceAllUsesInRegionWith(std::get<0>(it), std::get<1>(it),
                                 funcOp.getBody());
    }
  }

  // Replace fusion op with a call to the newly outlined function.
  rewriter.setInsertionPoint(fusionOp);
  rewriter.replaceOpWithNewOp<func::CallOp>(fusionOp, funcOp,
                                            fusionOp->getOperands());
}

LogicalResult outlineFusionOpPattern(func::FuncOp funcOp,
                                     PatternRewriter& rewriter) {
  // Outline fusion ops one by one.
  int64_t numOutlinedFusions = 0;
  funcOp.walk([&](gml_st::FusionOp fusionOp) {
    outlineFusionOp(funcOp, fusionOp, numOutlinedFusions++, rewriter);
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
    patterns.add(outlineFusionOpPattern);

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
