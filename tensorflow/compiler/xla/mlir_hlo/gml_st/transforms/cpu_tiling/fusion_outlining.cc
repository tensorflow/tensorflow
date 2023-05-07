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
#include "gml_st/transforms/transforms.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace gml_st {
namespace {

#define GEN_PASS_DEF_FUSIONOUTLININGPASS
#include "gml_st/transforms/passes.h.inc"

constexpr llvm::StringRef kFusionFunctionLabel = "fusion";
constexpr llvm::StringRef kElementwiseLabel = "__elementwise_label__";

void outlineFusionOp(func::FuncOp parentFuncOp, gml_st::FusionOp fusionOp,
                     int64_t localFusionId, PatternRewriter& rewriter) {
  Location loc = fusionOp.getLoc();
  MLIRContext* ctx = fusionOp.getContext();

  // Generate outlined fusion func ops right before the parent func op.
  rewriter.setInsertionPoint(parentFuncOp);
  std::string funcName =
      llvm::formatv("{0}_fusion_{1}", parentFuncOp.getName(), localFusionId)
          .str();
  TypeRange funcArgTypes = fusionOp->getOperandTypes();
  TypeRange funcResultTypes = fusionOp.getResultTypes();
  auto funcTy = FunctionType::get(ctx, funcArgTypes, funcResultTypes);
  auto funcOp =
      rewriter.create<func::FuncOp>(fusionOp.getLoc(), funcName, funcTy);
  setLabel(funcOp, kFusionFunctionLabel);

  // Generate entry block.
  Region& funcRegion = funcOp.getBody();
  Block* funcBlock =
      rewriter.createBlock(&funcRegion, funcRegion.begin(), funcArgTypes,
                           SmallVector<Location>(funcArgTypes.size(), loc));
  rewriter.setInsertionPointToStart(funcBlock);

  // Generate new fusion op and steal body.
  auto newFusionOp = rewriter.create<gml_st::FusionOp>(
      loc, funcResultTypes, funcBlock->getArguments(), fusionOp->getAttrs());
  newFusionOp.getRegion().takeBody(fusionOp.getRegion());

  // Forward fusion op results.
  rewriter.create<func::ReturnOp>(loc, newFusionOp->getResults());

  // Replace fusion op with a call to the newly outlined function.
  rewriter.setInsertionPoint(fusionOp);
  rewriter.replaceOpWithNewOp<func::CallOp>(fusionOp, funcOp,
                                            fusionOp->getOperands());
}

LogicalResult outlineFusionOpPattern(func::FuncOp funcOp,
                                     PatternRewriter& rewriter) {
  // Only apply to functions that are not the result of outlining.
  if (hasLabel(funcOp, kFusionFunctionLabel)) return failure();

  // Outline fusion ops one by one.
  int64_t numOutlinedFusions = 0;
  funcOp.walk([&](gml_st::FusionOp fusionOp) {
    // TODO(shyshkov): Enable outlining for elementwise clusters.
    if (hasLabel(fusionOp, kElementwiseLabel)) return;

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
