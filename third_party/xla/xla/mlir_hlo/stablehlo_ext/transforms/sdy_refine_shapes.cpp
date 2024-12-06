/* Copyright 2023 The StableHLO Authors.

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

#include "stablehlo_ext/transforms/sdy_refine_shapes.h"

#include <cstdint>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"

namespace mlir {
namespace stablehlo_ext {

namespace {

void refineBlockArguments(sdy::ManualComputationOp manualComputation,
                          TypeRange refinedTypes) {
  Region& body = manualComputation.getBody();
  OpBuilder builder(body);
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    auto arg = body.getArgument(i);
    arg.setType(refinedTypes[i]);
  }
}

LogicalResult refineArguments(sdy::ManualComputationOp manualComputation,
                              TypeRange refinedTypes,
                              PatternRewriter& rewriter) {
  // Verify that refinements are valid
  if (failed(stablehlo::validateRefinedTypes(
          manualComputation, manualComputation.getBody().getArgumentTypes(),
          refinedTypes)))
    return failure();

  if (failed(stablehlo::refineValues(rewriter, manualComputation,
                                     manualComputation.getBody().getArguments(),
                                     manualComputation.getOperandTypes()))) {
    return failure();
  }

  // Actually update block argument types.
  refineBlockArguments(manualComputation, refinedTypes);

  return success();
}

LogicalResult refineManualComputationBody(
    sdy::ManualComputationOp manualComputation, PatternRewriter& rewriter);

struct RefineManualComputationOpPattern
    : public OpRewritePattern<sdy::ManualComputationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(sdy::ManualComputationOp op,
                                PatternRewriter& rewriter) const override {
    return refineManualComputationBody(op, rewriter);
  }
};

LogicalResult applyShapeRefinementPatterns(
    sdy::ManualComputationOp manualComputation) {
  MLIRContext* context = manualComputation.getContext();
  RewritePatternSet patterns(context);
  GreedyRewriteConfig config;

  // The algorithm behind this pass consists of a single traversal of the
  // function. This is sufficient because we only support one function per
  // program at the moment.
  // TODO(#1048): Find out why .maxIterations = 1 no longer works.
  // There have been recent refactors to applyPatternsAndFoldGreedily
  // upstream, and that might be the reason.
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;
  config.maxIterations = 2;
  config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
  config.strictMode = GreedyRewriteStrictness::AnyOp;

  stablehlo::populateStablehloRefineShapesPatterns(&patterns, context);
  patterns.add<RefineManualComputationOpPattern>(context);

  // The folding patterns implement partial evaluation of shape computations
  // which is a critical part of implementing type refinement for ops like
  // dynamic_broadcast_in_dim, dynamic_iota and dynamic_reshape whose shape
  // depends on the value of their shape operands.
  stablehlo::populateStablehloShapeFolderPatterns(&patterns, context);

  if (failed(applyPatternsAndFoldGreedily(manualComputation,
                                          std::move(patterns), config)))
    manualComputation.emitError("Failed to converge StablehloRefineShapes in ")
        << config.maxIterations << " iterations";

  return success();
}

LogicalResult refineManualComputationBody(
    sdy::ManualComputationOp manualComputation, PatternRewriter& rewriter) {
  rewriter.setInsertionPointToStart(&manualComputation.getRegion().front());

  SymbolTable symbolTable(manualComputation->getParentOfType<ModuleOp>());
  ArrayRef<StringAttr> manualAxes = manualComputation.getManualAxes();
  sdy::MeshAttr mesh = sdy::getCommonMesh(
      manualComputation.getInShardings().getShardings(),
      manualComputation.getOutShardings().getShardings(), symbolTable);

  // Convert the global types to local types using the sharding consisting only
  // of manual axes.
  SmallVector<Type> localBlockArgTypes;
  localBlockArgTypes.reserve(manualComputation.getNumOperands());
  for (auto [arg, globalType, inSharding] :
       llvm::zip_equal(manualComputation.getBody().getArguments(),
                       manualComputation->getOperandTypes(),
                       manualComputation.getInShardings().getShardings())) {
    localBlockArgTypes.push_back(
        sdy::eraseFreeAxes(inSharding, manualAxes)
            .getLocalTensorType(cast<RankedTensorType>(globalType), mesh));
  }

  if (failed(refineArguments(manualComputation, localBlockArgTypes, rewriter)))
    return failure();

  // Now iterate into the function body and apply refinement patterns.
  if (failed(applyShapeRefinementPatterns(manualComputation))) return failure();

  // Convert the local types to global types using the sharding consisting only
  // of manual axes.
  SmallVector<Type> globalResultTypes;
  globalResultTypes.reserve(manualComputation.getNumResults());
  for (auto [localType, sharding] :
       llvm::zip_equal(sdy::getBodyTerminatorOpOperandTypes(manualComputation),
                       manualComputation.getOutShardings().getShardings())) {
    globalResultTypes.push_back(
        sdy::eraseFreeAxes(sharding, manualAxes)
            .getGlobalTensorType(cast<RankedTensorType>(localType), mesh));
  }

  return stablehlo::refineReturnTypes(rewriter, manualComputation,
                                      globalResultTypes);
}

struct RefineNamedComputationOpPattern
    : public OpRewritePattern<sdy::NamedComputationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(sdy::NamedComputationOp op,
                                PatternRewriter& rewriter) const override {
    rewriter.startOpModification(op);
    LogicalResult bodyStatus = stablehlo::refineValues(
        rewriter, op, op.getBody().getArguments(), op.getOperandTypes());
    if (succeeded(bodyStatus)) {
      rewriter.finalizeOpModification(op);
      return success();
    }
    rewriter.cancelOpModification(op);
    return failure();
  }
};

}  // namespace

/// Patterns for refining shapes of Shardy ops.
void populateSdyShapeRefinementPatterns(RewritePatternSet* patterns,
                                        MLIRContext* context) {
  patterns->add<RefineManualComputationOpPattern>(context);
  patterns->add<RefineNamedComputationOpPattern>(context);
}

}  // namespace stablehlo_ext
}  // namespace mlir
