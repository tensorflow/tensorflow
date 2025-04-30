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
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "shardy/dialect/sdy/ir/dialect.h"
#include "shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/Base.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/StablehloRefineShapes.h"
#include "stablehlo/transforms/optimization/Passes.h"
#include "stablehlo_ext/transforms/stablehlo_refine_shapes.h"

namespace mlir {
namespace stablehlo_ext {

namespace {

template <typename OpTy>
void refineBlockArguments(OpTy regionOp, TypeRange refinedTypes) {
  Region& body = regionOp.getBody();
  OpBuilder builder(body);
  for (int64_t i = 0; i < body.getNumArguments(); ++i) {
    auto arg = body.getArgument(i);
    arg.setType(refinedTypes[i]);
  }
}

// Refines the values using the given types.
//
// This is similar to `stablehlo::refineValues`, but the problem is that
// `hlo::inferMostSpecificType` doesn't account for the block argument types
// differing from the operand types due to the body having the local types.
// So to figure out the more specific type, we transform the refinement of
// the operand to the local refinement.
//
// For example:
//
// ```
// %0 = sdy.manual_computation(%0)
//   in_shardings=[<@mesh, [{"x"}, {}]>]
//   out_shardings=[<@mesh, [{"x"}, {}]>]
//   manual_axes={"x"} (%arg1: tensor<2x?xf32>) {
//   ...
// } : (tensor<4x?xf32>) -> tensor<4x?xf32>
// ```
//
// The global and local types differ for the known static dimension of the
// operand, so we need to convert the global refinement to the local refinement
// to figure out the more specific type.
LogicalResult refineValues(
    PatternRewriter& rewriter, sdy::ManualComputationOp manualComputation,
    ArrayRef<BlockArgument> blockArguments, TypeRange types,
    sdy::MeshAttr mesh) {
  if (blockArguments.size() != types.size()) {
    return rewriter.notifyMatchFailure(
        manualComputation, [&](Diagnostic& diag) {
      diag << "refineValues failed for " << types << ": expected "
           << blockArguments.size() << " types, got " << types.size();
    });
  }

  // Check whether `types` contain any new information with respect to
  // existing return types. Even if just a single dimension size out of an
  // entire tensor type got updated, using `inferMostSpecificType` ensures
  // that we don't miss that.
  bool needsRefinement = false;
  SmallVector<Type> refinedTypes;
  for (auto it : llvm::zip(blockArguments, types)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    BlockArgument blockArg = std::get<0>(it);
    Type blockArgType = blockArg.getType();
    auto refinement = cast<RankedTensorType>(std::get<1>(it));
    // inferMostSpecificType cannot account for the fact that the operand and
    // block arg types differ for their known static dimensions due to the body
    // having the local types. So to figure out the more specific type,
    // transform the refinement of the operand to the local refinement.
    sdy::TensorShardingAttr inSharding = eraseFreeAxes(
        manualComputation.getInSharding(blockArg.getArgNumber()),
        manualComputation.getManualAxes());
    auto refinedType = hlo::inferMostSpecificType(
        /*location=*/{}, {blockArgType, inSharding.getLocalTensorType(
          refinement, mesh)});
    if (failed(refinedType)) {
      return rewriter.notifyMatchFailure(manualComputation,
                                         [&](Diagnostic& diag) {
        diag << "inferMostSpecificType failed for " << blockArgType << " and "
             << refinement;
      });
    }
    refinedTypes.push_back(*refinedType);
    needsRefinement |= (blockArgType != *refinedType);
  }
  if (!needsRefinement)
    return rewriter.notifyMatchFailure(
        manualComputation, "doesn't need refinement");

  for (auto it : llvm::zip(blockArguments, refinedTypes)) {
    // Cannot use structured bindings to simplify this because capturing
    // structured bindings in a lambda is a C++ 20 extension.
    auto value = std::get<0>(it);
    auto refinedType = std::get<1>(it);
    if (value.getType() == refinedType) continue;

    // Check whether the users of this value are ready for the type of the
    // value to be refined.
    for (Operation* user : value.getUsers()) {
      // CHLO and StableHLO ops are designed to support type refinements of
      // their operands and results. Any operand type in these ops can change
      // within what's supported by `inferMostSpecificType` without breaking
      // verification of the op.
      if (isa<chlo::ChloDialect, stablehlo::StablehloDialect, sdy::SdyDialect>(
              user->getDialect()))
        continue;

      // Simply changing operand type of `func.return` won't work because
      // that won't update the FunctionType of the enclosing `func.func`.
      if (isa<func::ReturnOp>(user)) continue;
      if (isa<func::CallOp>(user)) continue;

      // Unlike in TensorFlow's type inference pass, here we work only with
      // allowlisted ops to focus our support on well-defined semantics of
      // StableHLO programs.
      return rewriter.notifyMatchFailure(
          manualComputation, [&](Diagnostic& diag) {
        diag << "unsupported refinement: tried to refine " << value.getType()
             << " to " << refinedType << " for user " << user;
      });
    }

    // Happy path: simply call setType here because most of our users are
    // fine with that.
    auto unrefinedType = value.getType();
    value.setType(refinedType);

    // Special case: for `func.return`, guard the refinement with a cast
    // and leave propagation of the refined return type to a dedicated pattern.
    auto isFuncReturn = [](OpOperand& use) -> bool {
      return isa<func::ReturnOp>(use.getOwner());
    };
    if (llvm::none_of(value.getUses(), isFuncReturn)) continue;
    rewriter.setInsertionPointAfter(manualComputation);
    auto castToUnrefinedType = rewriter.create<UnrealizedConversionCastOp>(
        manualComputation->getLoc(), unrefinedType, value);
    value.replaceUsesWithIf(castToUnrefinedType.getOutputs()[0], isFuncReturn);
  }

  return success();
}

LogicalResult refineArguments(sdy::ManualComputationOp manualComputation,
                              TypeRange refinedTypes,
                              sdy::MeshAttr mesh,
                              PatternRewriter& rewriter) {
  // Verify that refinements are valid
  if (failed(stablehlo::validateRefinedTypes(
          manualComputation, manualComputation.getBody().getArgumentTypes(),
          refinedTypes)))
    return failure();

  if (failed(refineValues(rewriter, manualComputation,
                          manualComputation.getBody().getArguments(),
                          manualComputation.getOperandTypes(), mesh))) {
    return failure();
  }

  // Actually update block argument types.
  refineBlockArguments(manualComputation, refinedTypes);

  return success();
}

LogicalResult refineArguments(sdy::NamedComputationOp namedComputation,
                              TypeRange refinedTypes,
                              PatternRewriter& rewriter) {
  // Verify that refinements are valid
  if (failed(stablehlo::validateRefinedTypes(
          namedComputation, namedComputation.getBody().getArgumentTypes(),
          refinedTypes)))
    return failure();

  if (failed(stablehlo::refineValues(rewriter, namedComputation,
                          namedComputation.getBody().getArguments(),
                          namedComputation.getOperandTypes()))) {
    return failure();
  }

  // Actually update block argument types.
  refineBlockArguments(namedComputation, refinedTypes);

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

// Applies shape refinement patterns to `ManualComputationOp`.
// NOTE: This will only work when the program is inlined. Refining `CallOp`s
// nested in `ManualComputationOp`s would require keeping track of scope, which
// requires more refactoring of the base pass. Likely requiring exposing
// `RefinementState` and a method for invoking all patterns available on
// the `ManualComputationOp` body.
template <typename OpTy>
LogicalResult applyShapeRefinementPatterns(OpTy regionOp) {
  MLIRContext* context = regionOp.getContext();
  RewritePatternSet patterns(context);
  GreedyRewriteConfig config;

  // The algorithm behind this pass consists of a single traversal of the
  // function. This is sufficient because we only support one function per
  // program at the moment.
  // TODO(#1048): Find out why .setMaxIterations(1) no longer works.
  // There have been recent refactors to applyPatternsGreedily
  // upstream, and that might be the reason.
  config.setUseTopDownTraversal(true)
      .setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive)
      .setMaxIterations(2)
      .setMaxNumRewrites(GreedyRewriteConfig::kNoLimit)
      .setStrictness(GreedyRewriteStrictness::AnyOp);

  populateStablehloExtRefineShapesPatterns(&patterns, context);
  patterns.add<RefineManualComputationOpPattern>(context);

  // The folding patterns implement partial evaluation of shape computations
  // which is a critical part of implementing type refinement for ops like
  // dynamic_broadcast_in_dim, dynamic_iota and dynamic_reshape whose shape
  // depends on the value of their shape operands.
  stablehlo::populateStablehloShapeFolderPatterns(&patterns, context);

  if (failed(applyPatternsGreedily(regionOp, std::move(patterns), config)))
    regionOp.emitError("Failed to converge StablehloRefineShapes in ")
        << config.getMaxIterations() << " iterations";

  return success();
}

// Manual computations need to be refined in-order like function calls
// since the output shape depends on the shape of the return in the
// ops body region, with some transformation based on the mesh, out shardings,
// and manual axes.
//
// For example, if the result is `tensor<?xf32>`, the op is manual on axis
// `x=2`, and the resut has sharding `<@mesh, [{"x"}]>`, then if the local type
// in the body region is `tensor<4xf32>`, the global type is `tensor<8xf32>`.
// And so we need to update the global result type of the manual computation
// to be `tensor<8xf32>` to reflect the actual shape of the result.
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

  if (failed(refineArguments(manualComputation, localBlockArgTypes,
                             mesh, rewriter)))
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

LogicalResult refineNamedComputationOpPattern(
    sdy::NamedComputationOp namedComputation, PatternRewriter& rewriter);

struct RefineNamedComputationOpPattern
    : public OpRewritePattern<sdy::NamedComputationOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(sdy::NamedComputationOp op,
                                PatternRewriter& rewriter) const override {
    return refineNamedComputationOpPattern(op, rewriter);
  }
};

LogicalResult refineNamedComputationOpPattern(
    sdy::NamedComputationOp namedComputation, PatternRewriter& rewriter) {
  rewriter.setInsertionPointToStart(&namedComputation.getRegion().front());

  SymbolTable symbolTable(namedComputation->getParentOfType<ModuleOp>());

  if (failed(refineArguments(namedComputation,
                             namedComputation.getOperandTypes(), rewriter)))
    return failure();

  // Now iterate into the function body and apply refinement patterns.
  if (failed(applyShapeRefinementPatterns(namedComputation))) return failure();

  // TODO(bartchr): Should be able to call `getBodyTerminatorOpOperandTypes`
  // but getting a refined template compilation error.
  return stablehlo::refineReturnTypes(
      rewriter, namedComputation,
      llvm::to_vector(namedComputation.getBody().front().getTerminator()
                      ->getOperandTypes()));
}

struct RefineInferTypeOpInterfacePattern
    : public OpInterfaceRewritePattern<InferTypeOpInterface> {
  explicit RefineInferTypeOpInterfacePattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context, /*benefit=*/0) {}
  LogicalResult matchAndRewrite(InferTypeOpInterface op,
                                PatternRewriter& rewriter) const override {
    // Unlike in TensorFlow's type inference pass, here we work only with
    // allowlisted ops to focus our support on well-defined semantics of
    // StableHLO programs.
    if (!isa<sdy::SdyDialect>(
            op->getDialect()))
      return rewriter.notifyMatchFailure(op, "unsupported dialect");

    // For the ops that implement InferTypeOpInterface, we reinfer their return
    // types and see what happens.
    // Operands of these ops might have been refined elsewhere (e.g. someone
    // might have updated argument types of a function) or earlier during this
    // pass, and this might enable refinement opportunities downstream.
    SmallVector<Type> inferredReturnTypes;
    if (failed(op.inferReturnTypes(getContext(), /*location=*/{},
                                   op->getOperands(), op->getAttrDictionary(),
                                   op->getPropertiesStorage(), op->getRegions(),
                                   inferredReturnTypes)))
      return rewriter.notifyMatchFailure(op, "inferReturnTypes failed");
    return stablehlo::refineReturnTypes(rewriter, op, inferredReturnTypes);
  }
};

}  // namespace

/// Patterns for refining shapes of Shardy ops.
void populateSdyShapeRefinementPatterns(RewritePatternSet* patterns,
                                        MLIRContext* context) {
  patterns->add<RefineManualComputationOpPattern>(context);
  patterns->add<RefineNamedComputationOpPattern>(context);
  patterns->add<RefineInferTypeOpInterfacePattern>(context);
}

}  // namespace stablehlo_ext
}  // namespace mlir
