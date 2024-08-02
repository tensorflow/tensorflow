/* Copyright 2024 The StableHLO Authors.
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

#include <cstdint>
#include <functional>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_CHLORECOMPOSEOPSPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

FailureOr<DictionaryAttr> getCustomCallOpAttributes(stablehlo::CustomCallOp op,
                                                    PatternRewriter& rewriter) {
  auto attrs = llvm::dyn_cast_or_null<DictionaryAttr>(
      op->getDiscardableAttr("mhlo.attributes"));
  if (!attrs)
    return rewriter.notifyMatchFailure(
        op, "Expected mhlo.attributes dictionary attribute.");
  return attrs;
}

LogicalResult verifyCustomCallOpAttributes(
    stablehlo::CustomCallOp op, PatternRewriter& rewriter,
    std::function<LogicalResult(NamedAttribute)> const& verifyFn) {
  auto attrs = getCustomCallOpAttributes(op, rewriter);
  if (failed(attrs)) return failure();

  for (auto attr : attrs->getValue()) {
    if (failed(verifyFn(attr))) return failure();
  }
  return success();
}

// Experimental, extension, and public ops in MHLO that do not exist yet in
// StableHLO can be encoded as a StableHLO CustomCallOp to allow round-tripping
// between dialects. Some of these ops are CHLO ops that are accelerated by XLA.
// For these ops we can recompose to CHLO.
//
// Example:
//  %0 = stablehlo.custom_call @mhlo.topk(...) {...}
//  ==>
//   %0 = "chlo.topk"(...) {...}
template <typename OpType>
LogicalResult recomposeChloOpFromCustomCall(stablehlo::CustomCallOp op,
                                            PatternRewriter& rewriter) {
  // Only call_target_name, backend_config, called_computations, mhlo.version,
  // and mhlo.attributes are compatible with the extensibility protocol.
  auto isSupportedAttrName = [](NamedAttribute attr) {
    auto name = attr.getName();
    return name == "call_target_name" || name == "backend_config" ||
           name == "called_computations" || name == "mhlo.attributes" ||
           name == "mhlo.version";
  };
  if (!llvm::all_of(op->getAttrs(), isSupportedAttrName) ||
      !op.hasEmptyBackendConfig()) {
    return rewriter.notifyMatchFailure(
        op, "CHLO Recompose custom call did not have required attributes.");
  }
  if (!op.getCalledComputations().empty())
    return rewriter.notifyMatchFailure(op, "Ops with regions not supported.");

  auto attrs = getCustomCallOpAttributes(op, rewriter);
  if (failed(attrs)) return failure();

  rewriter.replaceOpWithNewOp<OpType>(op, op->getResultTypes(),
                                      op->getOperands(), attrs->getValue());
  return success();
}

struct TopKOpRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "mhlo.topk") return failure();
    auto res = verifyCustomCallOpAttributes(
        op, rewriter, [&](NamedAttribute attr) -> LogicalResult {
          if (attr.getName() != "largest") return success();
          if (!cast<BoolAttr>(attr.getValue()).getValue())
            return rewriter.notifyMatchFailure(
                op, "largest = false is not supported.");
          return success();
        });
    if (failed(res)) return failure();
    return recomposeChloOpFromCustomCall<chlo::TopKOp>(op, rewriter);
  }
};

struct TanOpRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "mhlo.tan") return failure();
    return recomposeChloOpFromCustomCall<chlo::TanOp>(op, rewriter);
  }
};

struct ErfOpRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getCallTargetName() != "mhlo.erf") return failure();
    return recomposeChloOpFromCustomCall<chlo::ErfOp>(op, rewriter);
  }
};

}  // namespace

struct ChloRecomposeOpsPass
    : public impl::ChloRecomposeOpsPassBase<ChloRecomposeOpsPass> {
  using ChloRecomposeOpsPassBase::ChloRecomposeOpsPassBase;

  void runOnOperation() override {
    // Do a single traversal to recompose CustomCallOp to CHLO ops.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;
    config.maxIterations = 1;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    RewritePatternSet patterns(&getContext());
    patterns.add<TopKOpRecomposePattern>(&getContext());
    patterns.add<TanOpRecomposePattern>(&getContext());
    patterns.add<ErfOpRecomposePattern>(&getContext());

    // Only apply to CustomCallOps
    auto moduleOp = getOperation();
    llvm::SmallVector<Operation*> candidateOps;
    moduleOp.walk(
        [&](stablehlo::CustomCallOp op) { candidateOps.push_back(op); });

    if (failed(applyOpPatternsAndFold(candidateOps, std::move(patterns),
                                      config))) {
      moduleOp.emitError("Failed to converge ChloRecomposeOps in ")
          << config.maxIterations << " iterations";
      return signalPassFailure();
    }
  }
};

void createChloLegalizeToStablehloPipeline(OpPassManager& pm) {
  pm.addPass(mlir::stablehlo_ext::createChloRecomposeOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createShapeLegalizeToStablehloPass());
}

}  // namespace stablehlo_ext
}  // namespace mlir
