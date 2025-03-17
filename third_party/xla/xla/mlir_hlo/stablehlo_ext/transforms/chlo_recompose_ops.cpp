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
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/transforms/Passes.h"
#include "stablehlo_ext/transforms/passes.h"  // IWYU pragma: keep, passes.h.inc

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_CHLORECOMPOSEOPSPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

// ragged_dot_dimension_numbers
//   [[lhs_batch], [rhs_batch], [lhs_contract],
//    [rhs_contract], [lhs_ragged], [rhs_group]]
//   ==>
//   #chlo.ragged_dot<lhs_batch = [0], rhs_batch = [1], lhs_contract = [2],
//                    rhs_contract = [2], lhs_ragged = [1], rhs_group = [0]>
FailureOr<Attribute> deserializeRaggedDotDimensionNumbersAttr(
    Operation* op, NamedAttribute attr) {
  auto arrayAttr = llvm::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr || arrayAttr.size() != 6)
    return op->emitError() << "ragged_dot_dimension_numbers is not an "
                              "ArrayAttr with 6 elements";
  auto lhsBatch = llvm::dyn_cast<DenseIntElementsAttr>(arrayAttr[0]);
  auto rhsBatch = llvm::dyn_cast<DenseIntElementsAttr>(arrayAttr[1]);
  auto lhsContract = llvm::dyn_cast<DenseIntElementsAttr>(arrayAttr[2]);
  auto rhsContract = llvm::dyn_cast<DenseIntElementsAttr>(arrayAttr[3]);
  auto lhsRagged = llvm::dyn_cast<DenseIntElementsAttr>(arrayAttr[4]);
  auto rhsGroup = llvm::dyn_cast<DenseIntElementsAttr>(arrayAttr[5]);
  if (!lhsBatch || !rhsBatch || !lhsContract || !rhsContract || !lhsRagged ||
      !rhsGroup)
    return op->emitError() << "elements in ragged_dot_dimension_numbers are "
                              "not DenseIntElementsAttrs";
  return chlo::RaggedDotDimensionNumbersAttr::get(
      op->getContext(), llvm::to_vector(lhsBatch.getValues<int64_t>()),
      llvm::to_vector(rhsBatch.getValues<int64_t>()),
      llvm::to_vector(lhsContract.getValues<int64_t>()),
      llvm::to_vector(rhsContract.getValues<int64_t>()),
      llvm::to_vector(lhsRagged.getValues<int64_t>()),
      llvm::to_vector(rhsGroup.getValues<int64_t>()));
}

// precision_config
//   [["DEFAULT"], ["DEFAULT"]]
//   ==>
//   [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
FailureOr<Attribute> deserializePrecisionConfigAttr(Operation* op,
                                                    NamedAttribute attr) {
  auto arrayAttr = mlir::dyn_cast<ArrayAttr>(attr.getValue());
  if (!arrayAttr) return {};
  SmallVector<Attribute> chloAttrs;
  for (auto precisionAttr : arrayAttr) {
    auto precisionStr = mlir::dyn_cast<StringAttr>(precisionAttr);
    if (!precisionStr)
      return op->emitError()
             << "precision_config is not an ArrayAttr of StringAttrs";
    auto precisionOpt = chlo::symbolizePrecision(precisionStr.getValue());
    if (!precisionOpt.has_value())
      return op->emitError("invalid precision string");
    chloAttrs.push_back(
        chlo::PrecisionAttr::get(op->getContext(), precisionOpt.value()));
  }
  return ArrayAttr::get(op->getContext(), chloAttrs);
}

// Converts attributes serialized into builtin types to CHLO attributes.
// This is done since CHLO attributes cannot appear in VHLO.
// An alternative design would be to serialize the assembly format, but this
// approach should allow more flexibility in maintaining forward/backward
// compatibility.
FailureOr<Attribute> deserializeChloAttribute(Operation* op, StringRef opName,
                                              NamedAttribute attr) {
  if (opName == "chlo.ragged_dot") {
    if (attr.getName() == "ragged_dot_dimension_numbers")
      return deserializeRaggedDotDimensionNumbersAttr(op, attr);
    if (attr.getName() == "precision_config")
      return deserializePrecisionConfigAttr(op, attr);
  }

  // Only allow builtin attrs to pass through.
  if (attr.getValue().getDialect().getNamespace() != "builtin")
    return op->emitError() << "unsupported attribute for chlo recompose:"
                           << attr.getValue();

  // Default to passthrough
  return attr.getValue();
}

FailureOr<SmallVector<NamedAttribute>> deserializeChloAttributes(
    Operation* op, StringRef opName, DictionaryAttr attrs) {
  SmallVector<NamedAttribute> newAttrs;
  for (auto attr : attrs.getValue()) {
    auto chloAttr = deserializeChloAttribute(op, opName, attr);
    if (failed(chloAttr)) return failure();
    newAttrs.push_back({attr.getName(), chloAttr.value()});
  }
  return newAttrs;
}

/////////////
// CustomCall deserialization
/////////////

FailureOr<DictionaryAttr> getCustomCallOpAttributes(stablehlo::CustomCallOp op,
                                                    PatternRewriter& rewriter) {
  auto attrs = llvm::dyn_cast_or_null<DictionaryAttr>(
      op->getDiscardableAttr("mhlo.attributes"));
  if (!attrs)
    return rewriter.notifyMatchFailure(
        op, "Expected mhlo.attributes dictionary attribute.");
  return attrs;
}

using CustomCallAttrVerifier =
    std::function<LogicalResult(NamedAttribute, Operation*, PatternRewriter&)>;

LogicalResult verifyCustomCallOpAttributes(
    stablehlo::CustomCallOp op, PatternRewriter& rewriter,
    CustomCallAttrVerifier const& verifyFn) {
  auto attrs = getCustomCallOpAttributes(op, rewriter);
  if (failed(attrs)) return failure();

  for (auto attr : attrs->getValue()) {
    if (failed(verifyFn(attr, op, rewriter))) return failure();
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
  auto chloAttrs =
      deserializeChloAttributes(op, op.getCallTargetName(), attrs.value());
  if (failed(chloAttrs)) return failure();

  rewriter.replaceOpWithNewOp<OpType>(op, op->getResultTypes(),
                                      op->getOperands(), chloAttrs.value());
  return success();
}

/////////
// Composite deserialization patterns
////////

template <typename OpType>
LogicalResult recomposeChloOpFromCompositeOp(stablehlo::CompositeOp op,
                                             PatternRewriter& rewriter) {
  // Convert encoded attributes to CHLO attrs.
  auto attrs =
      deserializeChloAttributes(op, op.getName(), op.getCompositeAttributes());
  if (failed(attrs))
    return rewriter.notifyMatchFailure(op, "failed to deserialize attributes");
  rewriter.replaceOpWithNewOp<OpType>(op, op->getResultTypes(),
                                      op->getOperands(), attrs.value());
  return success();
}

struct RaggedDotOpRecomposePattern
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getName() != "chlo.ragged_dot")
      return rewriter.notifyMatchFailure(op, "not a chlo.ragged_dot");
    if (op.getVersion() != 1)
      return rewriter.notifyMatchFailure(
          op, "unsupported version for chlo.ragged_dot composite");
    return recomposeChloOpFromCompositeOp<chlo::RaggedDotOp>(op, rewriter);
  }
};

struct TopKOpRecomposePattern
    : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getName() != "chlo.top_k")
      return rewriter.notifyMatchFailure(op, "not a chlo.top_k");
    if (op.getVersion() != 1)
      return rewriter.notifyMatchFailure(
          op, "unsupported version for chlo.top_k composite");
    return recomposeChloOpFromCompositeOp<chlo::TopKOp>(op, rewriter);
  }
};

struct ErfOpRecomposePattern : public OpRewritePattern<stablehlo::CompositeOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CompositeOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getName() != "chlo.erf")
      return rewriter.notifyMatchFailure(op, "not a chlo.erf");
    if (op.getVersion() != 1)
      return rewriter.notifyMatchFailure(
          op, "unsupported version for chlo.erf composite");
    return recomposeChloOpFromCompositeOp<chlo::ErfOp>(op, rewriter);
  }
};

/////////
// (Deprecated) Custom call patterns
////////

LogicalResult defaultAttrVerifier(NamedAttribute, Operation*,
                                  PatternRewriter&) {
  return success();
}

template <typename ChloOpType>
LogicalResult recomposeChloOpFromCustomCall(
    stablehlo::CustomCallOp op, ArrayRef<StringRef> customCallNames,
    PatternRewriter& rewriter,
    CustomCallAttrVerifier const& verifyFn = defaultAttrVerifier) {
  StringRef customCallName = customCallNames[0];
  if (!llvm::is_contained(customCallNames, op.getCallTargetName()))
    return rewriter.notifyMatchFailure(
        op, "not a CHLO custom call for " + customCallName);
  if (failed(verifyCustomCallOpAttributes(op, rewriter, verifyFn)))
    return failure();
  return recomposeChloOpFromCustomCall<ChloOpType>(op, rewriter);
}

struct RaggedDotOpCustomCallRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<StringRef> customCallNames = {"chlo.ragged_dot"};
    return recomposeChloOpFromCustomCall<chlo::RaggedDotOp>(op, customCallNames,
                                                            rewriter);
  }
};

struct TopKOpCustomCallRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    SmallVector<StringRef> customCallNames = {"mhlo.topk", "chlo.top_k"};
    return recomposeChloOpFromCustomCall<chlo::TopKOp>(
        op, customCallNames, rewriter, verifyOpAttributes);
  }

  static LogicalResult verifyOpAttributes(NamedAttribute attr, Operation* op,
                                          PatternRewriter& rewriter) {
    if (attr.getName() != "largest") return success();
    if (!cast<BoolAttr>(attr.getValue()).getValue())
      return rewriter.notifyMatchFailure(op,
                                         "largest = false is not supported.");
    return success();
  }
};

struct TanOpCustomCallRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    return recomposeChloOpFromCustomCall<chlo::TanOp>(op, {"mhlo.tan"},
                                                      rewriter);
  }
};

struct ErfOpCustomCallRecomposePattern
    : public OpRewritePattern<stablehlo::CustomCallOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(stablehlo::CustomCallOp op,
                                PatternRewriter& rewriter) const override {
    return recomposeChloOpFromCustomCall<chlo::ErfOp>(
        op, {"mhlo.erf", "chlo.erf"}, rewriter);
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
    // CustomCall Patterns
    patterns.add<ErfOpCustomCallRecomposePattern>(&getContext());
    patterns.add<RaggedDotOpCustomCallRecomposePattern>(&getContext());
    patterns.add<TanOpCustomCallRecomposePattern>(&getContext());
    patterns.add<TopKOpCustomCallRecomposePattern>(&getContext());

    // Composite Patterns
    patterns.add<ErfOpRecomposePattern>(&getContext());
    patterns.add<RaggedDotOpRecomposePattern>(&getContext());
    patterns.add<TopKOpRecomposePattern>(&getContext());

    // Only apply to CustomCallOps
    auto moduleOp = getOperation();
    llvm::SmallVector<Operation*> candidateOps;
    moduleOp.walk(
        [&](stablehlo::CustomCallOp op) { candidateOps.push_back(op); });
    moduleOp.walk(
        [&](stablehlo::CompositeOp op) { candidateOps.push_back(op); });

    if (failed(applyOpPatternsGreedily(candidateOps, std::move(patterns),
                                       config))) {
      moduleOp.emitError("Failed to converge ChloRecomposeOps in ")
          << config.maxIterations << " iterations";
      return signalPassFailure();
    }
  }
};

void createChloLegalizeToStablehloPipeline(OpPassManager& pm) {
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo_ext::createChloRecomposeOpsPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createChloLegalizeToStablehloPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::stablehlo::createShapeLegalizeToStablehloPass());
}

}  // namespace stablehlo_ext
}  // namespace mlir
