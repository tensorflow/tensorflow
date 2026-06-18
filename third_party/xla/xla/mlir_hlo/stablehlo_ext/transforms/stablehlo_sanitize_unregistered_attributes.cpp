/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.
   Copyright 2023 The StableHLO Authors.
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
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo_ext/transforms/passes.h"  // NOLINT: Used in passes.h.inc
#include "utils/unregistered_attributes.h"

#define DEBUG_TYPE "stablehlo-ext-passes"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOSANITIZEDISCARDABLEATTRIBUTESPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

// To be extra safe we error if we encounter SDY discardable attributes.
// We should run Shardy Export pass prior to sanitizing, which converts
// most SDY attributes into mhlo.frontend_attributes.
// NOLINTNEXTLINE(llvm-prefer-static-over-anonymous-namespace)
LogicalResult verifyNoSdyAttributes(Operation* op, NamedAttribute attr) {
  auto name = attr.getName();
  if (name.getValue() == "sdy.sharding_rule") {
    // TODO: b/445482443 - Figure out why sharding rule fails to convert,
    // currently dropped on the way to HLO.
    return success();
  }
  if (name.getValue().starts_with("sdy.")) {
    return op->emitError("SDY attribute encountered: ")
           << name
           << ". Run SDY export pass prior to sanitizing unregistered "
              "attributes.";
  }
  return success();
}

struct SanitizeDiscardableFuncAttributes
    : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  LogicalResult rewriteArgAttrs(
      func::FuncOp funcOp, PatternRewriter& rewriter, ArrayAttr argAttrs,
      const std::function<void(int32_t, DictionaryAttr)>& updateAttrsFn) const {
    if (!argAttrs) {
      return rewriter.notifyMatchFailure(
          funcOp, "no discardable attributes or func attrs found");
    }

    bool changed = false;
    for (auto [idx, argAttr] : llvm::enumerate(argAttrs)) {
      auto dict = mlir::dyn_cast<DictionaryAttr>(argAttr);
      if (!dict)
        return rewriter.notifyMatchFailure(funcOp, "expected dict attr");

      SmallVector<NamedAttribute> sanitizedDictAttrs;
      for (NamedAttribute attr : dict) {
        if (failed(verifyNoSdyAttributes(funcOp, attr))) return failure();
        auto name = attr.getName();
        mlir::StringRef nameRef(name.data(), name.size());
        if (xla::IsKnownDiscardableFuncAttribute(nameRef)) {
          sanitizedDictAttrs.push_back(attr);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "Removing discardable attribute: " << name << "\n");
        }
      }
      if (sanitizedDictAttrs.size() != dict.size()) {
        updateAttrsFn(
            idx, DictionaryAttr::get(funcOp.getContext(), sanitizedDictAttrs));
        changed = true;
      }
    }
    return success(/*success=*/changed);
  }

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter& rewriter) const override {
    for (auto attr : funcOp->getDiscardableAttrs()) {
      if (failed(verifyNoSdyAttributes(funcOp, attr))) return failure();

      auto name = attr.getName();
      if (!xla::IsKnownDiscardableFuncAttribute({name.data(), name.size()})) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Removing discardable func attribute: " << name << "\n");
        funcOp->removeDiscardableAttr(name);
        return success();
      }
    }

    bool changedArgs =
        succeeded(rewriteArgAttrs(funcOp, rewriter, funcOp.getAllArgAttrs(),
                                  [&](int32_t idx, DictionaryAttr dict) {
                                    funcOp.setArgAttrs(idx, dict);
                                  }));
    bool changedResults =
        succeeded(rewriteArgAttrs(funcOp, rewriter, funcOp.getAllResultAttrs(),
                                  [&](int32_t idx, DictionaryAttr dict) {
                                    funcOp.setResultAttrs(idx, dict);
                                  }));
    if (changedArgs || changedResults) return success();
    return rewriter.notifyMatchFailure(funcOp,
                                       "no discardable attributes found");
  }
};

struct SanitizeDiscardableOpAttributes final : RewritePattern {
  explicit SanitizeDiscardableOpAttributes(MLIRContext* context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    if (mlir::isa<func::FuncOp>(op))
      return rewriter.notifyMatchFailure(op, "skipping func op");

    for (auto attr : op->getDiscardableAttrs()) {
      if (failed(verifyNoSdyAttributes(op, attr))) return failure();
      auto name = attr.getName();
      if (!xla::IsKnownDiscardableOpAttribute({name.data(), name.size()})) {
        LLVM_DEBUG(llvm::dbgs()
                   << "Removing discardable op attribute: " << name << "\n");
        op->removeDiscardableAttr(name);
        return success();
      }
    }
    return rewriter.notifyMatchFailure(op, "no unregistered attributes found");
  }
};

struct StablehloSanitizeDiscardableAttributesPass
    : public impl::StablehloSanitizeDiscardableAttributesPassBase<
          StablehloSanitizeDiscardableAttributesPass> {
  using StablehloSanitizeDiscardableAttributesPassBase::
      StablehloSanitizeDiscardableAttributesPassBase;

  void runOnOperation() override {
    // Remove discardable attributes from module.
    ModuleOp module = getOperation();
    for (auto attr : module->getDiscardableAttrs()) {
      if (failed(verifyNoSdyAttributes(module, attr)))
        return signalPassFailure();
      auto name = attr.getName();
      if (!xla::IsKnownDiscardableModuleAttribute({name.data(), name.size()})) {
        LLVM_DEBUG(llvm::dbgs() << "Removing discardable module attribute: "
                                << name << "\n");
        module->removeDiscardableAttr(name);
      }
    }

    PatternRewriter rewriter(&getContext());
    RewritePatternSet patterns(&getContext());
    patterns.add<SanitizeDiscardableFuncAttributes>(&getContext());
    patterns.add<SanitizeDiscardableOpAttributes>(&getContext());
    GreedyRewriteConfig config;
    config.enableConstantCSE(false);
    config.enableFolding(false);
    if (failed(applyPatternsGreedily(module, std::move(patterns), config))) {
      module->emitError("Failed to sanitize discardable attributes.");
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace stablehlo_ext
}  // namespace mlir
