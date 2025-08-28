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

#include <utility>

#include "llvm/ADT/STLExtras.h"
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

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_STABLEHLOSANITIZEDISCARDABLEATTRIBUTESPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

// To be extra safe we error if we encounter SDY discardable attributes.
// We should run Shardy Export pass prior to sanitizing, which converts
// most SDY attributes into mhlo.frontend_attributes.
LogicalResult verifyNoSdyAttributes(Operation* op, NamedAttribute attr) {
  auto name = attr.getName();
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

  LogicalResult matchAndRewrite(func::FuncOp funcOp,
                                PatternRewriter& rewriter) const override {
    for (auto attr : funcOp->getDiscardableAttrs()) {
      if (failed(verifyNoSdyAttributes(funcOp, attr))) return failure();

      auto name = attr.getName();
      if (!xla::IsKnownDiscardableFuncAttribute({name.data(), name.size()})) {
        funcOp->removeDiscardableAttr(name);
        return success();
      }
    }

    auto argAttrs = funcOp.getArgAttrs();
    if (!argAttrs.has_value())
      return rewriter.notifyMatchFailure(funcOp,
                                         "no discardable attributes found");

    bool changed = false;
    for (auto [idx, argAttr] : llvm::enumerate(argAttrs.value())) {
      auto dict = mlir::dyn_cast<DictionaryAttr>(argAttr);
      if (!dict)
        return rewriter.notifyMatchFailure(funcOp, "expected dict attr");

      SmallVector<NamedAttribute> sanitized_dict_attrs;
      for (NamedAttribute attr : dict) {
        if (failed(verifyNoSdyAttributes(funcOp, attr))) return failure();
        auto name = attr.getName();
        mlir::StringRef nameRef(name.data(), name.size());
        if (xla::IsKnownDiscardableFuncAttribute(nameRef)) {
          sanitized_dict_attrs.push_back(attr);
        }
      }
      if (sanitized_dict_attrs.size() != dict.size()) {
        funcOp.setArgAttrs(idx, DictionaryAttr::get(funcOp.getContext(),
                                                    sanitized_dict_attrs));
        changed = true;
      }
      if (changed) return success();
    }

    return rewriter.notifyMatchFailure(funcOp,
                                       "no unregistered attributes found");
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
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      module->emitError("Failed to sanitize discardable attributes.");
      return signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace stablehlo_ext
}  // namespace mlir
