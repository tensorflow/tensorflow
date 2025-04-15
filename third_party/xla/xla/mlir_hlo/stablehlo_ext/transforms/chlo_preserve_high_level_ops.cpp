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
#include <type_traits>
#include <utility>

#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo_ext/transforms/passes.h"  // IWYU pragma: keep, passes.h.inc

#define DEBUG_TYPE "stablehlo-ext-chlo"

namespace mlir {
namespace stablehlo_ext {

#define GEN_PASS_DEF_CHLOPRESERVEHIGHLEVELOPSPASS
#include "stablehlo_ext/transforms/passes.h.inc"

namespace {

/////////
// Composite Builder functions
// TODO: OSS these in StableHLO and use the functions from there.
///////////

// functionality of the provided `implOp`. The new function is named uniquely
// and is set to private visibility.
mlir::func::FuncOp buildFuncOpWrappingOperation(mlir::Operation* op,
                                                mlir::ModuleOp module) {
  mlir::SymbolTable symbolTable(module);

  // Create an OpBuilder, insertion point at the end of module's body.
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create the function operation, set private and add to the symbol table.
  // SymbolTable will resolve all name conflicts.
  Location loc = op->getLoc();
  auto funcName = (op->getName().getStringRef() + ".impl").str();
  mlir::func::FuncOp func = builder.create<mlir::func::FuncOp>(
      loc, funcName,
      builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
  func.setPrivate();
  symbolTable.insert(func);

  Block* body = func.addEntryBlock();
  builder.setInsertionPointToStart(body);
  Operation* clonedOp = builder.clone(*op);
  clonedOp->setOperands(body->getArguments());
  builder.create<mlir::func::ReturnOp>(loc, clonedOp->getResults());

  LLVM_DEBUG(llvm::dbgs() << "Created function " << func.getName() << "\n");
  return func;
}

stablehlo::CompositeOp wrapOperationInComposite(OpBuilder& builder,
                                                Operation* op,
                                                const NamedAttrList& attrs,
                                                int32_t version,
                                                ModuleOp module) {
  func::FuncOp decomposition = buildFuncOpWrappingOperation(op, module);
  auto compositeName = op->getName().getStringRef();
  auto compositeAttributes = builder.getDictionaryAttr(attrs);
  auto compositeVersion = version;
  auto compositeDecomposition = decomposition.getSymName();
  auto composite = builder.create<stablehlo::CompositeOp>(
      op->getLoc(), op->getResultTypes(), op->getOperands(), compositeName,
      compositeAttributes, compositeDecomposition, compositeVersion);
  return composite;
}

//////////
// CHLO Attribute Serialization
//////////

// ragged_dot_dimension_numbers
//   #chlo.ragged_dot<lhs_batch = [0], rhs_batch = [1], lhs_contract = [2],
//                    rhs_contract = [2], lhs_ragged = [1], rhs_group = [0]>
//   ==>
//   [[lhs_batch], [rhs_batch], [lhs_contract],
//    [rhs_contract], [lhs_ragged], [rhs_group]]
FailureOr<Attribute> serializeRaggedDotDimensionNumbersAttr(
    chlo::RaggedDotOp op, chlo::RaggedDotDimensionNumbersAttr attr) {
  OpBuilder builder(op);
  return builder.getArrayAttr({
      builder.getI64TensorAttr(attr.getLhsBatchingDimensions()),
      builder.getI64TensorAttr(attr.getRhsBatchingDimensions()),
      builder.getI64TensorAttr(attr.getLhsContractingDimensions()),
      builder.getI64TensorAttr(attr.getRhsContractingDimensions()),
      builder.getI64TensorAttr(attr.getLhsRaggedDimensions()),
      builder.getI64TensorAttr(attr.getRhsGroupDimensions()),
  });
}

// precision_config
//   [["DEFAULT"], ["DEFAULT"]]
//   ==>
//   [#chlo<precision DEFAULT>, #chlo<precision DEFAULT>]
FailureOr<Attribute> serializePrecisionConfigAttr(chlo::RaggedDotOp op,
                                                  ArrayAttr attr) {
  SmallVector<Attribute> stringAttrs;
  for (auto hloAttr : attr) {
    auto precisionAttr = llvm::cast<chlo::PrecisionAttr>(hloAttr);
    if (!precisionAttr)
      return op->emitError() << "precision_config is not an ArrayAttr of "
                                "chlo.precision attributes";
    StringRef precisionStr = chlo::stringifyPrecision(precisionAttr.getValue());
    if (precisionStr.empty())
      return op->emitError() << "invalid CHLO precision attribute";
    stringAttrs.push_back(StringAttr::get(hloAttr.getContext(), precisionStr));
  }
  return ArrayAttr::get(op.getContext(), stringAttrs);
}

template <typename ChloOpTy>
FailureOr<Attribute> serializeChloAttribute(ChloOpTy op, NamedAttribute attr) {
  // Handle RaggedDotOp.
  if constexpr (std::is_same<ChloOpTy, chlo::RaggedDotOp>::value) {
    if (auto raggedDotAttr =
            llvm::dyn_cast<chlo::RaggedDotDimensionNumbersAttr>(
                attr.getValue()))
      return serializeRaggedDotDimensionNumbersAttr(op, raggedDotAttr);
    if (attr.getName() == "precision_config") {
      return serializePrecisionConfigAttr(
          op, llvm::cast<ArrayAttr>(attr.getValue()));
    }
  }

  // Allow passthrough of builtin attributes.
  if (attr.getValue().getDialect().getNamespace() != "builtin")
    return op->emitError(
               "unsupported dialect attribute for CHLO preservation: ")
           << attr.getName() << " of dialect "
           << attr.getValue().getDialect().getNamespace();
  return attr.getValue();
}

template <typename ChloOpTy>
FailureOr<SmallVector<NamedAttribute>> serializeChloAttributes(ChloOpTy op) {
  SmallVector<NamedAttribute> newAttrs;
  for (auto attr : op->getAttrs()) {
    auto serializedAttr = serializeChloAttribute(op, attr);
    if (failed(serializedAttr)) return failure();
    newAttrs.emplace_back(attr.getName(), serializedAttr.value());
  }
  return newAttrs;
}

////////
// (Deprecated) Delete after 12w from submit and flip to composite approach.
// CHLO to CustomCallOp
////////

// Needs template since serialization uses constexpr logic.
template <typename ChloOpTy>
LogicalResult wrapChloOperationInCustomCall(PatternRewriter& rewriter,
                                            ChloOpTy op,
                                            StringRef encodedOpName,
                                            int32_t version) {
  auto opAttrs = serializeChloAttributes(op);
  if (failed(opAttrs)) return op->emitError("failed to serialize attributes");

  SmallVector<NamedAttribute> chloAttributes;
  chloAttributes.push_back(rewriter.getNamedAttr(
      "call_target_name", rewriter.getStringAttr(encodedOpName)));
  chloAttributes.push_back(rewriter.getNamedAttr(
      "mhlo.attributes", rewriter.getDictionaryAttr(opAttrs.value())));
  chloAttributes.push_back(rewriter.getNamedAttr(
      "mhlo.version", rewriter.getI64IntegerAttr(version)));
  rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
      op, op->getResultTypes(), op->getOperands(), chloAttributes);
  return success();
}

struct RaggedDotOpToCustomCallPattern
    : public OpRewritePattern<chlo::RaggedDotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::RaggedDotOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "chlo.ragged_dot",
                                         /*version=*/1);
  }
};

struct TopKOpToCustomCallPattern : public OpRewritePattern<chlo::TopKOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::TopKOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.topk",
                                         /*version=*/1);
  }
};

struct ErfOpToCustomCallPattern : public OpRewritePattern<chlo::ErfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::ErfOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.erf",
                                         /*version=*/1);
  }
};

///////
// CHLO to CompositeOp Patterns
///////

// Needs template since serialization uses constexpr logic.
template <typename ChloOpTy>
LogicalResult wrapChloOpInComposite(ChloOpTy op, int32_t version,
                                    PatternRewriter& rewriter) {
  auto compositeAttrs = serializeChloAttributes(op);
  if (failed(compositeAttrs))
    return op->emitError("failed to serialize attributes");
  auto composite =
      wrapOperationInComposite(rewriter, op, compositeAttrs.value(), version,
                               (*op).template getParentOfType<ModuleOp>());
  rewriter.replaceOp(op, composite.getResults());
  return success();
}

struct RaggedDotOpToCompositePattern
    : public OpRewritePattern<chlo::RaggedDotOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::RaggedDotOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct TopKOpToCompositePattern : public OpRewritePattern<chlo::TopKOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::TopKOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct ErfOpToCompositePattern : public OpRewritePattern<chlo::ErfOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::ErfOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

}  // namespace

struct ChloPreserveHighLevelOpsPass
    : public impl::ChloPreserveHighLevelOpsPassBase<
          ChloPreserveHighLevelOpsPass> {
  using ChloPreserveHighLevelOpsPassBase::ChloPreserveHighLevelOpsPassBase;

  void runOnOperation() override {
    // Do a single traversal to recompose CustomCallOp to CHLO ops.
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.enableRegionSimplification = GreedySimplifyRegionLevel::Aggressive;
    config.maxIterations = 2;
    config.maxNumRewrites = GreedyRewriteConfig::kNoLimit;
    config.strictMode = GreedyRewriteStrictness::ExistingOps;

    RewritePatternSet patterns(&getContext());
    if (useDeprecatedCustomCallEncoding) {
      // Deprecated CustomCall encoding.
      patterns.add<RaggedDotOpToCustomCallPattern>(patterns.getContext());
      patterns.add<TopKOpToCustomCallPattern>(&getContext());
      patterns.add<ErfOpToCustomCallPattern>(&getContext());
    } else {
      patterns.add<RaggedDotOpToCompositePattern>(patterns.getContext());
      patterns.add<TopKOpToCompositePattern>(&getContext());
      patterns.add<ErfOpToCompositePattern>(&getContext());
    }

    // Only apply to CustomCallOps
    auto moduleOp = getOperation();
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns), config))) {
      moduleOp.emitError("Failed to converge ChloPreserveHighLevelOpsPass in ");
      return signalPassFailure();
    }
  }
};

}  // namespace stablehlo_ext
}  // namespace mlir
