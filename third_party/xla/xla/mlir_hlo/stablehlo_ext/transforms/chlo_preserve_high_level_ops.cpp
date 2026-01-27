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
#include "mlir/IR/IRMapping.h"
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
  mlir::func::FuncOp func = mlir::func::FuncOp::create(
      builder, loc, funcName,
      builder.getFunctionType(op->getOperandTypes(), op->getResultTypes()));
  func.setPrivate();
  symbolTable.insert(func);

  Block* body = func.addEntryBlock();
  builder.setInsertionPointToStart(body);
  Operation* clonedOp = builder.clone(*op);
  clonedOp->setOperands(body->getArguments());
  mlir::func::ReturnOp::create(builder, loc, clonedOp->getResults());

  LLVM_DEBUG(llvm::dbgs() << "Created function " << func.getName() << "\n");
  return func;
}

mlir::func::FuncOp buildFuncOpFromRegion(Region& region, StringRef name,
                                         mlir::ModuleOp module) {
  mlir::SymbolTable symbolTable(module);
  mlir::OpBuilder builder(module);
  builder.setInsertionPointToEnd(&module.getBodyRegion().back());

  // Create function
  Block& entryBlock = region.front();
  TypeRange argTypes = entryBlock.getArgumentTypes();
  Operation* terminator = entryBlock.getTerminator();
  TypeRange resultTypes = terminator->getOperandTypes();

  mlir::func::FuncOp func = mlir::func::FuncOp::create(
      builder, region.getParentOp()->getLoc(), name,
      builder.getFunctionType(argTypes, resultTypes));
  func.setPrivate();
  symbolTable.insert(func);

  // Clone region
  IRMapping mapping;
  region.cloneInto(&func.getRegion(), mapping);

  // Replace stablehlo.return with func.return
  for (Block& block : func.getBody()) {
    Operation* term = block.getTerminator();
    if (term->getName().getStringRef() == "stablehlo.return") {
      builder.setInsertionPoint(term);
      mlir::func::ReturnOp::create(builder, term->getLoc(),
                                   term->getOperands());
      term->erase();
    }
  }
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
  auto composite = stablehlo::CompositeOp::create(
      builder, op->getLoc(), op->getResultTypes(), op->getOperands(),
      compositeName, compositeAttributes, compositeDecomposition,
      compositeVersion);
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

struct AcoshOpToCustomCallPattern : public OpRewritePattern<chlo::AcoshOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AcoshOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.acosh",
                                         /*version=*/1);
  }
};

struct AcosOpToCustomCallPattern : public OpRewritePattern<chlo::AcosOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AcosOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.acos",
                                         /*version=*/1);
  }
};

struct AtanhOpToCustomCallPattern : public OpRewritePattern<chlo::AtanhOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AtanhOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.atanh",
                                         /*version=*/1);
  }
};

struct CoshOpToCustomCallPattern : public OpRewritePattern<chlo::CoshOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::CoshOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.cosh",
                                         /*version=*/1);
  }
};

struct SinhOpToCustomCallPattern : public OpRewritePattern<chlo::SinhOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::SinhOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.sinh",
                                         /*version=*/1);
  }
};

struct AsinOpToCustomCallPattern : public OpRewritePattern<chlo::AsinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AsinOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.asin",
                                         /*version=*/1);
  }
};

struct AsinhOpToCustomCallPattern : public OpRewritePattern<chlo::AsinhOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AsinhOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOperationInCustomCall(rewriter, op, "mhlo.asinh",
                                         /*version=*/1);
  }
};

struct ScanOpToCustomCallPattern : public OpRewritePattern<chlo::ScanOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::ScanOp op,
                                PatternRewriter& rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    auto funcName = (op->getName().getStringRef() + ".impl").str();

    func::FuncOp calledFunc =
        buildFuncOpFromRegion(op.getBody(), funcName, module);

    auto opAttrs = serializeChloAttributes(op);
    if (failed(opAttrs)) return op->emitError("failed to serialize attributes");

    SmallVector<NamedAttribute> attributes;
    attributes.push_back(rewriter.getNamedAttr(
        "call_target_name", rewriter.getStringAttr("chlo.scan")));
    attributes.push_back(rewriter.getNamedAttr(
        "mhlo.attributes", rewriter.getDictionaryAttr(opAttrs.value())));
    attributes.push_back(
        rewriter.getNamedAttr("mhlo.version", rewriter.getI64IntegerAttr(1)));
    attributes.push_back(rewriter.getNamedAttr(
        "called_computations",
        ArrayAttr::get(getContext(), {FlatSymbolRefAttr::get(calledFunc)})));

    SmallVector<Value> operands;
    operands.append(op.getInputs().begin(), op.getInputs().end());
    operands.append(op.getInits().begin(), op.getInits().end());

    rewriter.replaceOpWithNewOp<stablehlo::CustomCallOp>(
        op, op->getResultTypes(), operands, attributes);
    return success();
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

struct AcoshOpToCompositePattern : public OpRewritePattern<chlo::AcoshOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AcoshOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct AcosOpToCompositePattern : public OpRewritePattern<chlo::AcosOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AcosOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct AtanhOpToCompositePattern : public OpRewritePattern<chlo::AtanhOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AtanhOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct CoshOpToCompositePattern : public OpRewritePattern<chlo::CoshOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::CoshOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct SinhOpToCompositePattern : public OpRewritePattern<chlo::SinhOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::SinhOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct AsinOpToCompositePattern : public OpRewritePattern<chlo::AsinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AsinOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct AsinhOpToCompositePattern : public OpRewritePattern<chlo::AsinhOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::AsinhOp op,
                                PatternRewriter& rewriter) const override {
    return wrapChloOpInComposite(op, /*version=*/1, rewriter);
  }
};

struct ScanOpToCompositePattern : public OpRewritePattern<chlo::ScanOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(chlo::ScanOp op,
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
    config.setUseTopDownTraversal(true)
        .setRegionSimplificationLevel(
            mlir::GreedySimplifyRegionLevel::Aggressive)
        .setMaxIterations(2)
        .setMaxNumRewrites(GreedyRewriteConfig::kNoLimit)
        .setStrictness(GreedyRewriteStrictness::ExistingOps);

    auto* ctx = &getContext();
    RewritePatternSet patterns(&getContext());
    // clang-format off
    if (useDeprecatedCustomCallEncoding) {
      // Deprecated CustomCall encoding.
      patterns.add<
        AcosOpToCustomCallPattern,
        AsinOpToCustomCallPattern,
        AsinhOpToCustomCallPattern,
        AcoshOpToCustomCallPattern,
        AtanhOpToCustomCallPattern,
        CoshOpToCustomCallPattern,
        SinhOpToCustomCallPattern,
        ErfOpToCustomCallPattern,
        RaggedDotOpToCustomCallPattern,
        ScanOpToCustomCallPattern,
        TopKOpToCustomCallPattern>(ctx);
    } else {
      patterns.add<
        AcosOpToCompositePattern,
        AsinOpToCompositePattern,
        AsinhOpToCompositePattern,
        AcoshOpToCompositePattern,
        AtanhOpToCompositePattern,
        CoshOpToCompositePattern,
        SinhOpToCompositePattern,
        ErfOpToCompositePattern,
        RaggedDotOpToCompositePattern,
        ScanOpToCompositePattern,
        TopKOpToCompositePattern>(ctx);
    }
    // clang-format on

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
