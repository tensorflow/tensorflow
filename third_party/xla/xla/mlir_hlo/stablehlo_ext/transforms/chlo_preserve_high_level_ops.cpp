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

#include "llvm/ADT/STLExtras.h"
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
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
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

template <typename ChloOpTy>
struct ChloOpToCompositePattern : public OpRewritePattern<ChloOpTy> {
  ChloOpToCompositePattern(MLIRContext* context, SymbolTable& symbolTable,
                           int32_t version)
      : OpRewritePattern<ChloOpTy>(context),
        symbolTable(symbolTable),
        version(version) {};

  LogicalResult matchAndRewrite(ChloOpTy op,
                                PatternRewriter& rewriter) const override {
    FailureOr<SmallVector<NamedAttribute>> compositeAttrs =
        serializeChloAttributes(op);
    if (failed(compositeAttrs))
      return op->emitError("failed to serialize attributes");

    Location loc = op->getLoc();

    // Create the composite operation and update uses.
    auto composite = stablehlo::CompositeOp::create(
        rewriter, loc, op->getResultTypes(), op->getOperands(),
        op->getName().getStringRef(),
        rewriter.getDictionaryAttr(*compositeAttrs), /*decomposition=*/{},
        version, op->getNumRegions());
    for (auto [srcRegion, dstRegion] :
         llvm::zip_equal(op->getRegions(), composite.getCompositeRegions())) {
      rewriter.cloneRegionBefore(srcRegion, dstRegion, dstRegion.end());
    }
    rewriter.replaceAllOpUsesWith(op, composite);

    // Create the function operation, set private and add to the symbol table.
    IRRewriter::InsertionGuard guard(rewriter);
    auto funcName = (op->getName().getStringRef() + ".impl").str();
    rewriter.setInsertionPointAfter(
        op->template getParentOfType<func::FuncOp>());
    auto funcType =
        rewriter.getFunctionType(op->getOperandTypes(), op->getResultTypes());
    mlir::func::FuncOp func =
        mlir::func::FuncOp::create(rewriter, loc, funcName, funcType);
    func.setPrivate();
    symbolTable.insert(func);
    composite.setDecomposition(func.getSymName());

    // Move the op into the function body.
    Block* body = func.addEntryBlock();
    rewriter.setInsertionPointToEnd(body);
    rewriter.moveOpBefore(op, body, body->begin());
    op->setOperands(body->getArguments());
    mlir::func::ReturnOp::create(rewriter, loc, op->getResults());

    return success();
  }

 private:
  SymbolTable& symbolTable;
  const int32_t version;
};

struct ChloPreserveHighLevelOpsPass
    : public impl::ChloPreserveHighLevelOpsPassBase<
          ChloPreserveHighLevelOpsPass> {
  using ChloPreserveHighLevelOpsPassBase::ChloPreserveHighLevelOpsPassBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SymbolTable symbolTable(moduleOp);
    RewritePatternSet patterns(&getContext());
    patterns.add<ChloOpToCompositePattern<chlo::AcosOp>,
                 ChloOpToCompositePattern<chlo::AcoshOp>,
                 ChloOpToCompositePattern<chlo::AsinOp>,
                 ChloOpToCompositePattern<chlo::AsinhOp>,
                 ChloOpToCompositePattern<chlo::AtanhOp>,
                 ChloOpToCompositePattern<chlo::CoshOp>,
                 ChloOpToCompositePattern<chlo::SinhOp>,
                 ChloOpToCompositePattern<chlo::ErfOp>,
                 ChloOpToCompositePattern<chlo::RaggedDotOp>,
                 ChloOpToCompositePattern<chlo::TopKOp>,
                 ChloOpToCompositePattern<chlo::ScanOp>>(
        &getContext(), symbolTable, /*version=*/1);

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    SmallVector<Operation*> moduleOps;

    // Walk the module in reverse order to prevent re-visiting moved ops.
    for (Operation& op : llvm::reverse(moduleOp.getBodyRegion().front())) {
      walkAndApplyPatterns(&op, frozenPatterns);
    }
  }
};

}  // namespace
}  // namespace stablehlo_ext
}  // namespace mlir
