/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for lowering MHLO dialect to SCF dialect.
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_LEGALIZECONTROLFLOWPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

// All transformations in this file take mhlo blocks which end with
// mhlo::ReturnOp and lower to SCF ops which end with scf::YieldOp. Inline an
// entire block with the only change being return -> yield.
void inlineMhloRegionIntoSCFRegion(PatternRewriter& rewriter, Region& mhlo,
                                   Region& scf) {
  // Remove an existing block, then move the region over.
  if (!scf.empty()) rewriter.eraseBlock(&scf.back());
  rewriter.inlineRegionBefore(mhlo, scf, scf.end());
  // Fix up the terminator.
  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(&scf.back());
  auto* terminator = scf.back().getTerminator();
  rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator,
                                            terminator->getOperands());
}

// mhlo ops need inputs to be tensors, but scalar values can be a scalar tensor
// or a 1 element tensor. To handle this, collapse shape before extracting the
// scalar value when necessary.
Value extractTensorValue(OpBuilder& b, Value tensor) {
  auto loc = tensor.getLoc();
  if (tensor.getType().cast<TensorType>().hasRank() &&
      tensor.getType().cast<TensorType>().getRank() != 0) {
    tensor = b.create<tensor::CollapseShapeOp>(
        loc, tensor, SmallVector<ReassociationIndices>());
  }
  return b.create<tensor::ExtractOp>(loc, tensor, ValueRange());
}

// Create a memref descriptor given a pointer and memref type information.
struct WhileOpPattern : public OpConversionPattern<mhlo::WhileOp> {
  using OpConversionPattern<WhileOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::WhileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();

    auto newWhileOp = rewriter.create<scf::WhileOp>(loc, op.getResultTypes(),
                                                    adaptor.getOperands());

    // Inline while condition. The block is the same, except the boolean result
    // needs to be extracted and used with an scf.condition.
    rewriter.inlineRegionBefore(op.getCond(), newWhileOp.getBefore(),
                                newWhileOp.getBefore().end());
    auto conditionReturn =
        cast<mhlo::ReturnOp>(newWhileOp.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&newWhileOp.getBefore().front());
    Value i1 = extractTensorValue(rewriter, conditionReturn->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        conditionReturn, i1, newWhileOp.getBeforeArguments());

    // Inline while body, and only replace the mhlo.return with an scf.yield.
    inlineMhloRegionIntoSCFRegion(rewriter, op.getBody(),
                                  newWhileOp.getAfter());

    rewriter.replaceOp(op, newWhileOp.getResults());
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct IfOpPattern : public OpConversionPattern<mhlo::IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto scfIf = rewriter.create<scf::IfOp>(
        op.getLoc(), op.getResultTypes(),
        extractTensorValue(rewriter, adaptor.getPred()),
        /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(rewriter, op.getTrueBranch(),
                                  scfIf.getThenRegion());
    inlineMhloRegionIntoSCFRegion(rewriter, op.getFalseBranch(),
                                  scfIf.getElseRegion());
    rewriter.replaceOp(op, scfIf.getResults());
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct CaseOpPattern : public OpConversionPattern<mhlo::CaseOp> {
  using OpConversionPattern<CaseOp>::OpConversionPattern;

  // Recursively create if/else ops to handle each possible value in a case op.
  scf::IfOp createNestedCases(int currentIdx, CaseOp op, OpAdaptor adaptor,
                              PatternRewriter& outerBuilder) const {
    Location loc = op.getLoc();
    Value idxValue = adaptor.getIndex();
    auto finalIdx = op.getBranches().size() - 2;

    // Determine if the current index matches the case index.
    auto scalarType = idxValue.getType();
    auto constAttr = DenseElementsAttr::get(
        scalarType,
        {outerBuilder.getI32IntegerAttr(currentIdx).cast<mlir::Attribute>()});
    Value currentIdxVal = outerBuilder.create<mhlo::ConstantOp>(
        loc, idxValue.getType(), constAttr);

    auto scfIf = outerBuilder.create<scf::IfOp>(
        loc, op.getResultTypes(),
        extractTensorValue(outerBuilder, outerBuilder.create<mhlo::CompareOp>(
                                             loc, idxValue, currentIdxVal,
                                             ComparisonDirection::EQ)),
        /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(outerBuilder, op.getBranches()[currentIdx],
                                  scfIf.getThenRegion());
    int nextIdx = currentIdx + 1;
    // Don't recurse for the final default block.
    if (currentIdx == static_cast<int64_t>(finalIdx)) {
      inlineMhloRegionIntoSCFRegion(outerBuilder, op.getBranches()[nextIdx],
                                    scfIf.getElseRegion());
    } else {
      PatternRewriter::InsertionGuard guard(outerBuilder);
      outerBuilder.setInsertionPointToEnd(&scfIf.getElseRegion().back());
      auto innerIf = createNestedCases(nextIdx, op, adaptor, outerBuilder);
      outerBuilder.create<scf::YieldOp>(op.getLoc(), innerIf.getResults());
    }
    return scfIf;
  }

  LogicalResult matchAndRewrite(
      mhlo::CaseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Inline the op if there is only a default block.
    if (op.getBranches().size() == 1) {
      Block& block = op.getBranches().front().front();
      auto results = block.getTerminator()->getOperands();
      // Remove the mhlo.return terminator, then inline the block.
      rewriter.eraseOp(block.getTerminator());
      rewriter.inlineBlockBefore(/*source=*/&block, /*dest=*/op.getOperation(),
                                 /*argValues=*/{});
      rewriter.replaceOp(op, results);
      return success();
    }

    // Begin recursion with case 0.
    rewriter.replaceOp(
        op, createNestedCases(0, op, adaptor, rewriter).getResults());
    return success();
  }
};

struct LegalizeControlFlowPass
    : public impl::LegalizeControlFlowPassBase<LegalizeControlFlowPass> {
  // Perform the lowering to MLIR control flow.
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = f.getContext();

    RewritePatternSet patterns(&getContext());
    patterns.add<WhileOpPattern, IfOpPattern, CaseOpPattern>(&getContext());

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<mhlo::IfOp, mhlo::WhileOp, mhlo::CaseOp>();

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::mhlo::createLegalizeControlFlowPass() {
  return std::make_unique<LegalizeControlFlowPass>();
}
