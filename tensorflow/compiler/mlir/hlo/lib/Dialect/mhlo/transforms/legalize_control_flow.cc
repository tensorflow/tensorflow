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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
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
    rewriter.inlineRegionBefore(op.cond(), newWhileOp.getBefore(),
                                newWhileOp.getBefore().end());
    auto conditionReturn =
        cast<mhlo::ReturnOp>(newWhileOp.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&newWhileOp.getBefore().front());
    Value i1 = extractTensorValue(rewriter, conditionReturn->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        conditionReturn, i1, newWhileOp.getBeforeArguments());

    // Inline while body, and only replace the mhlo.return with an scf.yield.
    inlineMhloRegionIntoSCFRegion(rewriter, op.body(), newWhileOp.getAfter());

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
    auto scfIf =
        rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
                                   extractTensorValue(rewriter, adaptor.pred()),
                                   /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(rewriter, op.true_branch(),
                                  scfIf.getThenRegion());
    inlineMhloRegionIntoSCFRegion(rewriter, op.false_branch(),
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
    Value idxValue = adaptor.index();
    auto finalIdx = op.branches().size() - 2;

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
    inlineMhloRegionIntoSCFRegion(outerBuilder, op.branches()[currentIdx],
                                  scfIf.getThenRegion());
    int nextIdx = currentIdx + 1;
    // Don't recurse for the final default block.
    if (currentIdx == finalIdx) {
      inlineMhloRegionIntoSCFRegion(outerBuilder, op.branches()[nextIdx],
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
    if (op.branches().size() == 1) {
      Block& block = op.branches().front().front();
      auto results = block.getTerminator()->getOperands();
      // Remove the mhlo.return terminator, then inline the block.
      rewriter.eraseOp(block.getTerminator());
      rewriter.mergeBlockBefore(/*source=*/&block, /*dest=*/op.getOperation(),
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

struct SortOpPattern : public OpConversionPattern<mhlo::SortOp> {
  using OpConversionPattern<SortOp>::OpConversionPattern;

  // Create a loop for each dimension of the input. Finally, create the inner
  // sorting loop and the inner scalar code. Track the indcution variables to be
  // used by the scalar loop and return the result of the outermost loop being
  // created by this (potentially recursive) call.
  static scf::ForOp lowerToLoopsImpl(OpBuilder& builder, mhlo::SortOp op,
                                     OpAdaptor adaptor, unsigned loopDepth,
                                     SmallVectorImpl<Value>& ivs,
                                     ValueRange args) {
    Location loc = op.getLoc();
    if (loopDepth ==
        op->getResultTypes().front().cast<TensorType>().getRank()) {
      return generateScalarImplementation(op, adaptor, builder, ivs, args);
    }

    auto lower = builder.create<arith::ConstantIndexOp>(op.getLoc(), 0);
    auto upper = builder.create<tensor::DimOp>(
        op.getLoc(), adaptor.operands().front(),
        builder.create<arith::ConstantIndexOp>(op.getLoc(), loopDepth));
    auto step = builder.create<arith::ConstantIndexOp>(op.getLoc(), 1);

    auto iterArgs = loopDepth ? args : adaptor.operands();
    return builder.create<scf::ForOp>(
        loc, lower, upper, step, iterArgs,
        [&](OpBuilder& b, Location loc, Value iv, ValueRange argsPrime) {
          ivs.push_back(iv);
          auto result =
              lowerToLoopsImpl(b, op, adaptor, loopDepth + 1, ivs, argsPrime);
          b.create<scf::YieldOp>(loc, result.getResults());
        });
  }

  static scf::ForOp generateScalarImplementation(mhlo::SortOp op,
                                                 OpAdaptor adaptor,
                                                 OpBuilder& b, ValueRange ivs,
                                                 ValueRange args) {
    auto loc = op.getLoc();
    auto sortDim = adaptor.dimension();
    SmallVector<Value> indices, sortArgs;
    indices.append(ivs.begin(), ivs.end());
    // Bubble sort innermost loop.
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value ub;

    auto firstOperandType =
        adaptor.getOperands().front().getType().cast<TensorType>();
    SmallVector<Value> results(args);
    // Create inner most loop with one less iterations, so 1 can be added later.
    if (firstOperandType.isDynamicDim(sortDim)) {
      ub = b.create<tensor::DimOp>(loc, adaptor.getOperands().front(), sortDim);
    } else {
      ub = b.create<arith::ConstantIndexOp>(
          loc, firstOperandType.getDimSize(sortDim));
    }
    ub = b.create<arith::SubIOp>(loc, ub, one);
    auto& srcBlock = op.comparator().front();
    auto scfFor = b.create<scf::ForOp>(
        loc, zero, ub, one, args,
        [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
          // Extract and create tensors with relevant values to merge with the
          // expected inputs to the original compare region of the mhlo.sort op.
          SmallVector<Value> indices(ivs);
          Value ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
          for (const auto& idxAndOutput : llvm::enumerate(args)) {
            indices[sortDim] = iv;
            sortArgs.push_back(b.create<tensor::FromElementsOp>(
                loc, srcBlock.getArgumentTypes()[2 * idxAndOutput.index()],
                b.create<tensor::ExtractOp>(loc, idxAndOutput.value(), indices)
                    .result()));
            indices[sortDim] = ivPlusOne;
            sortArgs.push_back(b.create<tensor::FromElementsOp>(
                loc, srcBlock.getArgumentTypes()[2 * idxAndOutput.index() + 1],
                b.create<tensor::ExtractOp>(loc, idxAndOutput.value(), indices)
                    .result()));
          }
        });

    // Clone the region twice. to compare A,B and B,A
    Region& region = scfFor.getRegion();
    BlockAndValueMapping bvm, bvm2;
    {
      OpBuilder::InsertionGuard guard(b);
      auto& block = region.front();
      b.setInsertionPointToEnd(&block);
      for (int i = 0; i < srcBlock.getNumArguments(); i += 2) {
        bvm.map(srcBlock.getArgument(i), sortArgs[i]);
        bvm.map(srcBlock.getArgument(i + 1), sortArgs[i + 1]);

        bvm2.map(srcBlock.getArgument(i), sortArgs[i + 1]);
        bvm2.map(srcBlock.getArgument(i + 1), sortArgs[i]);
      }
      for (auto& blockOp : srcBlock.without_terminator()) {
        b.clone(blockOp, bvm2);
      }
      for (auto& blockOp : srcBlock.without_terminator()) {
        b.clone(blockOp, bvm);
      }
    }

    // Determine if swapping should occur which happens only if NOT(CMP(A,B)) &&
    // CMP(B,A).
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(&region.front());
    Value cond = b.create<tensor::ExtractOp>(
        loc, bvm.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)));
    Value cond2 = b.create<tensor::ExtractOp>(
        loc, bvm2.lookupOrDefault(srcBlock.getTerminator()->getOperand(0)));
    Value negCond = b.create<arith::XOrIOp>(
        loc, cond, b.create<arith::ConstantIntOp>(loc, 1, cond.getType()));
    Value combined = b.create<arith::AndIOp>(loc, negCond, cond2);

    auto swapResult = b.create<scf::IfOp>(
        loc, op->getResultTypes(), combined,
        [&](OpBuilder& b, Location loc) {
          SmallVector<Value> indices(ivs.begin(), ivs.end());
          Value ivPlusOne =
              b.create<arith::AddIOp>(loc, scfFor.getInductionVar(), one);
          SmallVector<Value> swappedResults;
          for (const auto& idxAndOutput :
               llvm::enumerate(scfFor.getRegionIterArgs())) {
            Value v1 = sortArgs[idxAndOutput.index() * 2];
            Value v2 = sortArgs[idxAndOutput.index() * 2 + 1];
            indices[sortDim] = scfFor.getInductionVar();
            Value afterFirstInsert = b.create<tensor::InsertOp>(
                loc, b.create<tensor::ExtractOp>(loc, v2), idxAndOutput.value(),
                indices);
            indices[sortDim] = ivPlusOne;
            swappedResults.push_back(b.create<tensor::InsertOp>(
                loc, b.create<tensor::ExtractOp>(loc, v1), afterFirstInsert,
                indices));
          }
          b.create<scf::YieldOp>(loc, swappedResults);
        },
        [&](OpBuilder& b, Location loc) {
          b.create<scf::YieldOp>(loc, scfFor.getRegionIterArgs());
        });
    b.create<scf::YieldOp>(loc, swapResult.getResults());
    return scfFor;
  }

  LogicalResult matchAndRewrite(
      mhlo::SortOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> ivs;
    auto scfFor = lowerToLoopsImpl(rewriter, op, adaptor, 0, ivs, {});
    rewriter.replaceOp(op, scfFor.getResults());
    return success();
  }
};

struct LegalizeControlFlowPass
    : public LegalizeControlFlowPassBase<LegalizeControlFlowPass> {
  // Perform the lowering to MLIR control flow.
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = f.getContext();

    RewritePatternSet patterns(&getContext());
    patterns.add<WhileOpPattern, IfOpPattern, CaseOpPattern, SortOpPattern>(
        &getContext());

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target
        .addIllegalOp<mhlo::IfOp, mhlo::WhileOp, mhlo::CaseOp, mhlo::SortOp>();

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
