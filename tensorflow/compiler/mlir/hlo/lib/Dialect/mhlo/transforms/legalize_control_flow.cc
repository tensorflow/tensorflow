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

    auto new_while_op = rewriter.create<scf::WhileOp>(loc, op.getResultTypes(),
                                                      adaptor.getOperands());

    // Inline while condition. The block is the same, except the boolean result
    // needs to be extracted and used with an scf.condition.
    rewriter.inlineRegionBefore(op.cond(), new_while_op.getBefore(),
                                new_while_op.getBefore().end());
    auto condition_return =
        cast<mhlo::ReturnOp>(new_while_op.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&new_while_op.getBefore().front());
    Value i1 = extractTensorValue(rewriter, condition_return->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        condition_return, i1, new_while_op.getBeforeArguments());

    // Inline while body, and only replace the mhlo.return with an scf.yield.
    inlineMhloRegionIntoSCFRegion(rewriter, op.body(), new_while_op.getAfter());

    rewriter.replaceOp(op, new_while_op.getResults());
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct IfOpPattern : public OpConversionPattern<mhlo::IfOp> {
  using OpConversionPattern<IfOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto scf_if =
        rewriter.create<scf::IfOp>(op.getLoc(), op.getResultTypes(),
                                   extractTensorValue(rewriter, adaptor.pred()),
                                   /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(rewriter, op.true_branch(),
                                  scf_if.getThenRegion());
    inlineMhloRegionIntoSCFRegion(rewriter, op.false_branch(),
                                  scf_if.getElseRegion());
    rewriter.replaceOp(op, scf_if.getResults());
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct CaseOpPattern : public OpConversionPattern<mhlo::CaseOp> {
  using OpConversionPattern<CaseOp>::OpConversionPattern;

  // Recursively create if/else ops to handle each possible value in a case op.
  scf::IfOp createNestedCases(int current_idx, CaseOp op, OpAdaptor adaptor,
                              PatternRewriter& outer_builder) const {
    Location loc = op.getLoc();
    Value idx_value = adaptor.index();
    auto final_idx = op.branches().size() - 2;

    // Determine if the current index matches the case index.
    auto scalar_type = idx_value.getType();
    auto const_attr = DenseElementsAttr::get(
        scalar_type,
        {outer_builder.getI32IntegerAttr(current_idx).cast<mlir::Attribute>()});
    Value current_idx_val = outer_builder.create<mhlo::ConstOp>(
        loc, idx_value.getType(), const_attr);

    auto scf_if = outer_builder.create<scf::IfOp>(
        loc, op.getResultTypes(),
        extractTensorValue(outer_builder, outer_builder.create<mhlo::CompareOp>(
                                              loc, idx_value, current_idx_val,
                                              ComparisonDirection::EQ)),
        /*withElseRegion=*/true);
    inlineMhloRegionIntoSCFRegion(outer_builder, op.branches()[current_idx],
                                  scf_if.getThenRegion());
    int next_idx = current_idx + 1;
    // Don't recurse for the final default block.
    if (current_idx == final_idx) {
      inlineMhloRegionIntoSCFRegion(outer_builder, op.branches()[next_idx],
                                    scf_if.getElseRegion());
    } else {
      PatternRewriter::InsertionGuard guard(outer_builder);
      outer_builder.setInsertionPointToEnd(&scf_if.getElseRegion().back());
      auto inner_if = createNestedCases(next_idx, op, adaptor, outer_builder);
      outer_builder.create<scf::YieldOp>(op.getLoc(), inner_if.getResults());
    }
    return scf_if;
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

    auto iter_args = loopDepth ? args : adaptor.operands();
    return builder.create<scf::ForOp>(
        loc, lower, upper, step, iter_args,
        [&](OpBuilder& b, Location loc, Value iv, ValueRange args_prime) {
          ivs.push_back(iv);
          auto result =
              lowerToLoopsImpl(b, op, adaptor, loopDepth + 1, ivs, args_prime);
          b.create<scf::YieldOp>(loc, result.getResults());
        });
  }

  static scf::ForOp generateScalarImplementation(mhlo::SortOp op,
                                                 OpAdaptor adaptor,
                                                 OpBuilder& b, ValueRange ivs,
                                                 ValueRange args) {
    auto loc = op.getLoc();
    auto sort_dim = adaptor.dimension();
    SmallVector<Value> indices, sort_args;
    indices.append(ivs.begin(), ivs.end());
    // Bubble sort innermost loop.
    Value zero = b.create<arith::ConstantIndexOp>(loc, 0);
    Value one = b.create<arith::ConstantIndexOp>(loc, 1);
    Value ub;

    auto first_operand_type =
        adaptor.getOperands().front().getType().cast<TensorType>();
    SmallVector<Value> results(args);
    // Create inner most loop with one less iterations, so 1 can be added later.
    if (first_operand_type.isDynamicDim(sort_dim)) {
      ub =
          b.create<tensor::DimOp>(loc, adaptor.getOperands().front(), sort_dim);
    } else {
      ub = b.create<arith::ConstantIndexOp>(
          loc, first_operand_type.getDimSize(sort_dim));
    }
    ub = b.create<arith::SubIOp>(loc, ub, one);
    auto& src_block = op.comparator().front();
    auto scf_for = b.create<scf::ForOp>(
        loc, zero, ub, one, args,
        [&](OpBuilder& b, Location loc, Value iv, ValueRange args) {
          // Extract and create tensors with relevant values to merge with the
          // expected inputs to the original compare region of the mhlo.sort op.
          SmallVector<Value> indices(ivs);
          Value ivPlusOne = b.create<arith::AddIOp>(loc, iv, one);
          for (const auto& idx_and_output : llvm::enumerate(args)) {
            indices[sort_dim] = iv;
            sort_args.push_back(b.create<tensor::FromElementsOp>(
                loc, src_block.getArgumentTypes()[2 * idx_and_output.index()],
                b.create<tensor::ExtractOp>(loc, idx_and_output.value(),
                                            indices)
                    .result()));
            indices[sort_dim] = ivPlusOne;
            sort_args.push_back(b.create<tensor::FromElementsOp>(
                loc,
                src_block.getArgumentTypes()[2 * idx_and_output.index() + 1],
                b.create<tensor::ExtractOp>(loc, idx_and_output.value(),
                                            indices)
                    .result()));
          }
        });

    // Clone the region twice. to compare A,B and B,A
    Region& region = scf_for.getRegion();
    BlockAndValueMapping bvm, bvm2;
    {
      OpBuilder::InsertionGuard guard(b);
      auto& block = region.front();
      b.setInsertionPointToEnd(&block);
      for (int i = 0; i < src_block.getNumArguments(); i += 2) {
        bvm.map(src_block.getArgument(i), sort_args[i]);
        bvm.map(src_block.getArgument(i + 1), sort_args[i + 1]);

        bvm2.map(src_block.getArgument(i), sort_args[i + 1]);
        bvm2.map(src_block.getArgument(i + 1), sort_args[i]);
      }
      for (auto& block_op : src_block.without_terminator()) {
        b.clone(block_op, bvm2);
      }
      for (auto& block_op : src_block.without_terminator()) {
        b.clone(block_op, bvm);
      }
    }

    // Determine if swapping should occur which happens only if NOT(CMP(A,B)) &&
    // CMP(B,A).
    OpBuilder::InsertionGuard g(b);
    b.setInsertionPointToEnd(&region.front());
    Value cond = b.create<tensor::ExtractOp>(
        loc, bvm.lookupOrDefault(src_block.getTerminator()->getOperand(0)));
    Value cond2 = b.create<tensor::ExtractOp>(
        loc, bvm2.lookupOrDefault(src_block.getTerminator()->getOperand(0)));
    Value neg_cond = b.create<arith::XOrIOp>(
        loc, cond, b.create<arith::ConstantIntOp>(loc, 1, cond.getType()));
    Value combined = b.create<arith::AndIOp>(loc, neg_cond, cond2);

    auto swap_result = b.create<scf::IfOp>(
        loc, op->getResultTypes(), combined,
        [&](OpBuilder& b, Location loc) {
          SmallVector<Value> indices(ivs.begin(), ivs.end());
          Value ivPlusOne =
              b.create<arith::AddIOp>(loc, scf_for.getInductionVar(), one);
          SmallVector<Value> swapped_results;
          for (const auto& idx_and_output :
               llvm::enumerate(scf_for.getRegionIterArgs())) {
            Value v1 = sort_args[idx_and_output.index() * 2];
            Value v2 = sort_args[idx_and_output.index() * 2 + 1];
            indices[sort_dim] = scf_for.getInductionVar();
            Value after_first_insert = b.create<tensor::InsertOp>(
                loc, b.create<tensor::ExtractOp>(loc, v2),
                idx_and_output.value(), indices);
            indices[sort_dim] = ivPlusOne;
            swapped_results.push_back(b.create<tensor::InsertOp>(
                loc, b.create<tensor::ExtractOp>(loc, v1), after_first_insert,
                indices));
          }
          b.create<scf::YieldOp>(loc, swapped_results);
        },
        [&](OpBuilder& b, Location loc) {
          b.create<scf::YieldOp>(loc, scf_for.getRegionIterArgs());
        });
    b.create<scf::YieldOp>(loc, swap_result.getResults());
    return scf_for;
  }

  LogicalResult matchAndRewrite(
      mhlo::SortOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    SmallVector<Value> ivs;
    auto scf_for = lowerToLoopsImpl(rewriter, op, adaptor, 0, ivs, {});
    rewriter.replaceOp(op, scf_for.getResults());
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
