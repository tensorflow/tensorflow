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
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
// mhlo::ReturnOp and lower to SCF ops which end with scf::YieldOp. Clone an
// entire block with the only change being return -> yield.
void cloneMhloToSCFBlock(Block& block, OpBuilder& b, Location loc) {
  BlockAndValueMapping bvm;
  for (auto& i : block) {
    if (isa<mhlo::ReturnOp>(i)) {
      SmallVector<Value> operands;
      for (auto operand : i.getOperands()) {
        operands.push_back(bvm.lookup(operand));
      }
      b.create<scf::YieldOp>(loc, operands);
    } else {
      b.clone(i, bvm);
    }
  }
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

    // Clone while condition. The block is the same, except the boolean result
    // needs to be extracted and used with an scf.condition.
    rewriter.cloneRegionBefore(op.cond(), new_while_op.getBefore(),
                               new_while_op.getBefore().end());
    auto condition_return =
        cast<mhlo::ReturnOp>(new_while_op.getBefore().front().getTerminator());
    rewriter.setInsertionPointToEnd(&new_while_op.getBefore().front());
    Value i1 = extractTensorValue(rewriter, condition_return->getOperand(0));
    rewriter.replaceOpWithNewOp<scf::ConditionOp>(
        condition_return, i1, new_while_op.getBeforeArguments());

    // Clone while body, and only replace the mhlo.return with an scf.yield.
    rewriter.cloneRegionBefore(op.body(), new_while_op.getAfter(),
                               new_while_op.getAfter().end());
    rewriter.setInsertionPointToEnd(&new_while_op.getAfter().front());
    auto old_return =
        cast<mhlo::ReturnOp>(new_while_op.getAfter().front().getTerminator());
    rewriter.replaceOpWithNewOp<scf::YieldOp>(old_return,
                                              old_return.getOperands());

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
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, op.getResultTypes(), extractTensorValue(rewriter, adaptor.pred()),
        [&](OpBuilder& b, Location l) {
          cloneMhloToSCFBlock(op.true_branch().front(), b, l);
        },
        [&](OpBuilder& b, Location l) {
          cloneMhloToSCFBlock(op.false_branch().front(), b, l);
        });
    return success();
  }
};

// Create a memref descriptor given a pointer and memref type information.
struct CaseOpPattern : public OpConversionPattern<mhlo::CaseOp> {
  using OpConversionPattern<CaseOp>::OpConversionPattern;

  // Recursively create if/else ops to handle each possible value in a case op.
  scf::IfOp createNestedCases(int current_idx, CaseOp op, OpAdaptor adaptor,
                              OpBuilder outer_builder) const {
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
    auto eq_comparison = outer_builder.getStringAttr("EQ");

    return outer_builder.create<scf::IfOp>(
        loc, op.getResultTypes(),
        extractTensorValue(outer_builder,
                           outer_builder.create<mhlo::CompareOp>(
                               loc, idx_value, current_idx_val, eq_comparison)),
        [&](OpBuilder& b, Location l) {
          cloneMhloToSCFBlock(op.branches()[current_idx].front(), b, l);
        },
        [&](OpBuilder& b, Location l) {
          int next_idx = current_idx + 1;
          // Don't recurse for the final default block.
          if (current_idx == final_idx) {
            cloneMhloToSCFBlock(op.branches()[next_idx].back(), b, l);
          } else {
            auto inner_if = createNestedCases(next_idx, op, adaptor, b);
            b.create<scf::YieldOp>(l, inner_if.getResults());
          }
        });
  }

  LogicalResult matchAndRewrite(
      mhlo::CaseOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    // Inline the op if there is only a default block.
    if (op.branches().size() == 1) {
      BlockAndValueMapping bvm;
      for (auto& i : op.branches().front().front()) {
        if (isa<mhlo::ReturnOp>(i)) {
          SmallVector<Value> operands;
          for (auto operand : i.getOperands()) {
            operands.push_back(bvm.lookup(operand));
          }
          rewriter.replaceOp(op, operands);
        } else {
          rewriter.clone(i, bvm);
        }
      }
      return success();
    }

    // Begin recursion with case 0.
    rewriter.replaceOp(
        op, createNestedCases(0, op, adaptor, rewriter).getResults());
    return success();
  }
};

struct LegalizeControlFlowPass
    : public LegalizeControlFlowPassBase<LegalizeControlFlowPass> {
  // Perform the lowering to MLIR control flow.
  void runOnOperation() override {
    FuncOp f = getOperation();
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

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>>
mlir::mhlo::createLegalizeControlFlowPass() {
  return std::make_unique<LegalizeControlFlowPass>();
}
