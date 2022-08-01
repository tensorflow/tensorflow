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
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/PassDetail.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"  // TF:llvm-project
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace mhlo {
namespace {

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
                    .getResult()));
            indices[sortDim] = ivPlusOne;
            sortArgs.push_back(b.create<tensor::FromElementsOp>(
                loc, srcBlock.getArgumentTypes()[2 * idxAndOutput.index() + 1],
                b.create<tensor::ExtractOp>(loc, idxAndOutput.value(), indices)
                    .getResult()));
          }
        });

    // Clone the region twice. to compare A,B and B,A
    Region& region = scfFor.getRegion();
    BlockAndValueMapping bvm, bvm2;
    {
      OpBuilder::InsertionGuard guard(b);
      auto& block = region.front();
      b.setInsertionPointToEnd(&block);
      for (int64_t i = 0; i < srcBlock.getNumArguments(); i += 2) {
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

struct LegalizeSortPass : public HloLegalizeSortPassBase<LegalizeSortPass> {
  // Perform the lowering to MLIR control flow.
  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext* ctx = f.getContext();

    RewritePatternSet patterns(&getContext());
    patterns.add<SortOpPattern>(&getContext());

    mlir::ConversionTarget target(*ctx);
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<mhlo::SortOp>();

    if (failed(applyPartialConversion(f, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mhlo
}  // namespace mlir

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
mlir::mhlo::createLegalizeSortPass() {
  return std::make_unique<LegalizeSortPass>();
}
