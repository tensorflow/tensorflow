/* Copyright 2022 The OpenXLA Authors.

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

// This file replaces some complicated HLOs such as SelectAndScatter with a
// sequence of simpler HLOs.

#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {

#define GEN_PASS_DEF_MHLOEXPANDOPSSIMPLIFIERPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

namespace {

ShapedType getScalarizedType(ShapedType t) {
  return t.cloneWith(llvm::ArrayRef<int64_t>(std::nullopt), t.getElementType());
}

struct SelectAndScatterExpanderPattern
    : public OpRewritePattern<SelectAndScatterOp> {
  using OpRewritePattern<SelectAndScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SelectAndScatterOp sas,
                                PatternRewriter& rewriter) const override {
    // Capture original values with variables
    ImplicitLocOpBuilder builder(sas.getLoc(), rewriter);
    TypedValue<RankedTensorType> operand = sas.getOperand();
    llvm::ArrayRef<int64_t> operandShape = operand.getType().getShape();
    TypedValue<RankedTensorType> source = sas.getSource();
    Value initValue = sas.getInitValue();
    Region& select = sas.getSelect();
    Region& scatter = sas.getScatter();
    TensorType sasType = sas.getType();

    // Useful shapes
    const auto iotaShape =
        operand.getType().cloneWith(operandShape, rewriter.getI64Type());
    const auto sourceShape = source.getType().getShape();
    const auto iotaShapeReduced =
        source.getType().cloneWith(sourceShape, rewriter.getI64Type());
    const auto scalarIota = getScalarizedType(iotaShapeReduced);

    // Construct one iota for each dimension. This will reduced in the reduction
    // to determine the indices to be scattered to.
    llvm::SmallVector<Value> iotas;
    iotas.reserve(operandShape.size());
    for (size_t i = 0; i < operandShape.size(); ++i) {
      iotas.push_back(builder.create<mhlo::IotaOp>(iotaShape, i));
    }

    // ReduceWindow arguments
    auto numReduceValues = iotas.size() + 1;
    auto negOne = builder.create<mhlo::ConstantOp>(
        mlir::DenseIntElementsAttr::get(scalarIota, (uint64_t)-1));
    llvm::SmallVector<Value> reduceInitValues(numReduceValues, negOne);
    reduceInitValues.front() = initValue;

    // ReduceWindow arguments
    llvm::SmallVector<Value> ops;
    ops.reserve(numReduceValues);
    ops.push_back(operand);
    ops.insert(ops.end(), iotas.begin(), iotas.end());

    // Construct ReduceWindow and its region.
    auto reduceWindow = builder.create<mhlo::ReduceWindowOp>(
        ops, reduceInitValues, sas.getWindowDimensionsAttr(),
        sas.getWindowStridesAttr(), /*dilations=*/nullptr,
        /*dilations=*/nullptr, sas.getPaddingAttr(),
        [&](OpBuilder& b, Location loc, ValueRange /*values*/) {
          ImplicitLocOpBuilder builder(loc, b);
          Block* block = b.getBlock();
          auto rhsBegin = static_cast<int64_t>(numReduceValues);
          auto lhsBegin = 0;
          auto firstIota = 1;
          Value firstLhsIota = block->getArgument(firstIota);
          Value firstRhsIota = block->getArgument(firstIota + rhsBegin);
          Value lhsFirstInWindow = builder.create<mhlo::CompareOp>(
              firstLhsIota, negOne, mhlo::ComparisonDirection::NE);
          // Current implementations of ReduceWindow do not need the following
          // line in their implementations, but it is actually required in the
          // documented behavior of the implementation which allows the seed
          // value to occur on both lhs and rhs sides when padding occurs.
          Value rhsFirstInWindow = builder.create<mhlo::CompareOp>(
              firstRhsIota, negOne, mhlo::ComparisonDirection::NE);
          auto rhsNotFirstInWindow =
              builder.create<mhlo::NotOp>(rhsFirstInWindow);

          Value operandLhs = block->getArgument(0);
          Value operandRhs = block->getArgument(rhsBegin);
          llvm::SmallVector<Value> selectIns;
          selectIns.push_back(operandLhs);
          selectIns.push_back(operandRhs);
          rewriter.mergeBlocks(&select.front(), block, selectIns);
          Value call = block->back().getOperand(0);
          rewriter.eraseOp(&block->back());

          Value pred = builder.create<mhlo::AndOp>(call, lhsFirstInWindow);
          pred = builder.create<mhlo::OrOp>(pred, rhsNotFirstInWindow);

          llvm::SmallVector<Value> resultTuple;
          for (auto i = lhsBegin; i < rhsBegin; ++i) {
            Value iotaLhs = block->getArgument(i);
            Value iotaRhs = block->getArgument(i + rhsBegin);
            resultTuple.push_back(
                builder.create<mhlo::SelectOp>(pred, iotaLhs, iotaRhs));
          }
          builder.create<mhlo::ReturnOp>(resultTuple);
        });

    // Handle the results of the reduction
    llvm::SmallVector<Value> iotaIndices;
    llvm::SmallVector<int64_t> broadcastedIotaDims;
    broadcastedIotaDims.reserve(iotaShapeReduced.getRank() + 1);
    broadcastedIotaDims.insert(broadcastedIotaDims.end(),
                               iotaShapeReduced.getShape().begin(),
                               iotaShapeReduced.getShape().end());
    broadcastedIotaDims.push_back(1);
    auto broadcastedIotaShape = RankedTensorType::get(
        broadcastedIotaDims, iotaShapeReduced.getElementType());

    for (size_t i = 1; i < numReduceValues; ++i) {
      Value element = reduceWindow.getResult(i);
      iotaIndices.push_back(
          builder.create<mhlo::ReshapeOp>(broadcastedIotaShape, element)
              .getResult());
    }

    // Prepare scatter inputs
    llvm::SmallVector<int64_t> scatterDims(operandShape.size());
    std::iota(scatterDims.begin(), scatterDims.end(), 0);
    Value broadcastedInitValue = builder.create<mhlo::BroadcastOp>(
        initValue, rewriter.getI64TensorAttr(sasType.getShape()));

    llvm::SmallVector<int64_t> concatenatedIotasDims;
    concatenatedIotasDims.reserve(
        mlir::cast<ShapedType>(iotaIndices.front().getType()).getRank());
    concatenatedIotasDims.insert(concatenatedIotasDims.end(),
                                 broadcastedIotaDims.begin(),
                                 broadcastedIotaDims.end());
    concatenatedIotasDims.back() = static_cast<int64_t>(iotaIndices.size());
    Value indices = builder.create<mhlo::ConcatenateOp>(
        RankedTensorType::get(concatenatedIotasDims,
                              iotaShape.getElementType()),
        iotaIndices, iotaShape.getRank());

    // Scatter
    auto dimNums = mhlo::ScatterDimensionNumbersAttr::get(
        sas->getContext(),
        /*updateWindowDims=*/{},
        /*insertedWindowDims=*/scatterDims,
        // TODO: b/342172264 - Implement handling of batching dims.
        /*inputBatchingDims=*/{},
        /*scatterIndicesBatchingDims=*/{},
        /*scatterDimsToOperandDims=*/scatterDims,
        /*indexVectorDim=*/source.getType().getRank());
    auto scatterOp = builder.create<mhlo::ScatterOp>(
        /*shape=*/sasType, /*operand=*/broadcastedInitValue,
        /*scatter_indices=*/indices, /*updates=*/source,
        /*scatter_dim_numbers=*/dimNums,
        /*indices_are_sorted=*/false, /*unique_indices=*/false);

    // Prepare ScatterOp block and then copy SelectAndScatter's body
    llvm::SmallVector<Type> scatterIns;
    llvm::SmallVector<Location> scatterLocs;
    scatterIns.push_back(RankedTensorType::get(
        {}, mlir::cast<ShapedType>(broadcastedInitValue.getType())
                .getElementType()));
    scatterIns.push_back(
        RankedTensorType::get({}, source.getType().getElementType()));
    scatterLocs.push_back(broadcastedInitValue.getLoc());
    scatterLocs.push_back(source.getLoc());

    rewriter.inlineRegionBefore(scatter, scatterOp.getUpdateComputation(),
                                scatterOp.getUpdateComputation().end());
    rewriter.replaceOp(sas, scatterOp.getResults());
    return success();
  }
};

struct MhloExpandOpsSimplifierPass
    : impl::MhloExpandOpsSimplifierPassBase<MhloExpandOpsSimplifierPass> {
  void runOnOperation() override {
    auto* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SelectAndScatterExpanderPattern>(ctx);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createMhloExpandOpsSimplifierPass() {
  return std::make_unique<MhloExpandOpsSimplifierPass>();
}

}  // namespace mhlo
}  // namespace mlir
