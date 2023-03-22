/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file implements logic for simplifying HLO gather.

#include <iterator>
#include <memory>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mhlo/IR/hlo_ops.h"
#include "mhlo/transforms/passes.h"
#include "mhlo/utils/mhlo_scatter_gather_utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_HLOCANONICALIZEGATHERPASS
#include "mhlo/transforms/mhlo_passes.h.inc"

// Given an input tensor, collapse dimensions 1+collapsedSliceDims[...].
Value collapseSliceDims(ImplicitLocOpBuilder& b, TypedValue<TensorType> input,
                        ArrayRef<int64_t> collapsedSliceDims) {
  if (collapsedSliceDims.empty()) return input;

  SmallVector<int64_t> newShape{input.getType().getShape()};
  // collapsedSliceDims is small in practice.
  for (int64_t dim : llvm::reverse(collapsedSliceDims)) {
    // Dimension 0 is the collapsed batch dimension.
    newShape.erase(newShape.begin() + 1 + dim);
  }

  return b.create<tensor::CollapseShapeOp>(
      input, *getReassociationIndicesForCollapse(input.getType().getShape(),
                                                 newShape));
}

// Expands the first dimension of `input` into the shape of `startIndices`,
// removing the index vector dimension.
Value expandBatchDimension(ImplicitLocOpBuilder& b,
                           TypedValue<TensorType> input,
                           GatherOp originalGatherOp) {
  llvm::SmallVector<int64_t> newShape{
      originalGatherOp.getStartIndices().getType().getShape()};
  // Erase the index vector dimension if it wasn't implicit.
  int64_t indexDim = originalGatherOp.getDimensionNumbers().getIndexVectorDim();
  if (indexDim < static_cast<int64_t>(newShape.size()))
    newShape.erase(newShape.begin() + indexDim);

  // `input` has one batch dimension, if we still have one now, there is nothing
  // to do.
  if (newShape.size() == 1) return input;

  // Copy the slice dimensions.
  llvm::copy(input.getType().getShape().drop_front(1),
             std::back_inserter(newShape));

  auto newType =
      RankedTensorType::get(newShape, input.getType().getElementType());
  auto reassociation =
      *getReassociationIndicesForReshape(input.getType(), newType);
  if (static_cast<int64_t>(newShape.size()) > input.getType().getRank()) {
    return b.create<tensor::ExpandShapeOp>(newType, input, reassociation);
  }
  return b.create<tensor::CollapseShapeOp>(newType, input, reassociation);
}

Value moveOffsetDimensions(ImplicitLocOpBuilder& b,
                           TypedValue<TensorType> input,
                           GatherOp originalGatherOp) {
  const auto& dims = originalGatherOp.getDimensionNumbers();
  int64_t outputRank = input.getType().getRank();
  int64_t offsetDimIndex = outputRank - dims.getOffsetDims().size();
  int64_t batchDimIndex = 0;
  llvm::SmallVector<int64_t> outputPermutation;
  outputPermutation.reserve(outputRank);
  for (int64_t i = 0; i < outputRank; ++i) {
    if (llvm::is_contained(dims.getOffsetDims(), i)) {
      outputPermutation.push_back(offsetDimIndex++);
    } else {
      outputPermutation.push_back(batchDimIndex++);
    }
  }
  return b.create<TransposeOp>(input, b.getI64TensorAttr(outputPermutation));
}

template <typename R>
SmallVector<int64_t> permute(R&& values, ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> permutedValues;
  permutedValues.reserve(values.size());
  for (int64_t dim : permutation) permutedValues.push_back(values[dim]);
  return permutedValues;
}

struct CanonicalizeGatherPattern : public OpRewritePattern<GatherOp> {
  using OpRewritePattern<GatherOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GatherOp gatherOp,
                                PatternRewriter& rewriter) const override {
    if (isCanonicalGather(gatherOp)) return failure();
    ImplicitLocOpBuilder b(gatherOp.getLoc(), rewriter);

    // If any slice size is 0, we can just return a constant zero.
    if (llvm::is_contained(gatherOp.getSliceSizes(), 0)) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          gatherOp.getOperation(), b.getZeroAttr(gatherOp.getType()));
      return success();
    }

    const auto& dims = gatherOp.getDimensionNumbers();
    int64_t operandRank =
        dims.getCollapsedSliceDims().size() + dims.getOffsetDims().size();

    // Make the operand conform to start_index_map.
    auto [operandPermutation, operandPermutationInverse] =
        makeOperandStartIndexPermutations(dims.getStartIndexMap(), operandRank);

    Value operand = b.create<TransposeOp>(
        gatherOp.getOperand(), b.getI64TensorAttr(operandPermutation));
    auto startIndices = canonicalizeStartIndices(rewriter, b.getLoc(),
                                                 gatherOp.getStartIndices(),
                                                 dims.getIndexVectorDim());

    // Permute the slice sizes according to start_index_map and compute the new
    // output shape for the Gather op.
    auto offsetDims = llvm::to_vector(llvm::seq(int64_t{1}, 1 + operandRank));
    auto startIndexMap = llvm::to_vector(llvm::seq(
        int64_t{0}, static_cast<int64_t>(dims.getStartIndexMap().size())));

    auto newDims = GatherDimensionNumbersAttr::get(
        rewriter.getContext(), offsetDims,
        /*collapsedSliceDims=*/{}, startIndexMap,
        /*indexVectorDim=*/1);
    TypedValue<TensorType> result =
        b.create<GatherOp>(operand, startIndices, newDims,
                           b.getI64TensorAttr(permute(
                               gatherOp.getSliceSizes().getValues<int64_t>(),
                               operandPermutation)),
                           gatherOp.getIndicesAreSorted())
            .getResult();

    // Undo the startIndexMap transpose.
    for (int64_t& dim : operandPermutationInverse) ++dim;
    // Add the batch dimension and keep it at the front.
    operandPermutationInverse.insert(operandPermutationInverse.begin(), 0);
    result = b.create<TransposeOp>(
        result, b.getI64TensorAttr(operandPermutationInverse));

    // Collapse the requested dimensions.
    result = cast<TypedValue<TensorType>>(
        collapseSliceDims(b, result, dims.getCollapsedSliceDims()));

    // Expand the start index dimensions.
    result =
        cast<TypedValue<TensorType>>(expandBatchDimension(b, result, gatherOp));

    // Move the offset dims to the final locations.
    result =
        cast<TypedValue<TensorType>>(moveOffsetDimensions(b, result, gatherOp));

    rewriter.replaceOp(gatherOp.getOperation(), {result});
    return success();
  }
};

struct HloCanonicalizeGatherPass
    : impl::HloCanonicalizeGatherPassBase<HloCanonicalizeGatherPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CanonicalizeGatherPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createHloCanonicalizeGatherPass() {
  return std::make_unique<HloCanonicalizeGatherPass>();
}

}  // namespace mhlo
}  // namespace mlir
