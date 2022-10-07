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

// This file implements logic for simplifying HLO scatter.

#include <memory>
#include <numeric>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_scatter_utils.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace mhlo {
namespace {

#define GEN_PASS_DEF_HLOCANONICALIZESCATTERPASS
#include "mlir-hlo/Dialect/mhlo/transforms/mhlo_passes.h.inc"

DenseIntElementsAttr getI64ElementsAttr(ArrayRef<int64_t> values,
                                        Builder* builder) {
  auto ty = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                  builder->getIntegerType(64));
  return DenseIntElementsAttr::get(ty, values);
}

SmallVector<int64_t> getInversePermutation(
    llvm::ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inversePermutation(permutation.size());
  for (size_t i = 0, e = permutation.size(); i < e; ++i)
    inversePermutation[permutation[i]] = i;
  return inversePermutation;
}

bool isIdentityPermutation(ArrayRef<int64_t> permutation) {
  for (int64_t i = 0, e = permutation.size(); i < e; ++i)
    if (permutation[i] != i) return false;
  return true;
}

SmallVector<Value> transposeTensors(OpBuilder& b, Location loc,
                                    ValueRange tensors,
                                    ArrayRef<int64_t> permutation) {
  if (isIdentityPermutation(permutation)) return tensors;

  auto permutationAttr = getI64ElementsAttr(permutation, &b);
  SmallVector<Value> transposedTensors;
  for (Value tensor : tensors) {
    transposedTensors.push_back(
        b.create<TransposeOp>(loc, tensor, permutationAttr));
  }
  return transposedTensors;
}

// Expand the shape of `tensor`, inserting degenerate dimensions.
//
// For example tensor<10x4xf32> and dimsToInsert = {0, 1}
// will result in tensor<1x10x1x4xf32>.
Value insertDegenerateDimensions(OpBuilder& b, Location loc, Value tensor,
                                 ArrayRef<int64_t> dimsToInsert) {
  auto tensorType = tensor.getType().cast<RankedTensorType>();
  int64_t tensorRank = tensorType.getRank();
  int64_t numDimsToInsert = dimsToInsert.size();
  int64_t newRank = tensorRank + numDimsToInsert;

  SmallVector<int64_t> newShape;
  SmallVector<ReassociationIndices> reassociations;

  int64_t tensorDimIdx = 0;
  int64_t dimsToInsertIdx = 0;

  ReassociationIndices reassociation;
  for (int i = 0; i < newRank; ++i) {
    reassociation.push_back(i);
    if (dimsToInsertIdx < numDimsToInsert &&
        i == dimsToInsert[dimsToInsertIdx]) {
      newShape.push_back(1);
      ++dimsToInsertIdx;
    } else {
      newShape.push_back(tensorType.getDimSize(tensorDimIdx));
      ++tensorDimIdx;
      // Trailing 1s in the newShape need to be added to the last reassociation.
      if (tensorDimIdx != tensorRank) {
        reassociations.push_back(reassociation);
        reassociation.clear();
      }
    }
  }
  if (!reassociation.empty()) {
    reassociations.push_back(reassociation);
  }
  return b.create<tensor::ExpandShapeOp>(
      loc, RankedTensorType::get(newShape, tensorType.getElementType()), tensor,
      reassociations);
}

// Checks if the indexVectorDim is equal to the rank of `indices`. In that
// case add the trailing 1 dimension. If indexVectorDim is not the innermost
// dimension, insert transpose to make it so.
TypedValue<TensorType> ensureIndexVectorDimPosition(
    OpBuilder& b, Location loc, TypedValue<TensorType> indices,
    int64_t indexVectorDim) {
  auto indicesType = indices.getType();
  auto indicesRank = indicesType.getRank();

  if (indexVectorDim == indicesRank) {
    indices = insertDegenerateDimensions(b, loc, indices, {indicesRank});
  } else if (indexVectorDim != indicesRank - 1) {
    SmallVector<int64_t> permutation;
    for (int64_t i = 0; i < indicesRank; ++i)
      if (i != indexVectorDim) permutation.push_back(i);
    permutation.push_back(indexVectorDim);
    indices = b.create<TransposeOp>(loc, indices,
                                    getI64ElementsAttr(permutation, &b));
  }
  return indices;
}

// Insert transposes and reshapes to bring `indices` to the 2D shape, where
// the dim0 is the product of all dimensions that are not equal to
// `indexVectorDim` and dim1 is the index vector dim.
//
// Examples.
//
// [a, I, b] will be transposed to [a, b, I], then reshaped into [ab, I].
// [a, b] will be reshaped to [a, b, I(1)] and then reshaped into [ab, I(1)].
Value canonicalizeScatterIndices(OpBuilder& b, Location loc,
                                 TypedValue<TensorType> indices,
                                 int64_t indexVectorDim) {
  indices = ensureIndexVectorDimPosition(b, loc, indices, indexVectorDim);

  auto indicesType = indices.getType();
  auto indicesRank = indicesType.getRank();

  if (indicesRank == 2) return indices;

  if (indicesRank == 1) return insertDegenerateDimensions(b, loc, indices, {0});

  // Insert reshape to collapse all outer dimensions of `Indices`.
  SmallVector<ReassociationIndices> reassociation{
      llvm::to_vector<2>(llvm::seq<int64_t>(0, indicesRank - 1)),
      {indicesRank - 1}};
  return b.create<tensor::CollapseShapeOp>(loc, indices, reassociation);
}

// Transposes updates to align with the dims of operands.
SmallVector<Value> transposeUpdatesAccordingToScatterDimsMap(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> scatterDimsToOperandDims) {
  auto updatesType = updates.front().getType().cast<RankedTensorType>();
  int64_t updatesRank = updatesType.getRank();
  int64_t operandRank = updatesRank - 1;

  // For the updates, we need to add the scatter dimension to the permutation.
  SmallVector<int64_t> permutation{0};
  for (int64_t i : scatterDimsToOperandDims) {
    permutation.push_back(i + 1);
  }
  for (int64_t i = 0; i < operandRank; ++i) {
    if (!llvm::is_contained(scatterDimsToOperandDims, i))
      permutation.push_back(i + 1);
  }
  return transposeTensors(b, loc, updates, permutation);
}

// Makes window dimensions of `updates` the innermost ones.
SmallVector<Value> transposeUpdatesToMoveWindowDimensionsInside(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> updateWindowDims) {
  auto updatesType = updates.front().getType().cast<RankedTensorType>();
  int64_t updatesRank = updatesType.getRank();

  // Move update dimensions to the back
  SmallVector<int64_t> permutation;
  for (int i = 0; i < updatesRank; ++i) {
    if (!llvm::is_contained(updateWindowDims, i)) permutation.push_back(i);
  }
  permutation.append(updateWindowDims.begin(), updateWindowDims.end());
  return transposeTensors(b, loc, updates, permutation);
}

SmallVector<Value> reshapeUpdatesToEnsureSingleScatterDimension(
    OpBuilder& b, Location loc, ValueRange updates,
    ArrayRef<int64_t> updateWindowDims) {
  auto updatesType = updates.front().getType().cast<RankedTensorType>();
  int64_t updatesRank = updatesType.getRank();

  // Collapse scatter dimensions to 1D if there are more than 1 or prepend a
  // size-1 dimension if there are no explicit scatter dims.
  size_t numScatterDims = updatesRank - updateWindowDims.size();
  if (numScatterDims > 1) {
    SmallVector<ReassociationIndices> reassociation{
        llvm::to_vector<2>(llvm::seq<int64_t>(0, numScatterDims))};
    for (int i = numScatterDims, e = updatesRank; i < e; ++i)
      reassociation.push_back({i});

    return to_vector(llvm::map_range(updates, [&](Value update) -> Value {
      return b.create<tensor::CollapseShapeOp>(loc, update, reassociation);
    }));
  }
  if (numScatterDims == 0) {
    return to_vector(llvm::map_range(updates, [&](Value update) {
      return insertDegenerateDimensions(b, loc, update, {0});
    }));
  }
  return updates;
}

// Inserts size-1 dimensions to get rid of `insertedWindowDims` attribute.
SmallVector<Value> reshapeUpdatesToMatchOperandShape(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> insertedWindowDims) {
  size_t numScatterDims = insertedWindowDims.size();
  if (numScatterDims == 0) return updates;

  SmallVector<int64_t> shiftedScatterDimsToOperandDims;
  for (int64_t i : insertedWindowDims)
    shiftedScatterDimsToOperandDims.push_back(i + 1);

  return to_vector(map_range(updates, [&](Value update) {
    return insertDegenerateDimensions(b, loc, update,
                                      shiftedScatterDimsToOperandDims);
  }));
}

// Inserts transposes and reshapes to make window/slice dimensions become the
// innermost dimensions of updates. Also insert degenerate size-1 dimensions to
// match the shape of the slice and the shape of the operand.
SmallVector<Value> canonicalizeUpdates(
    OpBuilder& b, Location loc, SmallVector<Value> updates,
    ArrayRef<int64_t> scatterDimsToOperandDims,
    ArrayRef<int64_t> updateWindowDims, ArrayRef<int64_t> insertedWindowDims) {
  updates = transposeUpdatesToMoveWindowDimensionsInside(b, loc, updates,
                                                         updateWindowDims);
  updates = reshapeUpdatesToEnsureSingleScatterDimension(b, loc, updates,
                                                         updateWindowDims);
  updates =
      reshapeUpdatesToMatchOperandShape(b, loc, updates, insertedWindowDims);
  return transposeUpdatesAccordingToScatterDimsMap(b, loc, updates,
                                                   scatterDimsToOperandDims);
}

// Creates a permutation that shuffles dimensions of `operands` to match the
// order in the index vector.
SmallVector<int64_t> makeOperandPermutation(
    ArrayRef<int64_t> scatterDimsToOperandDims, int operandRank) {
  SmallVector<int64_t> permutation{scatterDimsToOperandDims};
  for (int i = 0; i < operandRank; ++i) {
    if (!llvm::is_contained(scatterDimsToOperandDims, i))
      permutation.push_back(i);
  }
  return permutation;
}

// This pattern rewrites scatter into a transposes, reshapes and a simpler
// scatter.
//
// It transposes and reshapes updates, scatterIndices and operands to get to
// the following characteristics:
//
// - scatter_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - inserted_window_dims is []
// - update_window_dims is [1, 2, ...]
// - scatter_dims_to_operand_dims is [0, 1, ...]
struct CanonicalizeScatterPattern : public OpRewritePattern<ScatterOp> {
  using OpRewritePattern<ScatterOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ScatterOp scatterOp,
                                PatternRewriter& rewriter) const override {
    if (isCanonicalScatter(scatterOp)) return failure();

    Location loc = scatterOp.getLoc();
    ScatterDimensionNumbersAttr dimsAttrs =
        scatterOp.getScatterDimensionNumbers();

    auto operandType =
        scatterOp.operands().front().getType().cast<RankedTensorType>();
    int64_t operandRank = operandType.getRank();
    SmallVector<int64_t> operandPermutation = makeOperandPermutation(
        dimsAttrs.getScatterDimsToOperandDims(), operandRank);

    Value canonicalIndices =
        canonicalizeScatterIndices(rewriter, loc, scatterOp.getScatterIndices(),
                                   dimsAttrs.getIndexVectorDim());

    SmallVector<Value> canonicalOperands = transposeTensors(
        rewriter, loc, scatterOp.operands(), operandPermutation);

    SmallVector<Value> canonicalUpdates = canonicalizeUpdates(
        rewriter, loc, scatterOp.getUpdates(),
        dimsAttrs.getScatterDimsToOperandDims(),
        dimsAttrs.getUpdateWindowDims(), dimsAttrs.getInsertedWindowDims());

    int64_t scatterIndicesVectorSize =
        canonicalIndices.getType().cast<RankedTensorType>().getDimSize(1);
    auto canonicalDimsAttrs = ScatterDimensionNumbersAttr::get(
        rewriter.getContext(),
        /*update_window_dims=*/
        llvm::to_vector<4>(llvm::seq<int64_t>(1, operandRank + 1)),
        /*inserted_window_dims=*/llvm::None,
        /*scatter_dims_to_operand_dims=*/
        llvm::to_vector<4>(llvm::seq<int64_t>(0, scatterIndicesVectorSize)),
        /*index_vector_dim=*/1);

    auto newScatterOp = rewriter.create<ScatterOp>(
        loc, TypeRange(ValueRange(canonicalOperands)), canonicalOperands,
        canonicalIndices, canonicalUpdates, canonicalDimsAttrs);
    Region& region = newScatterOp.getUpdateComputation();
    rewriter.inlineRegionBefore(scatterOp.getUpdateComputation(), region,
                                region.end());

    SmallVector<Value> transposedResults =
        transposeTensors(rewriter, loc, newScatterOp.getResults(),
                         getInversePermutation(operandPermutation));
    rewriter.replaceOp(scatterOp, transposedResults);
    return success();
  }
};

struct HloCanonicalizeScatterPass
    : impl::HloCanonicalizeScatterPassBase<HloCanonicalizeScatterPass> {
  void runOnOperation() override {
    MLIRContext* ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<CanonicalizeScatterPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createHloCanonicalizeScatterPass() {
  return std::make_unique<HloCanonicalizeScatterPass>();
}

}  // namespace mhlo
}  // namespace mlir
