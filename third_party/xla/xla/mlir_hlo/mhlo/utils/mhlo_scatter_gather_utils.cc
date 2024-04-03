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

// This file implements utilities for the canonicalization of ScatterOp and
// GatherOp.

#include "mhlo/utils/mhlo_scatter_gather_utils.h"

#include <utility>

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"

namespace mlir {
namespace mhlo {

template <typename R>
static bool isSeq(R&& range, int64_t start, int64_t size) {
  return llvm::equal(range, llvm::seq<int64_t>(start, start + size));
}

static SmallVector<int64_t> getInversePermutation(
    llvm::ArrayRef<int64_t> permutation) {
  SmallVector<int64_t> inversePermutation(permutation.size());
  for (size_t i = 0, e = permutation.size(); i < e; ++i)
    inversePermutation[permutation[i]] = i;
  return inversePermutation;
}

bool isCanonicalScatter(ScatterOp scatterOp) {
  if (llvm::any_of(scatterOp.getOperandTypes(), [](Type operandType) {
        return !operandType.isa<RankedTensorType>();
      }))
    return false;

  ScatterDimensionNumbersAttr dimsAttrs =
      scatterOp.getScatterDimensionNumbers();
  auto indicesType =
      scatterOp.getScatterIndices().getType().cast<RankedTensorType>();
  auto operandType =
      scatterOp.getOperands().front().getType().cast<RankedTensorType>();

  return indicesType.getRank() == 2 && dimsAttrs.getIndexVectorDim() == 1 &&
         dimsAttrs.getInsertedWindowDims().empty() &&
         isSeq(dimsAttrs.getUpdateWindowDims(), 1, operandType.getRank()) &&
         isSeq(dimsAttrs.getScatterDimsToOperandDims(), 0,
               indicesType.getDimSize(1));
}

bool isCanonicalGather(GatherOp gatherOp) {
  const auto& startIndiceShape = gatherOp.getStartIndices().getType();
  const auto& dims = gatherOp.getDimensionNumbers();

  return startIndiceShape.getRank() == 2 && dims.getIndexVectorDim() == 1 &&
         isSeq(dims.getStartIndexMap(), 0, dims.getStartIndexMap().size()) &&
         dims.getCollapsedSliceDims().empty() &&
         isSeq(dims.getOffsetDims(), 1, dims.getOffsetDims().size());
}

// Creates a permutation that shuffles dimensions of `operands` to match the
// order in the index vector.

std::pair<SmallVector<int64_t>, SmallVector<int64_t>>
makeOperandStartIndexPermutations(ArrayRef<int64_t> dimMap, int operandRank) {
  SmallVector<int64_t> permutation{dimMap};
  permutation.reserve(operandRank);
  for (int i = 0; i < operandRank; ++i) {
    if (!llvm::is_contained(dimMap, i)) permutation.push_back(i);
  }
  return {permutation, getInversePermutation(permutation)};
}

Value insertDegenerateDimensions(OpBuilder& b, Location loc, Value tensor,
                                 ArrayRef<int64_t> dimsToInsert) {
  assert(llvm::is_sorted(dimsToInsert) && "dimsToInsert must be sorted");
  if (dimsToInsert.empty()) return tensor;
  TensorType type = tensor.getType().cast<TensorType>();
  SmallVector<int64_t> newShape{type.getShape()};
  for (int64_t dim : dimsToInsert) newShape.insert(newShape.begin() + dim, 1);
  auto newType = RankedTensorType::get(newShape, type.getElementType());

  return b
      .create<tensor::ExpandShapeOp>(
          loc, newType, tensor,
          *getReassociationIndicesForReshape(type, newType))
      .getResult();
}

// Checks if the indexVectorDim is equal to the rank of `indices`. In that
// case add the trailing 1 dimension. If indexVectorDim is not the innermost
// dimension, insert transpose to make it so.
static Value ensureIndexVectorDimPosition(OpBuilder& b, Location loc,
                                          Value indices,
                                          int64_t indexVectorDim) {
  int64_t indicesRank = indices.getType().cast<TensorType>().getRank();
  if (indexVectorDim == indicesRank - 1) return indices;
  if (indexVectorDim == indicesRank)
    return insertDegenerateDimensions(b, loc, indices, {indicesRank});

  SmallVector<int64_t> permutation;
  for (int64_t i = 0; i < indicesRank; ++i)
    if (i != indexVectorDim) permutation.push_back(i);
  permutation.push_back(indexVectorDim);
  return b.create<TransposeOp>(loc, indices, b.getI64TensorAttr(permutation))
      .getResult();
}

Value canonicalizeStartIndices(OpBuilder& b, Location loc, Value indices,
                               int64_t indexVectorDim) {
  indices = ensureIndexVectorDimPosition(b, loc, indices, indexVectorDim);

  int64_t indicesRank = indices.getType().cast<TensorType>().getRank();

  if (indicesRank == 2) return indices;
  if (indicesRank == 1) return insertDegenerateDimensions(b, loc, indices, {0});

  // Insert reshape to collapse all outer dimensions of `Indices`.
  SmallVector<ReassociationIndices> reassociation{
      llvm::to_vector<2>(llvm::seq<int64_t>(0, indicesRank - 1)),
      {indicesRank - 1}};
  return b.create<tensor::CollapseShapeOp>(loc, indices, reassociation)
      .getResult();
}

}  // namespace mhlo
}  // namespace mlir
