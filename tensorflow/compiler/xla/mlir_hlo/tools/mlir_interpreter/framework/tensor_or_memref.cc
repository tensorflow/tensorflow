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

#include "tools/mlir_interpreter/framework/tensor_or_memref.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <optional>
#include <utility>

#include "mlir/Dialect/Utils/IndexingUtils.h"

namespace mlir {
namespace interpreter {

std::optional<int64_t> BufferView::getPhysicalIndex(
    llvm::ArrayRef<int64_t> viewIndices) const {
  int64_t result = offset;
  if (!inBounds(viewIndices)) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < viewIndices.size(); ++i) {
    result += viewIndices[i] * strides[i];
  }
  return result;
}

bool BufferView::inBounds(llvm::ArrayRef<int64_t> viewIndices) const {
  if (viewIndices.size() > sizes.size()) return false;
  for (auto [index, size] : llvm::zip(viewIndices, sizes)) {
    if (index < 0 || index >= size) return false;
  }
  return true;
}

SmallVector<int64_t> BufferView::getDefaultStrides(ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> result(sizes.size());
  int64_t stride = 1;
  for (int64_t i = result.size() - 1; i >= 0; --i) {
    result[i] = stride;
    stride *= sizes[i];
  }
  return result;
}

SmallVector<int64_t> BufferView::getStridesForLayout(ArrayRef<int64_t> sizes,
                                                     ArrayRef<int64_t> layout) {
  if (layout.empty()) return getDefaultStrides(sizes);
  auto inverseLayout = invertPermutationVector(layout);
  SmallVector<int64_t> result(sizes.size());
  int64_t stride = 1;
  for (int64_t i = 0; i < layout.size(); ++i) {
    result[inverseLayout[i]] = stride;
    stride *= sizes[inverseLayout[i]];
  }
  return result;
}

LogicalResult BufferView::slice(int64_t dimIndex, int64_t dimOffset) {
  llvm::SmallVector<int64_t> offsets(rank(), 0);
  offsets[dimIndex] = dimOffset;
  if (auto newOffset = getPhysicalIndex(offsets)) {
    offset = *newOffset;
  } else {
    return failure();
  }
  if (dimIndex >= rank()) --*numVectorDims;
  strides.erase(strides.begin() + dimIndex);
  sizes.erase(sizes.begin() + dimIndex);
  return success();
}

LogicalResult BufferView::slice(int64_t dimIndex, int64_t dimOffset,
                                int64_t dimSize, int64_t dimStride) {
  llvm::SmallVector<int64_t> offsets(rank(), 0);
  offsets[dimIndex] = dimOffset;
  if (dimSize == 0) {
    offset = 0;
  } else if (auto newOffset = getPhysicalIndex(offsets)) {
    offset = *newOffset;
  } else {
    return failure();
  }
  sizes[dimIndex] = dimSize;
  strides[dimIndex] *= dimStride;
  return success();
}

LogicalResult BufferView::subview(ArrayRef<int64_t> subviewOffsets,
                                  ArrayRef<int64_t> subviewSizes,
                                  ArrayRef<int64_t> subviewStrides) {
  if (auto newOffset = getPhysicalIndex(subviewOffsets)) {
    offset = *newOffset;
  } else {
    return failure();
  }

  for (auto [inSize, subview_offset, subview_size, subview_stride] :
       llvm::zip(sizes, subviewOffsets, subviewSizes, subviewStrides)) {
    int64_t limitIndex = subview_offset + (subview_size - 1) * subview_stride;
    if (subview_offset < 0 || subview_offset >= inSize || limitIndex < 0 ||
        limitIndex >= inSize) {
      return failure();
    }
  }

  for (auto [in_stride, subview_stride] : llvm::zip(strides, subviewStrides)) {
    in_stride *= subview_stride;
  }
  sizes = llvm::to_vector(subviewSizes);
  return success();
}

int64_t BufferView::getNumElements(bool includeVectorDims) const {
  size_t n = 1;
  for (auto size : ArrayRef<int64_t>(sizes).drop_back(
           includeVectorDims ? 0 : numVectorDims.value_or(0)))
    n *= size;
  return n;
}

std::optional<int64_t> BufferView::getCollapsedStride(
    llvm::ArrayRef<int64_t> dims) const {
  using StrideAndDim = std::pair<int64_t, int64_t>;
  llvm::SmallVector<StrideAndDim> stridesAndDims;
  for (auto dim : dims) {
    if (sizes[dim] != 1) {
      stridesAndDims.emplace_back(strides[dim], dim);
    }
  }

  if (stridesAndDims.empty()) return 0;

  llvm::sort(stridesAndDims);
  int64_t nextStride = stridesAndDims.front().first;
  for (auto [stride, dim] : stridesAndDims) {
    if (stride != nextStride) return std::nullopt;
    nextStride *= sizes[dim];
  }
  return stridesAndDims.front().first;
}

}  // namespace interpreter
}  // namespace mlir
