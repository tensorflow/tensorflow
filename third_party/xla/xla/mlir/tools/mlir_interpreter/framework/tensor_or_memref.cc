/* Copyright 2022 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/mlir/tools/mlir_interpreter/framework/tensor_or_memref.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace interpreter {

std::optional<int64_t> BufferView::GetPhysicalIndex(
    llvm::ArrayRef<int64_t> view_indices) const {
  int64_t result = offset;
  if (!InBounds(view_indices)) {
    return std::nullopt;
  }
  for (int64_t i = 0; i < view_indices.size(); ++i) {
    result += view_indices[i] * strides[i];
  }
  return result;
}

bool BufferView::InBounds(llvm::ArrayRef<int64_t> view_indices) const {
  if (view_indices.size() > sizes.size()) {
    return false;
  }
  for (auto [index, size] : llvm::zip(view_indices, sizes)) {
    if (index < 0 || index >= size) {
      return false;
    }
  }
  return true;
}

SmallVector<int64_t> BufferView::GetDefaultStrides(ArrayRef<int64_t> sizes) {
  SmallVector<int64_t> result(sizes.size());
  int64_t stride = 1;
  for (int64_t i = result.size() - 1; i >= 0; --i) {
    result[i] = stride;
    stride *= sizes[i];
  }
  return result;
}

SmallVector<int64_t> BufferView::GetStridesForLayout(ArrayRef<int64_t> sizes,
                                                     ArrayRef<int64_t> layout) {
  if (layout.empty()) return GetDefaultStrides(sizes);
  auto inverse_layout = invertPermutationVector(layout);
  SmallVector<int64_t> result(sizes.size());
  int64_t stride = 1;
  for (int64_t i = 0; i < layout.size(); ++i) {
    result[inverse_layout[i]] = stride;
    stride *= sizes[inverse_layout[i]];
  }
  return result;
}

LogicalResult BufferView::Slice(int64_t dim_index, int64_t dim_offset) {
  llvm::SmallVector<int64_t> offsets(Rank(), 0);
  offsets[dim_index] = dim_offset;
  if (auto new_offset = GetPhysicalIndex(offsets)) {
    offset = *new_offset;
  } else {
    return failure();
  }
  if (dim_index >= Rank()) --*num_vector_dims;
  strides.erase(strides.begin() + dim_index);
  sizes.erase(sizes.begin() + dim_index);
  return success();
}

LogicalResult BufferView::Slice(int64_t dim_index, int64_t dim_offset,
                                int64_t dim_size, int64_t dim_stride) {
  llvm::SmallVector<int64_t> offsets(Rank(), 0);
  offsets[dim_index] = dim_offset;
  if (dim_size == 0) {
    offset = 0;
  } else if (auto new_offset = GetPhysicalIndex(offsets)) {
    offset = *new_offset;
  } else {
    return failure();
  }
  sizes[dim_index] = dim_size;
  strides[dim_index] *= dim_stride;
  return success();
}

LogicalResult BufferView::Subview(ArrayRef<int64_t> subview_offsets,
                                  ArrayRef<int64_t> subview_sizes,
                                  ArrayRef<int64_t> subview_strides) {
  if (auto new_offset = GetPhysicalIndex(subview_offsets)) {
    offset = *new_offset;
  } else {
    return failure();
  }

  for (auto [in_size, subview_offset, subview_size, subview_stride] :
       llvm::zip(sizes, subview_offsets, subview_sizes, subview_strides)) {
    int64_t limit_index = subview_offset + (subview_size - 1) * subview_stride;
    if (subview_offset < 0 || subview_offset >= in_size || limit_index < 0 ||
        limit_index >= in_size) {
      return failure();
    }
  }

  for (auto [in_stride, subview_stride] : llvm::zip(strides, subview_strides)) {
    in_stride *= subview_stride;
  }
  sizes = llvm::to_vector(subview_sizes);
  return success();
}

int64_t BufferView::GetNumElements(bool include_vector_dims) const {
  size_t n = 1;
  for (auto size : ArrayRef<int64_t>(sizes).drop_back(
           include_vector_dims ? 0 : num_vector_dims.value_or(0))) {
    n *= size;
  }
  return n;
}

std::optional<int64_t> BufferView::GetCollapsedStride(
    llvm::ArrayRef<int64_t> dims) const {
  using StrideAndDim = std::pair<int64_t, int64_t>;
  llvm::SmallVector<StrideAndDim> strides_and_dims;
  for (auto dim : dims) {
    if (sizes[dim] != 1) {
      strides_and_dims.emplace_back(strides[dim], dim);
    }
  }

  if (strides_and_dims.empty()) {
    return 0;
  }

  llvm::sort(strides_and_dims);
  int64_t next_stride = strides_and_dims.front().first;
  for (auto [stride, dim] : strides_and_dims) {
    if (stride != next_stride) {
      return std::nullopt;
    }
    next_stride *= sizes[dim];
  }
  return strides_and_dims.front().first;
}

}  // namespace interpreter
}  // namespace mlir
