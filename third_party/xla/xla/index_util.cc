/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/index_util.h"

#include <cstdint>

#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/util.h"

namespace xla {

/* static */ DimensionVector IndexUtil::LinearIndexToMultidimensionalIndex(
    const Shape& shape, int64_t linear_index) {
  DCHECK_GE(linear_index, 0);
  DCHECK_LT(linear_index, ShapeUtil::ElementsIn(shape));

  // The following formula computes each element of the multidimensional index
  // (See comments in MultidimensionalIndexToLinearIndex for notation):
  //
  // I{L(0)} = linear_index % D{L(0)}
  // I{L(1)} = (linear_index / D{L(0)}) % D{L(1)}
  // I{L(2)} = (linear_index / (D{L(0)} * D{L(1)})) % D{L(2)}
  // ...
  DimensionVector multi_index(shape.dimensions_size());

  // Accumulated product D{L(0)} * D{L(1)} * ...
  int64_t divisor = 1;
  for (auto dimension : LayoutUtil::MinorToMajor(shape)) {
    multi_index[dimension] =
        (linear_index / divisor) % shape.dimensions(dimension);
    divisor *= shape.dimensions(dimension);
  }
  return multi_index;
}

/* static */ int64_t IndexUtil::GetDimensionStride(const Shape& shape,
                                                   int64_t dimension) {
  int64_t stride = 1;
  for (auto dim : LayoutUtil::MinorToMajor(shape)) {
    if (dim == dimension) {
      break;
    }
    stride *= shape.dimensions()[dim];
  }
  return stride;
}

/* static */ bool IndexUtil::IndexInBounds(const Shape& shape,
                                           absl::Span<const int64_t> index) {
  int64_t rank = shape.rank();
  const int64_t index_size = index.size();
  if (rank != index_size) {
    return false;
  }
  for (int64_t d = 0; d < rank; ++d) {
    if (index[d] >= shape.dimensions(d)) {
      return false;
    }
  }
  return true;
}

/* static */ int IndexUtil::CompareIndices(absl::Span<const int64_t> lhs,
                                           absl::Span<const int64_t> rhs) {
  int64_t rank = lhs.size();
  const int64_t rhs_rank = rhs.size();
  CHECK_EQ(rhs_rank, rank);
  for (int64_t dim = 0; dim < rank; ++dim) {
    if (lhs[dim] < rhs[dim]) {
      return -1;
    } else if (lhs[dim] > rhs[dim]) {
      return 1;
    }
  }
  return 0;
}

}  // namespace xla
