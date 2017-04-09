/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/index_util.h"

#include <algorithm>
#include <string>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

/* static */ int64 IndexUtil::MultidimensionalIndexToLinearIndex(
    const Shape& shape, tensorflow::gtl::ArraySlice<int64> multi_index) {
  DCHECK_EQ(shape.dimensions_size(), multi_index.size());
  // Padding and nested layouts not supported yet.
  DCHECK_EQ(0, shape.layout().padded_dimensions_size());

  for (size_t i = 0; i < multi_index.size(); ++i) {
    DCHECK_GE(multi_index[i], 0);
    DCHECK_LT(multi_index[i], shape.dimensions(i))
        << "indexing beyond extent in dimension " << i << ":"
        << "\n\tindex: " << tensorflow::str_util::Join(multi_index, ",")
        << "\n\tshape: " << ShapeUtil::HumanString(shape);
  }

  // Let the array be sized like so for dimensions i from 0 to n-1:
  //
  //   [D{n-1} x D{n-2} x .. x D{0}]
  //
  // Let the order of the dimensions in the minor_to_major field in
  // Layout be:
  //
  //   L(0), L(1), ... , L(n-1)
  //
  // where L(0) is the most-minor dimension and L(n-1) the most-major. The
  // multidimensional index:
  //
  //   [I{0}, I{1}, ... , I{n-1}]
  //
  // then corresponds to the following linear index:
  //
  // linear_index =
  //   (((  ... + I{L(2)}) * D{L(1)} + I{L(1)}) * D{L(0)} + I{L(0)}
  //
  // or equivalently:
  //
  // linear_index =
  //   I{L(n-1)} * (D{L(n-2)} * D{L(n-3)} * D{L(n-4)} *     ....    D{L(0)}) +
  //   I{L(n-2)} *             (D{L(n-3)} * D{L(n-4)} *     ....    D{L(0)}) +
  //   I{L(n-3)} *                         (D{L(n-4)} *     ....    D{L(0)}) +
  //                                   ...                                   +
  //   I{L(2)} *                                         (D{L(1)} * D{L(0)}) +
  //   I{L(1)} *                                                    D{L(0)}  +
  //   I{L(0)}
  //
  // We compute the linear index value by accumulating the terms above from
  // I{L(0)} up to I{L(n-1)}. Scale accumulates the product term D{L(0}} *
  // D{L(1)} * ...

  // Scale factor holding the growing product of D{L(i)} terms.
  int64 scale = 1;
  int64 linear_index = 0;
  bool first = true;
  for (auto dimension : shape.layout().minor_to_major()) {
    if (first) {
      // Avoid two multiplies on the first loop iteration
      linear_index = multi_index[dimension];
      scale = shape.dimensions(dimension);
      first = false;
    } else {
      linear_index += scale * multi_index[dimension];
      scale *= shape.dimensions(dimension);
    }
  }
  return linear_index;
}

/* static */ std::vector<int64> IndexUtil::LinearIndexToMultidimensionalIndex(
    const Shape& shape, int64 linear_index) {
  // Padding and nested layouts not supported yet.
  DCHECK_EQ(0, shape.layout().padded_dimensions_size());
  DCHECK_GE(linear_index, 0);
  DCHECK_LT(linear_index, ShapeUtil::ElementsIn(shape));

  // The following formula computes each element of the multidimensional index
  // (See comments in MultidimensionalIndexToLinearIndex for notation):
  //
  // I{L(0)} = linear_index % D{L(0)}
  // I{L(1)} = (linear_index / D{L(0)}) % D{L(1)}
  // I{L(2)} = (linear_index / (D{L(0)} * D{L(1)})) % D{L(2)}
  // ...
  std::vector<int64> multi_index(shape.dimensions_size());

  // Accumulated product D{L(0)} * D{L(1)} * ...
  int64 divisor = 1;
  for (auto dimension : shape.layout().minor_to_major()) {
    multi_index[dimension] =
        (linear_index / divisor) % shape.dimensions(dimension);
    divisor *= shape.dimensions(dimension);
  }
  return multi_index;
}

/* static */ bool IndexUtil::BumpIndices(const Shape& shape,
                                         std::vector<int64>* indices) {
  for (int64 dimno = indices->size() - 1; dimno >= 0; --dimno) {
    int64 limit = shape.dimensions(dimno);
    if ((*indices)[dimno] + 1 < limit) {
      (*indices)[dimno]++;
      std::fill(indices->begin() + dimno + 1, indices->end(), 0);
      return true;
    }
  }
  return false;
}

}  // namespace xla
