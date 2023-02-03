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

// Utility functions related to layouts of Shapes.

#ifndef TENSORFLOW_COMPILER_XLA_INDEX_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_INDEX_UTIL_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// Namespaced collection of (static) utilities related to indexing into
// multidimensional arrays.
class IndexUtil {
 public:
  // Converts a multidimensional index (eg {x, y, z}) into a linear index based
  // on the shape and its layout. The first index in the multi_index is
  // dimension 0.
  static inline int64_t MultidimensionalIndexToLinearIndex(
      const Shape& shape, absl::Span<const int64_t> multi_index) {
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
    for (size_t i = 0; i < multi_index.size(); ++i) {
      DCHECK_GE(multi_index[i], 0);
      DCHECK_LT(multi_index[i], shape.dimensions(i))
          << "indexing beyond extent in dimension " << i << ":"
          << "\n\tindex: " << absl::StrJoin(multi_index, ",")
          << "\n\tshape: " << ShapeUtil::HumanString(shape);
    }
    auto effective_shape = LayoutUtil::MinorToMajor(shape);
    if (effective_shape.empty()) {
      return 0;
    }
    int64_t linear_index = multi_index[effective_shape[0]];
    int64_t scale = 1;
    for (int i = 1; i < effective_shape.size(); ++i) {
      scale *= shape.dimensions(effective_shape[i - 1]);
      linear_index += scale * multi_index[effective_shape[i]];
    }
    return linear_index;
  }

  // Converts a linear index into multidimensional index (eg {x, y, z}) based on
  // the shape and its layout. The first index in the returned multidimensional
  // index is dimension 0.
  static std::vector<int64_t> LinearIndexToMultidimensionalIndex(
      const Shape& shape, int64_t linear_index);

  // Bumps a sequence of indices; e.g. {0,0,0,0} up by one index value; e.g. to
  // {0,0,0,1}. This is akin to std::next_permutation. If the index hits a limit
  // for the provided shape, the next most significant index is bumped, in a
  // counting-up process.
  //
  // E.g. for shape f32[2,3]
  //  {0,0}=>{0,1}
  //  {0,1}=>{0,2}
  //  {0,2}=>{1,0}
  //  etc.
  //
  // This is useful for traversing the indices in a literal.
  //
  // Returns true iff the indices were successfully bumped; false if we've hit
  // the limit where it can no longer be bumped in-bounds.
  static bool BumpIndices(const Shape& shape, absl::Span<int64_t> indices);

  // Calculates the stride size (in number of elements, not byte size) of a
  // given logical shape dimension (from 0 to rank-1).
  // Example:
  //  GetDimensionStride(F32[5,8,10,4]{3,2,1,0}, 1) ==
  //    sizeof(dimension(3)) * sizeof(dimension(2)) == 4 * 10
  static int64_t GetDimensionStride(const Shape& shape, int64_t dimension);

  // Returns true iff the given multi-index is contained in the bounds for the
  // shape.
  static bool IndexInBounds(const Shape& shape,
                            absl::Span<const int64_t> index);

  // Compares the given indices in lexicographic order.  lhs[0] and rhs[0] are
  // compared first, and lhs[rank-1] and rhs[rank-1] last.  If lhs is larger,
  // then -1 is returned. If rhs is larger, then 1 is returned.  Otherwise, 0 is
  // returned.
  static int CompareIndices(absl::Span<const int64_t> lhs,
                            absl::Span<const int64_t> rhs);

 private:
  IndexUtil(const IndexUtil&) = delete;
  IndexUtil& operator=(const IndexUtil&) = delete;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_INDEX_UTIL_H_
