/* Copyright 2015 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_UTIL_SPARSE_DIM_COMPARATOR_H_
#define TENSORFLOW_UTIL_SPARSE_DIM_COMPARATOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace sparse {

/////////////////
// DimComparator
/////////////////
//
// Helper class, mainly used by the IndexSortOrder. This comparator
// can be passed to e.g. std::sort, or any other sorter, to sort two
// rows of an index matrix according to the dimension(s) of interest.
// The dimensions to sort by are passed to the constructor as "order".
//
// Example: if given index matrix IX, two rows ai and bi, and order = {2,1}.
// operator() compares
//    IX(ai,2) < IX(bi,2).
// If IX(ai,2) == IX(bi,2), it compares
//    IX(ai,1) < IX(bi,1).
//
// This can be used to sort a vector of row indices into IX according to
// the values in IX in particular columns (dimensions) of interest.
class DimComparator {
 public:
  typedef typename gtl::ArraySlice<int64> VarDimArray;

  inline DimComparator(const TTypes<int64>::Matrix& ix,
                       const VarDimArray& order, int dims)
      : ix_(ix), order_(order), dims_(dims) {
    CHECK_GT(order.size(), size_t{0}) << "Must order using at least one index";
    CHECK_LE(order.size(), dims_) << "Can only sort up to dims";
    for (size_t d = 0; d < order.size(); ++d) {
      CHECK_GE(order[d], 0);
      CHECK_LT(order[d], dims);
    }
  }

  inline bool operator()(const int64 i, const int64 j) const {
    for (int di = 0; di < dims_; ++di) {
      const int64 d = order_[di];
      if (ix_(i, d) < ix_(j, d)) return true;
      if (ix_(i, d) > ix_(j, d)) return false;
    }
    return false;
  }

  // Compares two indices taken from corresponding index matrices, using the
  // standard, row-major (or lexicographic) order.  Useful for cases that need
  // to distinguish between all three orderings (<, ==, >).
  inline static int cmp(const TTypes<int64>::ConstMatrix& a_idx,
                        const TTypes<int64>::ConstMatrix& b_idx,
                        const int64 a_row, const int64 b_row, const int dims) {
    for (int d = 0; d < dims; ++d) {
      const int64 a = a_idx(a_row, d);
      const int64 b = b_idx(b_row, d);
      if (a < b) {
        return -1;
      } else if (a > b) {
        return 1;
      }
    }
    return 0;
  }

  const TTypes<int64>::Matrix ix_;
  const VarDimArray order_;
  const int dims_;
};

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_SPARSE_DIM_COMPARATOR_H_
