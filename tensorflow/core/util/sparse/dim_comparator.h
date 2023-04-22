/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_UTIL_SPARSE_DIM_COMPARATOR_H_
#define TENSORFLOW_CORE_UTIL_SPARSE_DIM_COMPARATOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
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

  DimComparator(const TTypes<int64>::Matrix& ix, const VarDimArray& order,
                const VarDimArray& shape)
      : ix_(ix), order_(order), dims_(shape.size()) {
    DCHECK_GT(order.size(), size_t{0}) << "Must order using at least one index";
    DCHECK_LE(order.size(), shape.size()) << "Can only sort up to dims";
    for (size_t d = 0; d < order.size(); ++d) {
      DCHECK_GE(order[d], 0);
      DCHECK_LT(order[d], shape.size());
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

 protected:
  const TTypes<int64>::Matrix ix_;
  const VarDimArray order_;
  const int dims_;
  const std::vector<int64>* ix_order_;
};

template <int ORDER_DIM>
class FixedDimComparator : DimComparator {
 public:
  FixedDimComparator(const TTypes<int64>::Matrix& ix, const VarDimArray& order,
                     const VarDimArray& shape)
      : DimComparator(ix, order, shape) {
    DCHECK_EQ(order.size(), ORDER_DIM);
  }
  inline bool operator()(const int64 i, const int64 j) const {
    bool value = false;
    for (int di = 0; di < ORDER_DIM; ++di) {
      const int64 d = order_[di];
      if (ix_(i, d) < ix_(j, d)) {
        value = true;
        break;
      }
      if (ix_(i, d) > ix_(j, d)) break;
    }
    return value;
  }
};

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SPARSE_DIM_COMPARATOR_H_
