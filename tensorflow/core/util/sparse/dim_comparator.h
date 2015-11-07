#ifndef TENSORFLOW_UTIL_SPARSE_DIM_COMPARATOR_H_
#define TENSORFLOW_UTIL_SPARSE_DIM_COMPARATOR_H_

#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/platform/logging.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

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
    CHECK_GT(order.size(), 0) << "Must order using at least one index";
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

  const TTypes<int64>::Matrix ix_;
  const VarDimArray order_;
  const int dims_;
};

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_SPARSE_DIM_COMPARATOR_H_
