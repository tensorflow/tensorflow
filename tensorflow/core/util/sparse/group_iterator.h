#ifndef TENSORFLOW_UTIL_SPARSE_GROUP_ITERATOR_H_
#define TENSORFLOW_UTIL_SPARSE_GROUP_ITERATOR_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/port.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"

namespace tensorflow {
namespace sparse {

class GroupIterable;  // Predeclare GroupIterable for Group.

// This class is returned when dereferencing a GroupIterable iterator.
// It provides the methods group(), indices(), and values(), which
// provide access into the underlying SparseTensor.
class Group {
 public:
  Group(GroupIterable* iter, int64 loc, int64 next_loc)
      : iter_(iter), loc_(loc), next_loc_(next_loc) {}

  std::vector<int64> group() const;
  TTypes<int64>::UnalignedConstMatrix indices() const;
  template <typename T>
  typename TTypes<T>::UnalignedVec values() const;

 private:
  GroupIterable* iter_;
  int64 loc_;
  int64 next_loc_;
};

/////////////////
// GroupIterable
/////////////////
//
// Returned when calling sparse_tensor.group({dim0, dim1, ...}).
//
// Please note: the sparse_tensor should already be ordered according
// to {dim0, dim1, ...}.  Otherwise this iteration will return invalid groups.
//
// Allows grouping and iteration of the SparseTensor according to the
// subset of dimensions provided to the group call.
//
// The actual grouping dimensions are stored in the
// internal vector group_dims_.  Iterators inside the iterable provide
// the three methods:
//
// *  group(): returns a vector with the current group dimension values.
// *  indices(): a map of index, providing the indices in
//    this group.
// *  values(): a map of values, providing the values in
//    this group.
//
// To iterate across GroupIterable, see examples in README.md.
//

// Forward declaration of SparseTensor
class GroupIterable {
 public:
  typedef gtl::ArraySlice<int64> VarDimArray;

  GroupIterable(Tensor ix, Tensor vals, int dims, const VarDimArray& group_dims)
      : ix_(ix), vals_(vals), dims_(dims), group_dims_(group_dims) {}

  class IteratorStep;

  IteratorStep begin() { return IteratorStep(this, 0); }
  IteratorStep end() { return IteratorStep(this, ix_.dim_size(0)); }

  template <typename TIX>
  inline bool GroupMatches(const TIX& ix, int64 loc_a, int64 loc_b) const {
    bool matches = true;
    for (int d : group_dims_) {
      if (ix(loc_a, d) != ix(loc_b, d)) {
        matches = false;
      }
    }
    return matches;
  }

  class IteratorStep {
   public:
    IteratorStep(GroupIterable* iter, int64 loc)
        : iter_(iter), loc_(loc), next_loc_(loc_) {
      UpdateEndOfGroup();
    }

    void UpdateEndOfGroup();
    bool operator!=(const IteratorStep& rhs) const;
    IteratorStep& operator++();    // prefix ++
    IteratorStep operator++(int);  // postfix ++
    Group operator*() const { return Group(iter_, loc_, next_loc_); }

   private:
    GroupIterable* iter_;
    int64 loc_;
    int64 next_loc_;
  };

 private:
  friend class Group;
  Tensor ix_;
  Tensor vals_;
  const int dims_;
  const VarDimArray group_dims_;
};

// Implementation of Group::values<T>()
template <typename T>
typename TTypes<T>::UnalignedVec Group::values() const {
  return typename TTypes<T>::UnalignedVec(&(iter_->vals_.vec<T>()(loc_)),
                                          next_loc_ - loc_);
}

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_UTIL_SPARSE_GROUP_ITERATOR_H_
