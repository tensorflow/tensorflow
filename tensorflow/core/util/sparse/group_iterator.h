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

#ifndef TENSORFLOW_CORE_UTIL_SPARSE_GROUP_ITERATOR_H_
#define TENSORFLOW_CORE_UTIL_SPARSE_GROUP_ITERATOR_H_

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace sparse {

class GroupIterable;  // Predeclare GroupIterable for Group.

// This class is returned when dereferencing a GroupIterable iterator.
// It provides the methods group(), indices(), and values(), which
// provide access into the underlying SparseTensor.
class Group {
 public:
  Group(GroupIterable* iter, int64_t loc, int64_t next_loc)
      : iter_(iter), loc_(loc), next_loc_(next_loc) {}

  std::vector<int64_t> group() const;
  int64_t group_at(size_t index) const;
  TTypes<int64_t>::UnalignedConstMatrix indices() const;
  template <typename T>
  typename TTypes<T>::UnalignedVec values() const;

 private:
  GroupIterable* iter_;
  int64_t loc_;
  int64_t next_loc_;
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
  typedef gtl::ArraySlice<int64_t> VarDimArray;

  GroupIterable(Tensor ix, Tensor vals, int dims, const VarDimArray& group_dims)
      : ix_(ix),
        ix_matrix_(ix_.matrix<int64_t>()),
        vals_(vals),
        dims_(dims),
        group_dims_(group_dims.begin(), group_dims.end()) {}

  class IteratorStep;

  IteratorStep begin() { return IteratorStep(this, 0); }
  IteratorStep at(int64_t loc) {
    CHECK(loc >= 0 && loc <= ix_.dim_size(0))
        << "loc provided must lie between 0 and " << ix_.dim_size(0);
    return IteratorStep(this, loc);
  }
  IteratorStep end() { return IteratorStep(this, ix_.dim_size(0)); }

  template <typename TIX>
  inline bool GroupMatches(const TIX& ix, int64_t loc_a, int64_t loc_b) const {
    for (int d : group_dims_) {
      if (ix(loc_a, d) != ix(loc_b, d)) {
        return false;
      }
    }
    return true;
  }

  class IteratorStep {
   public:
    IteratorStep(GroupIterable* iter, int64_t loc)
        : iter_(iter), loc_(loc), next_loc_(loc_) {
      UpdateEndOfGroup();
    }

    void UpdateEndOfGroup();
    bool operator!=(const IteratorStep& rhs) const;
    bool operator==(const IteratorStep& rhs) const;
    IteratorStep& operator++();    // prefix ++
    IteratorStep operator++(int);  // postfix ++
    Group operator*() const { return Group(iter_, loc_, next_loc_); }
    int64_t loc() const { return loc_; }

   private:
    GroupIterable* iter_;
    int64_t loc_;
    int64_t next_loc_;
  };

 private:
  friend class Group;
  const Tensor ix_;
  const TTypes<int64_t>::ConstMatrix ix_matrix_;
  Tensor vals_;
  const int dims_;
  const gtl::InlinedVector<int64_t, 8> group_dims_;
};

inline int64_t Group::group_at(size_t index) const {
  const auto& ix_t = iter_->ix_matrix_;
  return ix_t(loc_, index);
}

// Implementation of Group::values<T>()
template <typename T>
typename TTypes<T>::UnalignedVec Group::values() const {
  return typename TTypes<T>::UnalignedVec(&(iter_->vals_.vec<T>()(loc_)),
                                          next_loc_ - loc_);
}

}  // namespace sparse
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_SPARSE_GROUP_ITERATOR_H_
