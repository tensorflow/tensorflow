/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

// Utility class for managing sparse array indices.

#ifndef THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SPARSE_INDEX_ARRAY_H_
#define THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SPARSE_INDEX_ARRAY_H_

#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"

namespace xla {

// Encapsulates the array of indices for a sparse array.  A SparseIndexArray
// contain indices for up to `max_indices` elements of a sparse array.  Each
// sparse index is an array of `rank` int64 value that gives the location of a
// value within a sparse array.  Note that the dimensions of the array are not
// checked (except for the rank).  To avoid confusion, we refer to the position
// of an index within a SparseIndexArray as a sparse index number.
class SparseIndexArray {
 public:
  SparseIndexArray();
  SparseIndexArray(const SparseIndexArray&) = default;
  SparseIndexArray(SparseIndexArray&&) = default;
  SparseIndexArray& operator=(const SparseIndexArray&) = default;
  SparseIndexArray& operator=(SparseIndexArray&&) = default;

  // Constructs a SparseIndexArray that can hold up to `max_indices` sparse
  // indices, with an initial contents obtained from the given array.  The rank
  // is taken from the minor dimension of the array.  The major dimension of the
  // array must not exceed `max_indices`.
  SparseIndexArray(int64 max_indices, const Array2D<int64>& indices);

  // Like above, but the array is flattened.  For example, the following are
  // equivalent:
  //
  //  SparseIndexArray(10, 3,
  //                   Array2D{
  //                     {0, 1, 2},
  //                     {3, 4, 5},
  //                     {6, 7, 8},
  //                     {9, 10, 11},
  //                   })
  //
  //  SparseIndexArray(10, 3,
  //                   {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11})
  //
  SparseIndexArray(int64 max_indices, int64 rank,
                   std::vector<int64> indices = {});
  SparseIndexArray(int64 max_indices, int64 rank,
                   tensorflow::gtl::ArraySlice<int64> indices);

  // Returns the number of elements represented by the indices stored in the
  // array.
  int64 index_count() const;

  // Returns a slice that refers to the given sparse index number. The argument
  // must be in the range [0, element_count()).
  tensorflow::gtl::ArraySlice<int64> At(int64 sparse_element_number) const;
  tensorflow::gtl::MutableArraySlice<int64> At(int64 sparse_element_number);

  // Adds the given index at the end of the array.  The new size of the
  // SparseIndexArray must not exceed `max_indices`.
  void Append(tensorflow::gtl::ArraySlice<int64> index);

  // Removes all indices from the array.
  void Clear();

  // Resizes the array to contain the given number of sparse indices.  The new
  // size must be smaller than `max_indices`.  If the new size is larger than
  // the old size, the value of the new indices is not specified.
  void Resize(int64 num_indices);

  // Returns true iff all indices are unique and occur in sorted order, and are
  // valid for the given shape.
  bool Validate(const Shape& shape) const;

  int64 rank() const { return rank_; }
  int64 max_indices() const { return max_indices_; }

  // Returns a pointer to the int64 array that holds the sparse indices.
  tensorflow::gtl::MutableArraySlice<int64> mutable_data() { return &indices_; }
  tensorflow::gtl::ArraySlice<int64> data() const { return indices_; }

  // Sorts this sparse index array along with the set of corresponding values.
  // The indices and values are sorted in the lexicographic order of the
  // indices, from smallest to largest.
  //
  // For example:
  //
  //   std::vector<float> v{10.0, 11.0, 12.0};
  //   SparseIndexArray a(10, 3,
  //                      {{3, 4, 5},
  //                       {1, 2, 3},
  //                       {2, 3, 4}});
  //   a.SortWithValues(&v);
  //   // Prints "11.0, 12.0, 10.0":
  //   std::cout << v[0] << ", " << v[1] << ", " << v[2] << std::endl;
  //
  template <typename NativeT>
  void SortWithValues(tensorflow::gtl::MutableArraySlice<NativeT> values);

 private:
  std::vector<int64> indices_;
  int64 rank_;
  int64 max_indices_;
};

template <typename NativeT>
void SparseIndexArray::SortWithValues(
    tensorflow::gtl::MutableArraySlice<NativeT> values) {
  int64 num_elements = index_count();
  CHECK_EQ(values.size(), num_elements);
  std::vector<int64> sort_order;
  sort_order.reserve(num_elements);
  for (int64 i = 0; i < num_elements; ++i) {
    sort_order.push_back(i);
  }
  auto sort_order_less = [this](int64 lhs, int64 rhs) {
    return IndexUtil::CompareIndices(At(lhs), At(rhs)) < 0;
  };
  std::sort(sort_order.begin(), sort_order.end(), sort_order_less);

  // Reorder the array elements according to sort_order.  Work through the array
  // and follow cycles so we can do the reorder in-place.
  tensorflow::gtl::InlinedVector<int64, 8> saved_index(rank());
  for (int64 i = 0; i < num_elements; ++i) {
    // sort_order[i] == -1 indicates the element has already been copied.
    if (sort_order[i] < 0) {
      continue;
    } else if (i == sort_order[i]) {
      // The element is already in sorted order.
      sort_order[i] = -1;
      continue;
    }

    std::copy_n(At(i).begin(), rank(), saved_index.begin());
    NativeT saved_value = values[i];
    int64 j = i;
    for (;;) {
      if (sort_order[j] == i) {
        std::copy_n(saved_index.begin(), rank(), At(j).begin());
        values[j] = saved_value;
        sort_order[j] = -1;
        break;
      }

      std::copy_n(At(sort_order[j]).begin(), rank(), At(j).begin());
      values[j] = values[sort_order[j]];

      int64 k = sort_order[j];
      sort_order[j] = -1;
      j = k;
    }
  }
}

}  // namespace xla

#endif  // THIRD_PARTY_TENSORFLOW_COMPILER_XLA_SPARSE_INDEX_ARRAY_H_
