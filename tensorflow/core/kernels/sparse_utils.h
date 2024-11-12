/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Helpers for writing OpKernels for sparse tensors.
#ifndef TENSORFLOW_CORE_KERNELS_SPARSE_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_SPARSE_UTILS_H_

#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace sparse_utils {

// Find the index i of the first element for which
// indices_mat(sparse_index_begin, 0) < indices_mat(i, 0).
// The search is conducted in the open interval
// [sparse_index_begin, indices_mat.dimension(0)) and when no such i is found,
// indices_mat.dimension(0) is returned.
// indices_mat(k, 0) should be non-decreasing over the interval
// [begin, indices_mat.dimension(0)).
// Requires 0 <= sparse_index_begin < indices_mat.dimension(0).
template <typename Tindices>
Tindices FindNextDenseRowStartIndex(
    const Tindices sparse_index_begin,
    const typename TTypes<Tindices>::ConstMatrix& indices_mat);

// Returns the vector v of indices in indices_mat at which new dense matrix
// rows begin.
// v.front() = 0, v.back() = indices_mat.dimension(0), and for i > 0,
// v[i] - v[i-1] is the length of the ith dense row in indices_mat.
// *contains_empty_rows = true if and only if indices_mat contains empty rows
// (rows without values) between row 0 and the last row.
template <typename Tindices>
std::vector<Tindices> GetStartIndicesOfEachDenseRow(
    const typename TTypes<Tindices>::ConstMatrix& indices_mat,
    bool* contains_empty_rows);

// Converts tensor.vec<Tindices> to an std::vector<Tindices> object, appends
// the value num_nonzero_entries_in_sparse_mat, and returns the result.
template <typename Tindices>
std::vector<Tindices> ParseRowStartIndices(
    const tensorflow::Tensor& tensor,
    const Tindices num_nonzero_entries_in_sparse_mat);

// Returns true if and only if the sparse matrix indices_mat whose row start
// indices are represented by row_start_indices has empty dense rows
// (between its first and last dense rows).
// This function satisfies the identity row_start_indices ==
// GetStartIndicesOfEachDenseRow(indices_mat, &return_value).
template <typename Tindices>
bool ContainsEmptyRows(const std::vector<Tindices>& row_start_indices);

// Methods for validating sparse indices.
enum class IndexValidation {
  kNone,      // Indices are not used by the op, or are not directly accessible
              // (e.g. on GPU).
  kOrdered,   // Indices must be unique, in lexicographical order, and within
              // safe bounds.
  kUnordered  // Indices must be within safe bounds, but may repeat or appear
              // out-of-order.
};

// Validates the three component tensors of a sparse tensor have the proper
// shapes.  Also validates index values according to the method supplied.
template <typename Tindices>
absl::Status ValidateSparseTensor(const Tensor& indices, const Tensor& values,
                                  const Tensor& shape,
                                  IndexValidation index_validation);

}  // namespace sparse_utils
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SPARSE_UTILS_H_
