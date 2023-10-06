/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_TENSOR_SLICE_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_TENSOR_SLICE_UTIL_H_

#include <algorithm>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace ops {
namespace builtin {

// Element i contains the index of an entry in the i-th dimension.
template <typename IndexType>
using Index = std::vector<IndexType>;

// Returns true if the `array` contains `value`.
// Always returns false if the array is empty.
inline bool ArrayContains(const int64_t* array, int size, int64_t value) {
  if (size == 0) {
    return false;
  }
  return std::find(array, array + size, value) != array + size;
}

// Creates a new Index based on the provided `scatter_dims`.
// Example:
// For `index`={s0, s1}, `scatter_dims`=[1, 0], returns {s2,s1,0}.
// Result has same size as rank of input. All the result dimensions not in
// `scatter_dims` get the value 0.
template <typename IndexType>
TfLiteStatus ScatterIndex(const Index<IndexType>& index,
                          const int64_t* scatter_dims, int num_scatter_dims,
                          int64_t to_rank, Index<IndexType>* result) {
  if (result == nullptr) {
    return kTfLiteError;
  }

  *result = Index<IndexType>(to_rank, 0);
  for (int idx = 0; idx < num_scatter_dims; ++idx) {
    if (scatter_dims[idx] >= result->size()) {
      return kTfLiteError;
    }
    (*result)[scatter_dims[idx]] = index[idx];
  }
  return kTfLiteOk;
}

// A helper function that converts a tensor index into a flat array index.
template <typename IndexType>
IndexType TensorIndexToFlat(const IndexType* index, const int64_t dims,
                            const RuntimeShape& shape) {
  // If it's a scalar, just return the index of the first element.
  if (dims == 0) {
    return 0;
  }
  IndexType flat_index = index[0];
  for (int64_t i = 1; i < dims; ++i) {
    flat_index = flat_index * shape.Dims(i) + index[i];
  }
  return flat_index;
}

template <typename IndexType>
Index<IndexType> AddIndices(const Index<IndexType>& index1,
                            const Index<IndexType>& index2) {
  Index<IndexType> result;
  result.reserve(index1.size());
  for (int64_t dim = 0; dim < index1.size(); ++dim) {
    result.push_back(index1[dim] + index2[dim]);
  }
  return result;
}

// Creates a new Index with the number of dimensions increased with respect to
// `avoided_dims` array.
// Example: `index`=[i, j], `avoided_dims`=[1], the result is [i, 0, j]
template <typename IndexType>
TfLiteStatus ExpandDims(const Index<IndexType>& index,
                        const int64_t* avoided_dims, int num_avoided_dims,
                        Index<IndexType>* result) {
  std::vector<int64_t> scatter_dims;
  scatter_dims.reserve(index.size());
  int64_t ctr = 0;
  for (int idx = 0; idx < index.size(); ++idx) {
    while (ArrayContains(avoided_dims, num_avoided_dims, ctr)) {
      ++ctr;
    }
    scatter_dims.push_back(ctr);
    ++ctr;
  }
  TF_LITE_ENSURE_STATUS(ScatterIndex(index, scatter_dims.data(),
                                     scatter_dims.size(),
                                     index.size() + num_avoided_dims, result));
  return kTfLiteOk;
}

// Reads one array from a given tensor.
// Example: `other_indices`=[j, l, m], `dim_to_read`= 2
// The resulting read array is: [j, l, :, m]
template <typename IndexType>
Index<IndexType> ReadIndexVector(const TfLiteTensor* indices_tensor,
                                 const RuntimeShape& tensor_shape,
                                 const Index<IndexType>& other_indices,
                                 int64_t dim_to_read) {
  Index<IndexType> index;
  index.reserve(tensor_shape.DimensionsCount());
  int shift = 0;
  for (int64_t dim = 0; dim < tensor_shape.DimensionsCount(); ++dim) {
    if (dim == dim_to_read) {
      index.push_back(0);
      shift = 1;
    } else {
      index.push_back(other_indices[dim - shift]);
    }
  }
  int64_t index_vector_size = tensor_shape.Dims(dim_to_read);
  Index<IndexType> result;
  result.reserve(index_vector_size);
  for (IndexType index_vector_idx = 0; index_vector_idx < index_vector_size;
       ++index_vector_idx) {
    index[dim_to_read] = index_vector_idx;

    IndexType flat_index = TensorIndexToFlat(
        index.data(), tensor_shape.DimensionsCount(), tensor_shape);
    const IndexType* tensor_data = GetTensorData<IndexType>(indices_tensor);
    result.push_back(tensor_data[flat_index]);
  }
  return result;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_TENSOR_SLICE_UTIL_H_
