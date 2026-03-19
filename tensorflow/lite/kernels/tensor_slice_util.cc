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
#include "tensorflow/lite/kernels/tensor_slice_util.h"

#include <cstdint>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"

namespace tflite {
namespace ops {
namespace builtin {

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

template Index<int32_t> ReadIndexVector(const TfLiteTensor* indices_tensor,
                                        const RuntimeShape& tensor_shape,
                                        const Index<int32_t>& other_indices,
                                        int64_t dim_to_read);

template Index<int64_t> ReadIndexVector(const TfLiteTensor* indices_tensor,
                                        const RuntimeShape& tensor_shape,
                                        const Index<int64_t>& other_indices,
                                        int64_t dim_to_read);

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
