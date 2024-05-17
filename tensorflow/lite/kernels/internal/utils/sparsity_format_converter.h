/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_UTILS_SPARSITY_FORMAT_CONVERTER_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_UTILS_SPARSITY_FORMAT_CONVERTER_H_

#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"

namespace tflite {
namespace internal {
namespace sparsity {

// A converter that keeps an internal representation of sparse tensor parameters
// and converts tensors between dense and sparse formats.
template <typename T>
class FormatConverter {
 public:
  /*
   * Creates a dense to sparse converter.
   * @param shape             Shape of the dense tensor.
   * @param traversal_order   In what order to traverse all dimensions,
   *                          including block dimensions.
   * @param format            Whether each dimension in the dense tensor is
   *                          dense or sparse (not in the traversal order).
   * @param block_size        Size of each block dimension.
   * @param block_map         Map from block dimension to original tensor
   *                          dimension.
   */
  FormatConverter(const std::vector<int>& shape,
                  const std::vector<int>& traversal_order,
                  const std::vector<TfLiteDimensionType>& format,
                  const std::vector<int>& block_size = {},
                  const std::vector<int>& block_map = {});

  /*
   * Creates a sparse to dense converter.
   * @param shape             Shape of the target dense tensor.
   * @param traversal_order   In what order to traverse all dimensions,
   *                          including block dimensions.
   * @param format            Whether each dimension in the dense tensor is
   *                          dense or sparse (not in the traversal order).
   * @param dense_size        Size of each dense dimension in the sparse tensor.
   *                          Should be 0 for sparse dimensions.
   * @param segments          Segments of each dimension in the sparse tensor.
   *                          Should be empty for dense dimensions.
   * @param indices           Indices in the dense tensor for each dimension.
   *                          Should be empty for dense dimensions.
   * @param block_map         Map from block dimension to original tensor
   *                          dimension.
   */
  FormatConverter(const std::vector<int>& shape,
                  const std::vector<int>& traversal_order,
                  const std::vector<TfLiteDimensionType>& format,
                  const std::vector<int>& dense_size,
                  const std::vector<std::vector<int>>& segments,
                  const std::vector<std::vector<int>>& indices,
                  const std::vector<int>& block_map = {});

  /* Creates a sparse to dense converter.
   * @param shape      Shape of the target dense tensor.
   * @param sparsity   Sparsity parameter of the sparse TfLiteTensor.
   */
  FormatConverter(const std::vector<int>& shape,
                  const TfLiteSparsity& sparsity);

  const std::vector<T>& GetData() { return data_; }
  const std::vector<std::vector<int>>& GetDimMetadata() {
    return dim_metadata_;
  }

  // Method for dense to sparse conversion. Need to call GetData() method to get
  // the compressed data.
  TfLiteStatus DenseToSparse(const T* src_data);

  // Method for sparse to dense conversion. Need to call GetData() method to get
  // the decompressed data.
  TfLiteStatus SparseToDense(const T* src_data);
  // Method for sparse to dense conversion with caller provided buffer. No need
  // to call GetData() with this method.
  TfLiteStatus SparseToDense(const T* src_data, const size_t dest_size,
                             T* dest_data, TfLiteContext* context = nullptr);

 private:
  // Helper function for initializing this converter for sparse to dense
  // conversion.
  void InitSparseToDenseConverter(std::vector<int> shape,
                                  std::vector<int> traversal_order,
                                  std::vector<TfLiteDimensionType> format,
                                  std::vector<int> dense_size,
                                  std::vector<std::vector<int>> segments,
                                  std::vector<std::vector<int>> indices,
                                  std::vector<int> block_map);

  // A recursive function to fetch data from the compressed src_data buffer and
  // populate the dense buffer.
  void Populate(const T* src_data, std::vector<int> indices, int level,
                int prev_idx, int* src_data_ptr, T* dest_data);

  // Check if val is equal to zero.
  bool IsZero(const T val);

  // Shape of the conceptual dense tensor.
  std::vector<int> dense_shape_;
  // Shape of the dense tensor with inner blocks reduced. For example, a (4, 4)
  // tensor with (2, 2) block has blocked_shape (2, 2).
  std::vector<int> blocked_shape_;
  // Total number of elements in the dense tensor.
  size_t dense_size_;
  // Has n(original dimension)+k(block_dimension) elements.
  std::vector<int> traversal_order_;
  // Format of each dimension in the traversal order.
  std::vector<TfLiteDimensionType> format_;
  // Size of each block dimension, in the same order as block map.
  std::vector<int> block_size_;
  // Map from block dimension to the original tensor dimension.
  std::vector<int> block_map_;
  // Metadata of each dimension in the traversal order.
  // Each dimension needs two vectors. For dense dimensions, the first vector
  // stores the size of that dimension, and the second vector is empty. For
  // sparse dimensions, the first vector stores the segments and the second one
  // stores the indices.
  std::vector<std::vector<int>> dim_metadata_;
  // Actual buffer holding data after conversion. Could be sparse buffer or
  // dense buffer.
  std::vector<T> data_;
};

extern template class FormatConverter<int32_t>;
extern template class FormatConverter<int8_t>;
extern template class FormatConverter<float>;
extern template class FormatConverter<Eigen::half>;
}  // namespace sparsity
}  // namespace internal
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_UTILS_SPARSITY_FORMAT_CONVERTER_H_
