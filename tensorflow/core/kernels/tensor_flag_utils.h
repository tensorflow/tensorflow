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

// Helpers for parsing tensors as runtime flags.
#ifndef TENSORFLOW_CORE_KERNELS_TENSOR_FLAG_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_TENSOR_FLAG_UTILS_H_

#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace tensor_flag_utils {

// Converts tensor.vec<Tindices> to an std::vector<Tindices> object, appends
// the value num_nonzero_entries_in_sparse_mat, and returns the result.
template <typename Tindices>
std::vector<Tindices> ParseRowStartIndices(
    const tensorflow::Tensor& tensor,
    const Tindices num_nonzero_entries_in_sparse_mat);

// Returns OkStatus() if and only if config is a float scalar or a matrix with
// dimensions M x 3. If config is a scalar then config must be in the range
// [0, 1.0). If config is a matrix then config must have shape M x 3, all of
// its entries must be positive, and entries in the last column may not
// exceed 1.0. If config is a matrix then it may not be empty.
absl::Status ValidateSparseMatrixShardingConfig(const Tensor& config);

// Returns OkStatus() if and only if config is a float scalar or a non-empty
// matrix with dimensions M x 2.
absl::Status ValidateScalarQuantityShardingConfig(const Tensor& config);

// Returns the last entry of the first row in config_mat for which the first
// two entries are no smaller than the respective entries in key. If no such
// row exists then returns the last entry in the last row in config_mat.
// config_mat may not be empty.
template <typename MatrixType, typename K>
MatrixType FindConfigValueForKey(
    const typename TTypes<MatrixType>::ConstMatrix& config_mat,
    const std::pair<K, K>& key);

// Returns the last entry of the first row in config_mat for which the first
// two entries are no smaller than the respective entries in key. If no such
// row exists then returns the last entry in the last row in config_mat.
// config_mat may not be empty.
template <typename MatrixType, typename K>
MatrixType FindConfigValueForKey(
    const typename TTypes<MatrixType>::ConstMatrix& config_mat, const K key);

// Returns largest multiple of bucket_size less than value.
// Expects 1 <= bucket_size <= value.
template <typename Tindices>
Tindices GetLinearBucket(const Tindices value, const Tindices bucket_size);

// Returns the largest power of bucket_size less than value.
// Expects 1 <= bucket_size <= value. If bucket_size = 1, returns 1.
template <typename Tindices>
Tindices GetPowerBucket(const Tindices value, const Tindices bucket_size);

}  // namespace tensor_flag_utils
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_TENSOR_FLAG_UTILS_H_
