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

#include "tensorflow/core/kernels/tensor_flag_utils.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {
namespace tensor_flag_utils {

Status ValidateSparseMatrixShardingConfig(const Tensor& config) {
  if (TensorShapeUtils::IsScalar(config.shape())) {
    const float scalar_config = config.template scalar<float>()();
    if (0 < scalar_config && scalar_config <= 1.0) {
      return Status::OK();
    }
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat("Expected config to be in range (0, 1] but instead found ",
                     scalar_config));
  }
  if (!TensorShapeUtils::IsMatrix(config.shape())) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Expected config to be either scalar or matrix "
                               "but instead found tensor of rank ",
                               config.dims()));
  }
  if (config.dim_size(1) != 3) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat(
            "Expected config matrix to have dim(1) = 3 but instead found ",
            config.dim_size(1)));
  }

  auto config_matrix = config.matrix<float>();
  for (int i = 0; i < config.dim_size(0); ++i) {
    if (0 > config_matrix(i, 0)) {
      return errors::InvalidArgument(
          "First column of fraction_rows_per_thread_config "
          "should "
          "have non-negative values but found ",
          config_matrix(i, 0), " in row ", i);
    }
    if (0 > config_matrix(i, 1)) {
      return errors::InvalidArgument(
          "Second column of fraction_rows_per_thread_config "
          "should "
          "have non-negative values but found ",
          config_matrix(i, 1), " in row ", i);
    }
    if (!(0 < config_matrix(i, 2) && config_matrix(i, 2) <= 1)) {
      return errors::InvalidArgument(
          "Last column of fraction_rows_per_thread_config should "
          "have values in the range (0, 1] but found ",
          config_matrix(i, 2), " in row ", i);
    }
  }
  return Status::OK();
}

template <typename MatrixType, typename K>
MatrixType FindConfigValueForKey(
    const typename TTypes<MatrixType>::ConstMatrix& config_mat,
    const std::pair<K, K>& key) {
  const int last_row_index = config_mat.dimension(0) - 1;
  for (int i = 0; i < last_row_index; ++i) {
    if (key.first >= config_mat(i, 0) && key.second >= config_mat(i, 1)) {
      return config_mat(i, 2);
    }
  }
  return config_mat(last_row_index, 2);
}

Status ValidateScalarQuantityShardingConfig(const Tensor& config) {
  if (TensorShapeUtils::IsScalar(config.shape())) {
    const float scalar_config = config.template scalar<float>()();
    if (0 < scalar_config && scalar_config <= 1.0) {
      return Status::OK();
    }
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat("Expected config to be in range (0, 1] but instead found ",
                     scalar_config));
  }
  if (!TensorShapeUtils::IsMatrix(config.shape())) {
    return Status(error::INVALID_ARGUMENT,
                  absl::StrCat("Expected config to be either scalar or matrix "
                               "but instead found tensor of rank ",
                               config.dims()));
  }
  if (config.dim_size(1) != 2) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrCat(
            "Expected config matrix to have dim(1) = 2 but instead found ",
            config.dim_size(1)));
  }

  auto config_matrix = config.matrix<float>();
  for (int i = 0; i < config.dim_size(0); ++i) {
    if (0 > config_matrix(i, 0)) {
      return errors::InvalidArgument(
          "First column of fraction_rows_per_thread_config "
          "should "
          "have non-negative values but found ",
          config_matrix(i, 0), " in row ", i);
    }
    if (!(0 < config_matrix(i, 1) && config_matrix(i, 1) <= 1)) {
      return errors::InvalidArgument(
          "Last column of fraction_rows_per_thread_config should "
          "have values in the range (0, 1] but found ",
          config_matrix(i, 1), " in row ", i);
    }
  }
  return Status::OK();
}

template <typename MatrixType, typename K>
MatrixType FindConfigValueForKey(
    const typename TTypes<MatrixType>::ConstMatrix& config_mat, const K key) {
  const int last_row_index = config_mat.dimension(0) - 1;
  for (int i = 0; i < last_row_index; ++i) {
    if (key >= config_mat(i, 0)) {
      return config_mat(i, 1);
    }
  }
  return config_mat(last_row_index, 1);
}

template <typename Tindices>
Tindices GetLinearBucket(const Tindices value, const Tindices bucket_size) {
  const Tindices next_multiple_of_bucket_size =
      (value + bucket_size - 1) / bucket_size * bucket_size;
  return next_multiple_of_bucket_size - (bucket_size - 1);
}

template <typename Tindices>
Tindices GetPowerBucket(const Tindices value, const Tindices bucket_size) {
  if (bucket_size == 1) {
    return 1;
  }
  return std::pow(bucket_size, std::floor(std::log(bucket_size * (value - 1)) /
                                          std::log(bucket_size)) -
                                   1) +
         1;
}

#define REGISTER_SPARSE_UTIL_FUNCTIONS(TypeIndex)                           \
  template float FindConfigValueForKey<float, TypeIndex>(                   \
      const TTypes<float>::ConstMatrix& config_mat,                         \
      const std::pair<TypeIndex, TypeIndex>& key);                          \
  template float FindConfigValueForKey<float, TypeIndex>(                   \
      const TTypes<float>::ConstMatrix& config_mat, const TypeIndex key);   \
  template int64 FindConfigValueForKey<int64, TypeIndex>(                   \
      const TTypes<int64>::ConstMatrix& config_mat, const TypeIndex key);

REGISTER_SPARSE_UTIL_FUNCTIONS(int32);
REGISTER_SPARSE_UTIL_FUNCTIONS(int64);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint8);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint16);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint32);
REGISTER_SPARSE_UTIL_FUNCTIONS(uint64);

template int32 GetLinearBucket(const int32 value, const int32 bucket_size);

template int64 GetLinearBucket(const int64 value, const int64 bucket_size);

template int32 GetPowerBucket(const int32 value, const int32 bucket_size);

template int64 GetPowerBucket(const int64 value, const int64 bucket_size);

}  // namespace tensor_flag_utils
}  // namespace tensorflow
