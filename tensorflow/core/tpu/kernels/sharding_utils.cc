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

#include "tensorflow/core/tpu/kernels/sharding_utils.h"

#include <cstdint>
#include <functional>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/status.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep
#include "tsl/platform/macros.h"

namespace tensorflow {
namespace sharding_internal {
absl::Status ValidateShapesForSlice(absl::string_view input_name,
                                    const Tensor* input,
                                    const std::vector<int32_t>& num_splits,
                                    const std::vector<int32_t>& paddings) {
  const auto& ishape = input->shape();

  absl::Status s;

  const int rank = ishape.dims();
  const auto& input_shape = ishape.dim_sizes();
  if (rank <= 0 || rank > 8) {
    s = absl::InvalidArgumentError(absl::StrCat(
        input_name, " must have rank in range (0, 8], but got ", rank, "."));
  } else if (rank != num_splits.size()) {
    s = absl::InvalidArgumentError(absl::StrCat(
        input_name, " rank must be the same as 'num_splits' length ",
        num_splits.size(), ", but got rank ", rank, "."));
  } else {
    for (int dim = 0; dim < rank; ++dim) {
      const auto input_shape_dim = input_shape[dim];
      const auto paddings_dim = paddings[dim];
      const auto num_splits_dim = num_splits[dim];
      if ((input_shape_dim + paddings_dim) % num_splits_dim != 0) {
        s = absl::InvalidArgumentError(absl::StrCat(
            input_name, " shape dimension ", dim, " (", input_shape_dim,
            ") with padding ", paddings_dim,
            " must be evenly divisible by 'num_splits' ", num_splits_dim, "."));
        break;
      }
    }
  }
  return s;
}

}  // namespace sharding_internal

template <>
Eigen::DSizes<Eigen::DenseIndex, 1> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 1>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 1> subscript;
  subscript[0] = index * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 2> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 2>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 2> subscript;
  subscript[1] = (index % num_partitions[1]) * slice_shape[1];
  subscript[0] = (index / num_partitions[1]) * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 3> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 3>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 3> subscript;
  subscript[2] = (index % num_partitions[2]) * slice_shape[2];
  subscript[1] =
      ((index / num_partitions[2]) % num_partitions[1]) * slice_shape[1];
  subscript[0] =
      (index / (num_partitions[2] * num_partitions[1])) * slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 4> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 4>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 4> subscript;
  subscript[3] = (index % num_partitions[3]) * slice_shape[3];
  subscript[2] =
      ((index / num_partitions[3]) % num_partitions[2]) * slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[3] * num_partitions[2])) % num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[3] * num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 5> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 5>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 5> subscript;
  subscript[4] = (index % num_partitions[4]) * slice_shape[4];
  subscript[3] =
      ((index / num_partitions[4]) % num_partitions[3]) * slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[4] * num_partitions[3])) % num_partitions[2]) *
      slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[4] * num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] = (index / (num_partitions[4] * num_partitions[3] *
                           num_partitions[2] * num_partitions[1])) *
                 slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 6> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 6>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 6> subscript;
  subscript[5] = (index % num_partitions[5]) * slice_shape[5];
  subscript[4] =
      ((index / num_partitions[5]) % num_partitions[4]) * slice_shape[4];
  subscript[3] =
      ((index / (num_partitions[5] * num_partitions[4])) % num_partitions[3]) *
      slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[5] * num_partitions[4] * num_partitions[3])) %
       num_partitions[2]) *
      slice_shape[2];
  subscript[1] = ((index / (num_partitions[5] * num_partitions[4] *
                            num_partitions[3] * num_partitions[2])) %
                  num_partitions[1]) *
                 slice_shape[1];
  subscript[0] =
      (index / (num_partitions[5] * num_partitions[4] * num_partitions[3] *
                num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 7> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 7>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 7> subscript;
  subscript[6] = (index % num_partitions[6]) * slice_shape[6];
  subscript[5] =
      ((index / num_partitions[6]) % num_partitions[5]) * slice_shape[5];
  subscript[4] =
      ((index / (num_partitions[6] * num_partitions[5])) % num_partitions[4]) *
      slice_shape[4];
  subscript[3] =
      ((index / (num_partitions[6] * num_partitions[5] * num_partitions[4])) %
       num_partitions[3]) *
      slice_shape[3];
  subscript[2] = ((index / (num_partitions[6] * num_partitions[5] *
                            num_partitions[4] * num_partitions[3])) %
                  num_partitions[2]) *
                 slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[6] * num_partitions[5] * num_partitions[4] *
                 num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[6] * num_partitions[5] * num_partitions[4] *
                num_partitions[3] * num_partitions[2] * num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

template <>
Eigen::DSizes<Eigen::DenseIndex, 8> GetSliceIndices(
    absl::Span<const int32_t> num_partitions,
    const Eigen::DSizes<Eigen::DenseIndex, 8>& slice_shape, const int index) {
  Eigen::DSizes<Eigen::DenseIndex, 8> subscript;
  subscript[7] = (index % num_partitions[7]) * slice_shape[7];
  subscript[6] =
      ((index / num_partitions[7]) % num_partitions[6]) * slice_shape[6];
  subscript[5] =
      ((index / (num_partitions[7] * num_partitions[6])) % num_partitions[5]) *
      slice_shape[5];
  subscript[4] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5])) %
       num_partitions[4]) *
      slice_shape[4];
  subscript[3] = ((index / (num_partitions[7] * num_partitions[6] *
                            num_partitions[5] * num_partitions[4])) %
                  num_partitions[3]) *
                 slice_shape[3];
  subscript[2] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                 num_partitions[4] * num_partitions[3])) %
       num_partitions[2]) *
      slice_shape[2];
  subscript[1] =
      ((index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                 num_partitions[4] * num_partitions[3] * num_partitions[2])) %
       num_partitions[1]) *
      slice_shape[1];
  subscript[0] =
      (index / (num_partitions[7] * num_partitions[6] * num_partitions[5] *
                num_partitions[4] * num_partitions[3] * num_partitions[2] *
                num_partitions[1])) *
      slice_shape[0];
  return subscript;
}

}  // namespace tensorflow
