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
#ifndef TENSORFLOW_CORE_KERNELS_RAGGED_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_RAGGED_UTILS_H_

#include <cstdint>

#include "absl/status/status.h"
#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

// Utility functions for RaggedTensor

// Verifies that the splits are valid for ragged tensor
template <typename SPLIT_TYPE>
absl::Status RaggedTensorVerifySplits(const Tensor& ragged_splits,
                                      bool check_last_element,
                                      int64_t num_ragged_values) {
  auto flat_ragged_splits = ragged_splits.flat<SPLIT_TYPE>();

  if (ragged_splits.dims() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid ragged splits: ragged splits must be rank 1 but is rank ",
        ragged_splits.dims()));
  }

  if (ragged_splits.NumElements() < 1) {
    return absl::InvalidArgumentError(
        "Invalid ragged splits: ragged splits must have at least one splits, "
        "but is empty");
  }

  if (flat_ragged_splits(0) != static_cast<SPLIT_TYPE>(0)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid ragged splits: first element of ragged splits "
                     " must be 0 but is ",
                     flat_ragged_splits(0)));
  }

  SPLIT_TYPE last_split = 0;
  for (int j = 1; j < ragged_splits.dim_size(0); j++) {
    auto split = flat_ragged_splits(j);
    if (split < last_split) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid ragged splits: ragged splits must be "
                       "monotonically increasing, but ragged_splits[",
                       j, "]=", split, " is smaller than row_splits[", j - 1,
                       "]=", last_split));
    }
    last_split = split;
  }

  if (check_last_element & last_split != num_ragged_values) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Invalid ragged splits: last element of ragged splits must be ",
        "the number of ragged values(", num_ragged_values, ") but is ",
        last_split));
  }

  return absl::OkStatus();
}
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_RAGGED_UTILS_H_
