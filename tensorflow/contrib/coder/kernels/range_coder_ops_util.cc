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

#include "tensorflow/contrib/coder/kernels/range_coder_ops_util.h"

#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

using tensorflow::errors::InvalidArgument;

namespace tensorflow {
Status MergeAxes(const TensorShape& broadcast_shape,
                 const TensorShape& storage_shape,
                 std::vector<int64>* merged_broadcast_shape_pointer,
                 std::vector<int64>* merged_storage_shape_pointer) {
  CHECK_EQ(storage_shape.dims(), broadcast_shape.dims() + 1);

  std::vector<int64>& merged_broadcast_shape = *merged_broadcast_shape_pointer;
  std::vector<int64>& merged_storage_shape = *merged_storage_shape_pointer;

  // The shapes are simplified so that the conversions between linear index
  // and coordinates takes less CPU cycles. Two adjacent dimensions are
  // merged if they both are broadcasting dimensions or if they both are
  // non-broadcasting dimensions.
  merged_broadcast_shape.resize(1);
  merged_broadcast_shape[0] = 1;
  merged_storage_shape.resize(1);
  merged_storage_shape[0] = 1;

  for (int i = 0, j = 0; j < broadcast_shape.dims(); ++j) {
    if (TF_PREDICT_FALSE(
            (broadcast_shape.dim_size(j) != storage_shape.dim_size(j)) &&
            (storage_shape.dim_size(j) != 1))) {
      return InvalidArgument("Cannot broadcast shape ",
                             storage_shape.DebugString(), " to ",
                             broadcast_shape.DebugString());
    }

    const bool was_broadcasting = (merged_storage_shape[i] == 1);
    const bool is_broadcasting = (storage_shape.dim_size(j) == 1);

    // Merge two adjacent axes if they both are broadcasting or both are
    // non-broadcasting axes. The second and the third conditions in the if
    // clause below are when the previously merged axis or the next j-th axis
    // may be interpreted as either a broadcasting or a non-broadcasting axis.
    const bool merge = (was_broadcasting == is_broadcasting) ||
                       (broadcast_shape.dim_size(j) <= 1) ||
                       (merged_broadcast_shape[i] <= 1);

    if (merge) {
      merged_broadcast_shape[i] *= broadcast_shape.dim_size(j);
      merged_storage_shape[i] *= storage_shape.dim_size(j);
    } else {
      // Move to the next axis.
      merged_broadcast_shape.push_back(broadcast_shape.dim_size(j));
      merged_storage_shape.push_back(storage_shape.dim_size(j));
      ++i;
    }
  }

  int64 storage_stride = 1;
  for (int i = broadcast_shape.dims(); i < storage_shape.dims(); ++i) {
    storage_stride *= storage_shape.dim_size(i);
  }
  merged_storage_shape.push_back(storage_stride);

  return Status::OK();
}
}  // namespace tensorflow
