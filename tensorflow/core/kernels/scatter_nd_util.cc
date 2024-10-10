/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/scatter_nd_util.h"

#include "tensorflow/core/framework/tensor_shape.h"

namespace tensorflow {

absl::Status ValidateScatterNdUpdateShape(const TensorShape& params_shape,
                                          const TensorShape& indices_shape,
                                          const TensorShape& updates_shape) {
  const int64_t slice_dim =
      (indices_shape.dims() > 1)
          ? indices_shape.dim_size(indices_shape.dims() - 1)
          : 1;
  const int64_t batch_dim =
      (indices_shape.dims() > 1) ? indices_shape.dims() - 1 : 1;

  auto shape_err_prefix = [&]() {
    return errors::InvalidArgument(
        "Dimensions [0,", batch_dim,
        ") of indices[shape=", indices_shape.DebugString(),
        "] must match dimensions [0,", batch_dim,
        ") of updates[shape=", updates_shape.DebugString(), "]");
  };
  auto shape_err_suffix = [&]() {
    return errors::InvalidArgument(
        "Dimensions [", slice_dim, ",", params_shape.dims(),
        ") of input[shape=", params_shape.DebugString(),
        "] must match dimensions [", slice_dim, ",", updates_shape.dims(),
        ") of updates[shape=", updates_shape.DebugString(), "]");
  };

  if (updates_shape.dims() < batch_dim) return shape_err_prefix();
  if (params_shape.dims() < slice_dim + (updates_shape.dims() - batch_dim)) {
    return shape_err_suffix();
  }
  if (updates_shape.dims() != batch_dim + params_shape.dims() - slice_dim) {
    return shape_err_suffix();
  }
  for (int d = 0; d < batch_dim; ++d) {
    if (updates_shape.dim_size(d) != indices_shape.dim_size(d))
      return shape_err_prefix();
  }
  for (int d = 0; d < updates_shape.dims() - batch_dim; ++d) {
    if (updates_shape.dim_size(d + batch_dim) !=
        params_shape.dim_size(d + slice_dim)) {
      return shape_err_suffix();
    }
  }
  return absl::OkStatus();
}

}  // namespace tensorflow
