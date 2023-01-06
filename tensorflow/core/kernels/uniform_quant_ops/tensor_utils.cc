/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/kernels/uniform_quant_ops/tensor_utils.h"

namespace tensorflow {

using tensorflow::errors::InvalidArgument;

Status QuantizationAxisAndShapeValid(const TensorShape& data_shape,
                                     const TensorShape& scales_shape,
                                     const TensorShape& zero_points_shape,
                                     int quantization_axis) {
  if (!scales_shape.IsSameSize(zero_points_shape)) {
    return InvalidArgument(
        "scales and zero_points shape must be same, but given scales shape ",
        scales_shape.DebugString(), " and zero_points shape ",
        zero_points_shape.DebugString());
  }
  if (quantization_axis < -1 || quantization_axis >= data_shape.dims()) {
    return InvalidArgument(
        "quantization_axis must be -1 or in range [0, input.rank), but given ",
        quantization_axis);
  }

  if (quantization_axis == -1) {
    if (scales_shape.dims() != 0) {
      return InvalidArgument(
          "If quantization_axis is -1, scales and zero_points must be scalar "
          "tensors, but given scales shape ",
          scales_shape.DebugString(), " and zero_points shape ",
          zero_points_shape.DebugString());
    }
  } else {
    if (!(scales_shape.dims() == 1 &&
          scales_shape.dim_size(0) == data_shape.dim_size(quantization_axis))) {
      return InvalidArgument(
          "If quantization_axis is not -1, scales and zero_points must be a "
          "tensor of rank 1 and the size must be equal to the "
          "input.dim_size(quantization_axis), but given quantization_axis ",
          quantization_axis, ", scales shape ", scales_shape.DebugString(),
          " and zero_points shape ", zero_points_shape.DebugString());
    }
  }
  return OkStatus();
}

TensorShape TransposedShape(const TensorShape& in_shape,
                            const gtl::ArraySlice<int32_t> perm) {
  TensorShape out_shape = in_shape;
  for (int i = 0; i < out_shape.dims(); ++i) {
    out_shape.set_dim(i, in_shape.dim_size(perm[i]));
  }
  return out_shape;
}

}  // namespace tensorflow
