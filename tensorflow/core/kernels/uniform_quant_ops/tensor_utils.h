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
#ifndef TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_TENSOR_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_TENSOR_UTILS_H_

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

// Returns if all elements in given tensors are positive.
template <typename T>
bool AllElementsPositive(const Tensor& tensor) {
  Eigen::Tensor<bool, 0, Eigen::RowMajor> positive =
      (tensor.flat<T>() > 0).all();
  return positive();
}

// Given data tensor's shape and quantization params, returns if the shapes are
// valid.
Status QuantizationAxisAndShapeValid(const TensorShape& data_shape,
                                     const TensorShape& scales_shape,
                                     const TensorShape& zero_points_shape,
                                     int quantization_axis);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_TENSOR_UTILS_H_
