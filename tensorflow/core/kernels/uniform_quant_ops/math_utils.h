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
#ifndef TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_MATH_UTILS_H_
#define TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_MATH_UTILS_H_

#include <cmath>

#include "tensorflow/core/framework/tensor.h"

namespace tensorflow {

// Quantize input_val using given inv_scale and zero_point, using the formula:
// quantized_val = input_val * inv_scale + zero_point
//
// The caller is reponsible for the validity of the inv_scale (Avoid precision
// loss from taking inverse, and ensure that inv_scale is a finite number.)
template <typename Tin, typename Tout>
void AffineQuantize(const Tin& input_tensor, float inv_scale,
                    int32_t zero_point, int32_t quantization_min_val,
                    int32_t quantization_max_val, Tout& quantized_tensor) {
  quantized_tensor = ((input_tensor.template cast<float>() * inv_scale + 0.5f)
                          .floor()
                          .template cast<int32_t>() +
                      zero_point)
                         .cwiseMin(quantization_max_val)
                         .cwiseMax(quantization_min_val)
                         .template cast<typename Tout::Scalar>();
}

// Given a portion of input float tensor, quantizes the data and writes output
// to the corresponding portion in quantized_tensor. The quantization scale and
// zero_point is calculated using the input data min and max.
// This function is used for dynamic range quantization in hybrid (float x qint)
// kernels.
//
// This function behavior aligns with TFLite AsymmetricQuantize() to achieve
// feature parity with TFLite which is required since supporting mobile
// executions is the one of the major use cases. The behavior is same except for
// following difference:
// TFLite AsymmetricQuantize() uses
// round(input / scale + zero_point),
// while AffineQuantize() uses
// floor(input_val * (1./scale) + 0.5) + zero_point
void AsymmetricQuantize(const Tensor& tensor, int apply_offset, int apply_size,
                        int32_t quantization_min_val,
                        int32_t quantization_max_val, float& scale,
                        int32_t& zero_point, Tensor& quantized_tensor);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_UNIFORM_QUANT_OPS_MATH_UTILS_H_
