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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_

#include <cmath>
#include <functional>

#include "third_party/eigen3/Eigen/Core"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/constants.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

namespace gelu_internal {

constexpr float kSqrt2dPi = M_2_SQRTPI * M_SQRT1_2;  // sqrt( 2 / pi )

}  // namespace gelu_internal

// Plain implementations for GELU. Used for populating lookup table.
inline float GeluTransform(float in) {
  // 0.5 * x * ( 1 + erf( x / sqrt( 2 ) ) )
  return 0.5f * in * (1.f + std::erf(in * M_SQRT1_2));
}

inline float GeluTransformApproximate(float in) {
  // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )
  return 0.5f * in *
         (1.f + std::tanh(gelu_internal::kSqrt2dPi *
                          // Note: Avoid std::pow for integer exponents
                          // as it leads to much slower performance.
                          (in + 0.044715f * in * in * in)));
}

template <typename T>
inline void Gelu(const RuntimeShape& input_shape, const T* input_data,
                 bool approximate, const RuntimeShape& output_shape,
                 T* output_data) {
  using VectorType = Eigen::VectorX<T>;
  auto input_map = VectorType::Map(input_data, input_shape.FlatSize());
  auto output_map = VectorType::Map(output_data, output_shape.FlatSize());

  if (approximate) {
    // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )
    output_map.array() = static_cast<T>(0.5) * input_map.array() *
                         (static_cast<T>(1) +
                          (static_cast<T>(gelu_internal::kSqrt2dPi) *
                           (input_map.array() + static_cast<T>(0.044715) *
                                                    input_map.array().cube()))
                              .tanh());
  } else {
    // 0.5 * x * ( 1 + erf( x / sqrt( 2 ) ) )
    output_map.array() =
        static_cast<T>(0.5) * input_map.array() *
        (static_cast<T>(1) +
         (input_map.array() * static_cast<T>(M_SQRT1_2)).erf());
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_
