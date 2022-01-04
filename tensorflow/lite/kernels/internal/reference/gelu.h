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

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/constants.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_ops {

constexpr float kSqrt2dPi = M_2_SQRTPI * M_SQRT1_2;  // sqrt( 2 / pi )

template <typename T>
inline void Gelu(const RuntimeShape& input_shape, const T* input_data,
                 bool approximate, const RuntimeShape& output_shape,
                 T* output_data) {
  auto matching_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < matching_size; i++) {
    const T in = input_data[i];
    if (approximate) {
      // 0.5 * x * ( 1 + tanh( sqrt( 2 / pi ) * ( x + 0.044715 * x^3 ) ) )
      output_data[i] =
          static_cast<T>(0.5) * in *
          (static_cast<T>(1) +
           std::tanh(static_cast<T>(kSqrt2dPi) *
                     // Note: Avoid std::pow for integer exponents
                     // as it leads to much slower performance.
                     (in + static_cast<T>(0.044715) * in * in * in)));
    } else {
      // 0.5 * x * ( 1 + erf( x / sqrt( 2 ) ) )
      output_data[i] =
          static_cast<T>(0.5) * in *
          (static_cast<T>(1) + std::erf(in * static_cast<T>(M_SQRT1_2)));
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_GELU_H_
