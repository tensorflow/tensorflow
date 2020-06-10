/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ACTIVATIONS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ACTIVATIONS_H_

#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/internal/common.h"


namespace tflite {
namespace reference_ops {

inline std::int32_t RoundingDivideByPOT(std::int32_t numerator, int exponent) {
  std::int32_t sign = numerator >= 0 ? 1 : -1;
  std::int32_t abs_numerator = std::abs(numerator);
  std::int32_t mask = (1LL << exponent) - 1;
  std::int32_t remainder = abs_numerator & mask;
  std::int32_t threshold = mask >> 1;
  std::int32_t abs_result =
      (abs_numerator >> exponent) + (remainder > threshold ? 1 : 0);
  return sign * abs_result;
}

template <typename T>
inline void HardSwish(const RuntimeShape& input_shape, const T* input_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("ReferenceHardSwish/Float");
  auto matching_size = MatchingFlatSize(input_shape, output_shape);
  const T* in_end = input_data + matching_size;
  for (; input_data < in_end; input_data++, output_data++) {
    const float in = *input_data;
    *output_data =
        in * std::min(static_cast<T>(6), std::max(static_cast<T>(0), in + 3)) /
        6;
  }
}

}  // namespace reference_ops
}  // namespace tflite


#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CONV_H_
