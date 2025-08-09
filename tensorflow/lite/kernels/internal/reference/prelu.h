/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PRELU_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PRELU_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Broadcast prelu to output_shape for quantized uint8_t/int8_t data.
template <typename T, typename U>
inline void BroadcastPrelu4DSlow(
    const PreluParams& params, const RuntimeShape& input_shape,
    const T* input_data, const RuntimeShape& alpha_shape, const U* alpha_data,
    const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_LE(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(alpha_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), 4);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input_shape, alpha_shape, &desc1, &desc2);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          int output_index = Offset(extended_output_shape, b, y, x, c);
          int input_index = SubscriptToIndex(desc1, b, y, x, c);
          const int32_t input_value =
              params.input_offset + input_data[input_index];
          int32_t output_value;
          if (input_value >= 0) {
            output_value = MultiplyByQuantizedMultiplier(
                input_value, params.output_multiplier_1, params.output_shift_1);
          } else {
            auto alpha_index = SubscriptToIndex(desc2, b, y, x, c);
            const int32_t alpha_value =
                params.alpha_offset + alpha_data[alpha_index];

            output_value = MultiplyByQuantizedMultiplier(
                input_value * alpha_value, params.output_multiplier_2,
                params.output_shift_2);
          }
          output_value += params.output_offset;

          const int32_t quantized_min = std::numeric_limits<T>::min();
          const int32_t quantized_max = std::numeric_limits<T>::max();
          const int32_t clamped_output =
              std::min(quantized_max, std::max(quantized_min, output_value));
          output_data[output_index] = static_cast<T>(clamped_output);
        }
      }
    }
  }
}

template <typename T, typename U>
inline void Prelu(const PreluParams& params, const RuntimeShape& input_shape,
                  const T* input_data, const RuntimeShape& alpha_shape,
                  const U* alpha_data, const RuntimeShape& output_shape,
                  T* output_data) {
  const int32_t quantized_min = std::numeric_limits<T>::min();
  const int32_t quantized_max = std::numeric_limits<T>::max();

  const int flat_size =
      MatchingElementsSize(input_shape, alpha_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    const int32_t input_value = params.input_offset + input_data[i];
    int32_t output_value;
    if (input_value >= 0) {
      output_value = MultiplyByQuantizedMultiplier(
          input_value, params.output_multiplier_1, params.output_shift_1);
    } else {
      const int32_t alpha_value = params.alpha_offset + alpha_data[i];

      output_value = MultiplyByQuantizedMultiplier(input_value * alpha_value,
                                                   params.output_multiplier_2,
                                                   params.output_shift_2);
    }
    output_value += params.output_offset;

    const int32_t clamped_output =
        std::min(quantized_max, std::max(quantized_min, output_value));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PRELU_H_
