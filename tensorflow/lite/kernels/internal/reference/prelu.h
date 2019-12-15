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

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Broadcast prelu to output_shape for quantized uint8 data.
inline void BroadcastPrelu4DSlow(const PreluParams& params,
                                 const RuntimeShape& input_shape,
                                 const uint8* input_data,
                                 const RuntimeShape& alpha_shape,
                                 const uint8* alpha_data,
                                 const RuntimeShape& output_shape,
                                 uint8* output_data) {
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
          const int32 input_value =
              params.input_offset + input_data[input_index];
          if (input_value >= 0) {
            output_data[output_index] = input_data[input_index];
          } else {
            auto alpha_index = SubscriptToIndex(desc2, b, y, x, c);
            const int32 alpha_value =
                params.alpha_offset + alpha_data[alpha_index];
            const int32 unclamped_output =
                params.output_offset +
                MultiplyByQuantizedMultiplierSmallerThanOneExp(
                    input_value * alpha_value, params.output_multiplier,
                    params.output_shift);
            const int32 quantized_min = std::numeric_limits<uint8_t>::min();
            const int32 quantized_max = std::numeric_limits<uint8_t>::max();
            const int32 clamped_output = std::min(
                quantized_max, std::max(quantized_min, unclamped_output));
            output_data[output_index] = static_cast<uint8>(clamped_output);
          }
        }
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_PRELU_H_
