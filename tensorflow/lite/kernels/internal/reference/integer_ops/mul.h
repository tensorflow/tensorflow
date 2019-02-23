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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_

#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_integer_ops {

inline void MulElementwise(int size, const ArithmeticParams& params,
                           const int8_t* input1_data, const int8_t* input2_data,
                           int8_t* output_data) {
  for (int i = 0; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(input1_val * input2_val,
                                                       params.output_multiplier,
                                                       params.output_shift);
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  gemmlowp::ScopedProfilingLabel label("Mul/8bit");
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

// Mul with 16 bit inputs and int8_t outputs.
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16* input1_data,
                const RuntimeShape& input2_shape, const int16* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("Mul/Int16Int8");
  int32 output_offset = params.output_offset;
  int32 output_activation_min = params.quantized_activation_min;
  int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16 rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16 clamped_result =
        std::min<int16>(output_activation_max - output_offset, rescaled_result);
    clamped_result =
        std::max<int16>(output_activation_min - output_offset, clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

inline void BroadcastMul4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const int8_t* input1_data,
                               const RuntimeShape& input2_shape,
                               const int8_t* input2_data,
                               const RuntimeShape& output_shape,
                               int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastMul4DSlow/8bit");

  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  // The input shapes are extended as part of NdArrayDesc initialization.
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          const int32 input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          const int32 input2_val =
              params.input2_offset +
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          const int32 unclamped_result =
              params.output_offset +
              MultiplyByQuantizedMultiplierSmallerThanOneExp(
                  input1_val * input2_val, params.output_multiplier,
                  params.output_shift);
          const int32 clamped_output = std::min(
              params.quantized_activation_max,
              std::max(params.quantized_activation_min, unclamped_result));
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              static_cast<int8_t>(clamped_output);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_
