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

#include <algorithm>

#include "fixedpoint/fixedpoint.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_loop.h"

namespace tflite {
namespace reference_integer_ops {

// Maximum dimension supported by the broadcast mul operation.
constexpr int kMaxMulBroadcastDim = 6;

template <typename InputType, typename OutputType>
void MulElementwise(int size, const ArithmeticParams& params,
                    const InputType* input1_data, const InputType* input2_data,
                    OutputType* output_data) {
  for (int i = 0; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<OutputType>(clamped_output);
  }
}

template <typename T>
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  ruy::profiler::ScopeLabel label("Mul/8bit");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

// Mul with 16 bit inputs and int8_t outputs.
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  ruy::profiler::ScopeLabel label("Mul/Int16Int8");
  int32_t output_offset = params.output_offset;
  int32_t output_activation_min = params.quantized_activation_min;
  int32_t output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 unclamped_result =
        F0::FromRaw(input1_data[i]) * F0::FromRaw(input2_data[i]);
    int16_t rescaled_result =
        gemmlowp::RoundingDivideByPOT(unclamped_result.raw(), 8);
    int16_t clamped_result = std::min<int16_t>(
        output_activation_max - output_offset, rescaled_result);
    clamped_result = std::max<int16_t>(output_activation_min - output_offset,
                                       clamped_result);
    output_data[i] = output_offset + clamped_result;
  }
}

template <typename T>
inline void BroadcastMul6DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastMul6DSlow");
  auto op = [&params](T a, T b) {
    const int32_t input1_val = params.input1_offset + a;
    const int32_t input2_val = params.input2_offset + b;
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input1_val * input2_val,
                                      params.output_multiplier,
                                      params.output_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    return static_cast<T>(clamped_output);
  };
  reference_ops::BroadcastBinaryOpSimple(input1_shape, input1_data,
                                         input2_shape, input2_data,
                                         output_shape, output_data, op);
}

template <typename T>
inline void BroadcastMul4DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  BroadcastMul6DSlow(params, input1_shape, input1_data, input2_shape,
                     input2_data, output_shape, output_data);
}

}  // namespace reference_integer_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_MUL_H_
