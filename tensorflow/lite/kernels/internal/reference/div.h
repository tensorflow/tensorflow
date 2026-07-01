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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DIV_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DIV_H_

#include <algorithm>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/broadcast_loop.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void DivCheckArithmeticParams(const ArithmeticParams& params) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  constexpr int32_t max_value =
      (static_cast<int32_t>(std::numeric_limits<T>::max()) + 1);
  TFLITE_DCHECK_GE(params.input1_offset, -max_value);
  TFLITE_DCHECK_LE(params.input1_offset, max_value);
  TFLITE_DCHECK_GE(params.input2_offset, -max_value);
  TFLITE_DCHECK_LE(params.input2_offset, max_value);
  TFLITE_DCHECK_GE(params.output_offset, -max_value);
  TFLITE_DCHECK_LE(params.output_offset, max_value);
}

// Element-wise div that can often be used for inner loop of broadcast Div as
// well as the non-broadcast Div.
template <typename T>
inline void DivElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
  DivCheckArithmeticParams<T>(params);

  for (int i = 0; i < size; ++i) {
    int32_t input1_val = params.input1_offset + input1_data[i];
    int32_t input2_val = params.input2_offset + input2_data[i];
    TFLITE_DCHECK_NE(input2_val, 0);
    if (input2_val < 0) {
      // Invert signs to avoid a negative input2_val as input2_inv needs to be
      // positive to be used as multiplier of MultiplyByQuantizedMultiplier.
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;
    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = params.output_shift - recip_shift - headroom;
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            unscaled_quotient, params.output_multiplier, total_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int16_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  DivElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T, int N = 5>
inline void BroadcastDivSlowQuantized(
    const ArithmeticParams& params, const RuntimeShape& unextended_input1_shape,
    const T* input1_data, const RuntimeShape& unextended_input2_shape,
    const T* input2_data, const RuntimeShape& unextended_output_shape,
    T* output_data) {
  DivCheckArithmeticParams<T>(params);

  auto op = [&params](T a, T b) {
    int32_t input1_val = params.input1_offset + a;
    int32_t input2_val = params.input2_offset + b;
    TFLITE_DCHECK_NE(input2_val, 0);
    if (input2_val < 0) {
      input1_val = -input1_val;
      input2_val = -input2_val;
    }
    int recip_shift;
    const int32_t input2_inv = GetReciprocal(input2_val, 31, &recip_shift);
    const int headroom = CountLeadingSignBits(input1_val);
    const int32_t unscaled_quotient =
        MultiplyByQuantizedMultiplierGreaterThanOne(input1_val, input2_inv,
                                                    headroom);
    const int total_shift = params.output_shift - recip_shift - headroom;
    const int32_t unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            unscaled_quotient, params.output_multiplier, total_shift);
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, unclamped_result));
    return static_cast<T>(clamped_output);
  };

  BroadcastBinaryOpSimple(unextended_input1_shape, input1_data,
                          unextended_input2_shape, input2_data,
                          unextended_output_shape, output_data, op);
}

template <int N = 5>
inline void BroadcastDivSlow(const ArithmeticParams& params,
                             const RuntimeShape& unextended_input1_shape,
                             const uint8_t* input1_data,
                             const RuntimeShape& unextended_input2_shape,
                             const uint8_t* input2_data,
                             const RuntimeShape& unextended_output_shape,
                             uint8_t* output_data) {
  BroadcastDivSlowQuantized<uint8_t, N>(
      params, unextended_input1_shape, input1_data, unextended_input2_shape,
      input2_data, unextended_output_shape, output_data);
}

template <int N = 5>
inline void BroadcastDivSlow(const ArithmeticParams& params,
                             const RuntimeShape& unextended_input1_shape,
                             const int8_t* input1_data,
                             const RuntimeShape& unextended_input2_shape,
                             const int8_t* input2_data,
                             const RuntimeShape& unextended_output_shape,
                             int8_t* output_data) {
  BroadcastDivSlowQuantized<int8_t, N>(
      params, unextended_input1_shape, input1_data, unextended_input2_shape,
      input2_data, unextended_output_shape, output_data);
}

template <int N = 5>
inline void BroadcastDivSlow(const ArithmeticParams& params,
                             const RuntimeShape& unextended_input1_shape,
                             const int16_t* input1_data,
                             const RuntimeShape& unextended_input2_shape,
                             const int16_t* input2_data,
                             const RuntimeShape& unextended_output_shape,
                             int16_t* output_data) {
  BroadcastDivSlowQuantized<int16_t, N>(
      params, unextended_input1_shape, input1_data, unextended_input2_shape,
      input2_data, unextended_output_shape, output_data);
}

template <typename T, int N = 5>
void BroadcastDivSlow(const ArithmeticParams& params,
                      const RuntimeShape& unextended_input1_shape,
                      const T* input1_data,
                      const RuntimeShape& unextended_input2_shape,
                      const T* input2_data,
                      const RuntimeShape& unextended_output_shape,
                      T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  auto op = [output_activation_min, output_activation_max](T a, T b) {
    return ActivationFunctionWithMinMax(a / b, output_activation_min,
                                        output_activation_max);
  };

  BroadcastBinaryOpSimple(unextended_input1_shape, input1_data,
                          unextended_input2_shape, input2_data,
                          unextended_output_shape, output_data, op);
}

template <typename T>
inline void Div(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] / input2_data[i], output_activation_min,
        output_activation_max);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DIV_H_
