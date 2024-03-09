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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {

namespace reference_ops {

template <typename T>
inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  T activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] + input2_data[i], activation_min, activation_max);
  }
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.

// This function is used for 8-bit as well as for 16-bit, but the accumulator
// is 32-bit for both cases. The overflow does not happen due to the
// choice of the shift (20 or 15, accordingly - see add.cc for more comments).
template <typename T>
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
  TFLITE_DCHECK_GT(params.input1_offset, -std::numeric_limits<T>::max());
  TFLITE_DCHECK_GT(params.input2_offset, -std::numeric_limits<T>::max());
  TFLITE_DCHECK_LT(params.input1_offset, std::numeric_limits<T>::max());
  TFLITE_DCHECK_LT(params.input2_offset, std::numeric_limits<T>::max());

  for (int i = 0; i < size; ++i) {
    const int32_t input1_val = params.input1_offset + input1_data[i];
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

// Scalar-broadcast add that can be used for inner loop of more general
// broadcast add, so that, for example, scalar-broadcast with batch will still
// be fast.
inline void AddScalarBroadcast(int size, const ArithmeticParams& params,
                               uint8_t input1_data, const uint8_t* input2_data,
                               uint8_t* output_data) {
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

  const int32_t input1_val = params.input1_offset + input1_data;
  const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
  const int32_t scaled_input1_val =
      MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_input1_val, params.input1_multiplier, params.input1_shift);
  for (int i = 0; i < size; ++i) {
    const int32_t input2_val = params.input2_offset + input2_data[i];
    const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32_t scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void AddGeneralParamScale(const ArithmeticParams& params,
                                 const RuntimeShape& input1_shape,
                                 const int16_t* input1_data,
                                 const RuntimeShape& input2_shape,
                                 const int16_t* input2_data,
                                 const RuntimeShape& output_shape,
                                 int16_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  int max_value = std::numeric_limits<int16_t>::max();

  TFLITE_DCHECK_GT(params.input1_offset, -max_value);
  TFLITE_DCHECK_GT(params.input2_offset, -max_value);
  TFLITE_DCHECK_LT(params.input1_offset, max_value);
  TFLITE_DCHECK_LT(params.input2_offset, max_value);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int16_t* output_data,
                bool pot_scale = true) {
  if (!pot_scale) {
    AddGeneralParamScale(params, input1_shape, input1_data, input2_shape,
                         input2_data, output_shape, output_data);
    return;
  }

  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int input1_shift = params.input1_shift;
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  const int16_t output_activation_min = params.quantized_activation_min;
  const int16_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK(input1_shift == 0 || params.input2_shift == 0);
  TFLITE_DCHECK_LE(input1_shift, 0);
  TFLITE_DCHECK_LE(params.input2_shift, 0);
  const int16_t* not_shift_input =
      input1_shift == 0 ? input1_data : input2_data;
  const int16_t* shift_input = input1_shift == 0 ? input2_data : input1_data;
  const int input_right_shift =
      input1_shift == 0 ? -params.input2_shift : -input1_shift;

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;

    F0 input_ready_scaled = F0::FromRaw(not_shift_input[i]);
    F0 scaled_input = F0::FromRaw(
        gemmlowp::RoundingDivideByPOT(shift_input[i], input_right_shift));
    F0 result = gemmlowp::SaturatingAdd(scaled_input, input_ready_scaled);
    const int16_t raw_output = result.raw();
    const int16_t clamped_output = std::min(
        output_activation_max, std::max(output_activation_min, raw_output));
    output_data[i] = clamped_output;
  }
}

template <typename T>
inline void AddBroadcast(const T* input_data, const T* broadcast_data,
                         T* output_data, size_t size, T activation_min,
                         T activation_max) {
  for (size_t c = 0; c < size; ++c) {
    output_data[c] = ActivationFunctionWithMinMax<T>(
        input_data[c] + broadcast_data[0], activation_min, activation_max);
  }
}

template <>
inline void AddBroadcast<int32_t>(const int32_t* input_data,
                                  const int32_t* broadcast_data,
                                  int32_t* output_data, size_t size,
                                  int32_t activation_min,
                                  int32_t activation_max) {
  size_t c = 0;
#ifdef USE_NEON
  const int32x4_t vmax = vdupq_n_s32(activation_max);
  const int32x4_t vmin = vdupq_n_s32(activation_min);
  const int32x4_t vb = vdupq_n_s32(broadcast_data[0]);
  for (; c + 4 <= size; c += 4) {
    const int32x4_t va = vld1q_s32(&input_data[c]);
    int32x4_t vres = vaddq_s32(va, vb);
    vres = vmaxq_s32(vmin, vres);
    vres = vminq_s32(vmax, vres);
    vst1q_s32(&output_data[c], vres);
  }
#endif
  for (; c < size; ++c) {
    output_data[c] = ActivationFunctionWithMinMax<int32_t>(
        input_data[c] + broadcast_data[0], activation_min, activation_max);
  }
}

template <typename T>
void AddElementwise(const T* input1_data, const T* input2_data, T* output_data,
                    size_t size, T activation_min, T activation_max) {
  for (size_t c = 0; c < size; ++c) {
    output_data[c] = ActivationFunctionWithMinMax<T>(
        input1_data[c] + input2_data[c], activation_min, activation_max);
  }
}

template <>
inline void AddElementwise<int32_t>(const int32_t* input1_data,
                                    const int32_t* input2_data,
                                    int32_t* output_data, size_t size,
                                    int32_t activation_min,
                                    int32_t activation_max) {
  size_t c = 0;
#ifdef USE_NEON
  const int32x4_t vmax = vdupq_n_s32(activation_max);
  const int32x4_t vmin = vdupq_n_s32(activation_min);
  for (; c + 4 <= size; c += 4) {
    const int32x4_t va = vld1q_s32(&input1_data[c]);
    const int32x4_t vb = vld1q_s32(&input2_data[c]);
    int32x4_t vres = vaddq_s32(va, vb);
    vres = vmaxq_s32(vmin, vres);
    vres = vminq_s32(vmax, vres);
    vst1q_s32(&output_data[c], vres);
  }
#endif
  for (; c < size; ++c) {
    output_data[c] = ActivationFunctionWithMinMax<int32_t>(
        input1_data[c] + input2_data[c], activation_min, activation_max);
  }
}

template <typename T>
inline void BroadcastAddRecursiveDimensions(
    int dimension, size_t* input1_offset_p, size_t* input2_offset_p,
    size_t* output_offset, size_t* compressed_input1_stride,
    size_t* compressed_input2_stride, size_t* compressed_output_shape,
    T activation_min, T activation_max, const T* input1_data,
    const T* input2_data, T* output_data) {
  if (dimension > 0) {
    for (size_t c = 0; c < compressed_output_shape[dimension]; ++c) {
      size_t input1_offset_c = *input1_offset_p;
      size_t input2_offset_c = *input2_offset_p;
      BroadcastAddRecursiveDimensions(
          dimension - 1, &input1_offset_c, &input2_offset_c, output_offset,
          compressed_input1_stride, compressed_input2_stride,
          compressed_output_shape, activation_min, activation_max, input1_data,
          input2_data, output_data);
      *input1_offset_p += compressed_input1_stride[dimension];
      *input2_offset_p += compressed_input2_stride[dimension];
    }
  } else {
    TFLITE_DCHECK(dimension == 0);
    bool input1_is_broadcast = compressed_input1_stride[dimension] == 0;
    bool input2_is_broadcast = compressed_input2_stride[dimension] == 0;
    TFLITE_DCHECK(!(input1_is_broadcast && input2_is_broadcast));
    const T* input1_data_ptr = input1_data + *input1_offset_p;
    const T* input2_data_ptr = input2_data + *input2_offset_p;
    T* output_data_ptr = output_data + *output_offset;
    if (input1_is_broadcast) {
      // input1 is broadcast.
      AddBroadcast<T>(input2_data_ptr, input1_data_ptr, output_data_ptr,
                      compressed_output_shape[dimension], activation_min,
                      activation_max);
      *input2_offset_p += compressed_output_shape[dimension];
    } else if (input2_is_broadcast) {
      // input2 is broadcast.
      AddBroadcast<T>(input1_data_ptr, input2_data_ptr, output_data_ptr,
                      compressed_output_shape[dimension], activation_min,
                      activation_max);
      *input1_offset_p += compressed_output_shape[dimension];
    } else {
      // Add element-wise.
      AddElementwise<T>(input1_data_ptr, input2_data_ptr, output_data_ptr,
                        compressed_output_shape[dimension], activation_min,
                        activation_max);
      *input1_offset_p += compressed_output_shape[dimension];
      *input2_offset_p += compressed_output_shape[dimension];
    }
    *output_offset += compressed_output_shape[dimension];
  }
}

template <typename T,
          // For unquantized add for small integers, explicitly set to true.
          bool dummy = false>
inline typename std::enable_if<!is_small_integer<T>::value || dummy, void>::type
BroadcastAdd6DSlow(const ArithmeticParams& params,
                   const RuntimeShape& input1_shape, const T* input1_data,
                   const RuntimeShape& input2_shape, const T* input2_data,
                   const RuntimeShape& output_shape, T* output_data) {
  constexpr int kMaxBroadcastDim = 6;
  T activation_min, activation_max;
  GetActivationParams(params, &activation_min, &activation_max);

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  size_t compressed_input1_stride[kMaxBroadcastDim];
  size_t compressed_input2_stride[kMaxBroadcastDim];
  size_t compressed_output_shape[kMaxBroadcastDim];
  bool broadcastable_shape = ReduceDimensionsForBroadcast<kMaxBroadcastDim>(
      input1_shape, input2_shape, compressed_input1_stride,
      compressed_input2_stride, compressed_output_shape);
  // Skip broadcasting for degenerate shapes.
  if (!broadcastable_shape) {
    return;
  }

  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastAddRecursiveDimensions<T>(
      kMaxBroadcastDim - 1, &input1_offset, &input2_offset, &output_offset,
      compressed_input1_stride, compressed_input2_stride,
      compressed_output_shape, activation_min, activation_max, input1_data,
      input2_data, output_data);
}

// This function is used for 8-bit as well as for 16-bit, but the accumulator
// is 32-bit for both cases. The overflow does not happen due to the
// choice of the shift (20 or 15, accordingly - see add.cc for more comments).
template <typename T>
inline void BroadcastAddRecursiveDimensions(
    const ArithmeticParams& params, int dimension, size_t* input1_offset_p,
    size_t* input2_offset_p, size_t* output_offset,
    size_t* compressed_input1_stride, size_t* compressed_input2_stride,
    size_t* compressed_output_shape, const T* input1_data, const T* input2_data,
    T* output_data) {
  for (size_t c = 0; c < compressed_output_shape[dimension]; ++c) {
    if (dimension > 0) {
      size_t input1_offset_c = *input1_offset_p;
      size_t input2_offset_c = *input2_offset_p;
      BroadcastAddRecursiveDimensions(
          params, dimension - 1, &input1_offset_c, &input2_offset_c,
          output_offset, compressed_input1_stride, compressed_input2_stride,
          compressed_output_shape, input1_data, input2_data, output_data);
    } else {
      TFLITE_DCHECK(dimension == 0);
      const int32_t input1_val =
          params.input1_offset + input1_data[*input1_offset_p];
      const int32_t input2_val =
          params.input2_offset + input2_data[*input2_offset_p];
      const int32_t shifted_input1_val = input1_val * (1 << params.left_shift);
      const int32_t shifted_input2_val = input2_val * (1 << params.left_shift);
      const int32_t scaled_input1_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_input1_val, params.input1_multiplier,
              params.input1_shift);
      const int32_t scaled_input2_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_input2_val, params.input2_multiplier,
              params.input2_shift);
      const int32_t raw_sum = scaled_input1_val + scaled_input2_val;
      const int32_t raw_output =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              raw_sum, params.output_multiplier, params.output_shift) +
          params.output_offset;
      const int32_t clamped_output =
          std::min(params.quantized_activation_max,
                   std::max(params.quantized_activation_min, raw_output));
      output_data[*output_offset] = static_cast<T>(clamped_output);
      ++(*output_offset);
    }
    *input1_offset_p += compressed_input1_stride[dimension];
    *input2_offset_p += compressed_input2_stride[dimension];
  }
}

// This function is used for 8-bit as well as for 16-bit, but the accumulator
// is 32-bit for both cases. The overflow does not happen due to the
// choice of the shift (20 or 15, accordingly - see add.cc for more comments).
template <typename T>
inline typename std::enable_if<is_small_integer<T>::value, void>::type
BroadcastAdd6DSlow(const ArithmeticParams& params,
                   const RuntimeShape& input1_shape, const T* input1_data,
                   const RuntimeShape& input2_shape, const T* input2_data,
                   const RuntimeShape& output_shape, T* output_data) {
  constexpr int kMaxBroadcastDim = 6;

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest stride,
  // typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for the
  // best cache behavior.
  size_t compressed_input1_stride[kMaxBroadcastDim];
  size_t compressed_input2_stride[kMaxBroadcastDim];
  size_t compressed_output_shape[kMaxBroadcastDim];
  bool broadcastable_shape = ReduceDimensionsForBroadcast<kMaxBroadcastDim>(
      input1_shape, input2_shape, compressed_input1_stride,
      compressed_input2_stride, compressed_output_shape);
  // Skip broadcasting for degenerate shapes.
  if (!broadcastable_shape) {
    return;
  }

  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastAddRecursiveDimensions(
      params, kMaxBroadcastDim - 1, &input1_offset, &input2_offset,
      &output_offset, compressed_input1_stride, compressed_input2_stride,
      compressed_output_shape, input1_data, input2_data, output_data);
}

template <typename T>
inline void BroadcastAdd4DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  return BroadcastAdd6DSlow(params, input1_shape, input1_data, input2_shape,
                            input2_data, output_shape, output_data);
}

inline void BroadcastAddFivefold(const ArithmeticParams& unswitched_params,
                                 const RuntimeShape& unswitched_input1_shape,
                                 const uint8_t* unswitched_input1_data,
                                 const RuntimeShape& unswitched_input2_shape,
                                 const uint8_t* unswitched_input2_data,
                                 const RuntimeShape& output_shape,
                                 uint8_t* output_data) {
  ArithmeticParams switched_params = unswitched_params;
  switched_params.input1_offset = unswitched_params.input2_offset;
  switched_params.input1_multiplier = unswitched_params.input2_multiplier;
  switched_params.input1_shift = unswitched_params.input2_shift;
  switched_params.input2_offset = unswitched_params.input1_offset;
  switched_params.input2_multiplier = unswitched_params.input1_multiplier;
  switched_params.input2_shift = unswitched_params.input1_shift;

  const bool use_unswitched =
      unswitched_params.broadcast_category ==
      tflite::BroadcastableOpCategory::kFirstInputBroadcastsFast;

  const ArithmeticParams& params =
      use_unswitched ? unswitched_params : switched_params;
  const uint8_t* input1_data =
      use_unswitched ? unswitched_input1_data : unswitched_input2_data;
  const uint8_t* input2_data =
      use_unswitched ? unswitched_input2_data : unswitched_input1_data;

  // Fivefold nested loops. The second input resets its position for each
  // iteration of the second loop. The first input resets its position at the
  // beginning of the fourth loop. The innermost loop is an elementwise add of
  // sections of the arrays.
  uint8_t* output_data_ptr = output_data;
  const uint8_t* input1_data_ptr = input1_data;
  const uint8_t* input2_data_reset = input2_data;
  // In the fivefold pattern, y0, y2 and y4 are not broadcast, and so shared
  // between input shapes. y3 for input 1 is always broadcast, and so the
  // dimension there is 1, whereas optionally y1 might be broadcast for input 2.
  // Put another way,
  // input1.shape.FlatSize = y0 * y1 * y2 * y4,
  // input2.shape.FlatSize = y0 * y2 * y3 * y4.
  int y0 = params.broadcast_shape[0];
  int y1 = params.broadcast_shape[1];
  int y2 = params.broadcast_shape[2];
  int y3 = params.broadcast_shape[3];
  int y4 = params.broadcast_shape[4];
  if (y4 > 1) {
    // General fivefold pattern, with y4 > 1 so there is a non-broadcast inner
    // dimension.
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8_t* input2_data_ptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          for (int i3 = 0; i3 < y3; ++i3) {
            AddElementwise(y4, params, input1_data_ptr, input2_data_ptr,
                           output_data_ptr);
            input2_data_ptr += y4;
            output_data_ptr += y4;
          }
          // We have broadcast y4 of input1 data y3 times, and now move on.
          input1_data_ptr += y4;
        }
      }
      // We have broadcast y2*y3*y4 of input2 data y1 times, and now move on.
      input2_data_reset = input2_data_ptr;
    }
  } else {
    // Special case of y4 == 1, in which the innermost loop is a single element
    // and can be combined with the next (y3) as an inner broadcast.
    //
    // Note that this handles the case of pure scalar broadcast when
    // y0 == y1 == y2 == 1. With low overhead it handles cases such as scalar
    // broadcast with batch (as y2 > 1).
    //
    // NOTE The process is the same as the above general case except simplified
    // for y4 == 1 and the loop over y3 is contained within the
    // AddScalarBroadcast function.
    for (int i0 = 0; i0 < y0; ++i0) {
      const uint8_t* input2_data_ptr;
      for (int i1 = 0; i1 < y1; ++i1) {
        input2_data_ptr = input2_data_reset;
        for (int i2 = 0; i2 < y2; ++i2) {
          AddScalarBroadcast(y3, params, *input1_data_ptr, input2_data_ptr,
                             output_data_ptr);
          input2_data_ptr += y3;
          output_data_ptr += y3;
          input1_data_ptr += 1;
        }
      }
      input2_data_reset = input2_data_ptr;
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_H_
