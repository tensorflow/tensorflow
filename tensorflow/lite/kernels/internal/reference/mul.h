/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_MUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_MUL_H_

#include <algorithm>
#include <complex>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {

namespace reference_ops {

// Maximum dimension supported by the broadcast mul operation.
constexpr int kMaxMulBroadcastDim = 6;

// Element-wise mul that can often be used for inner loop of broadcast Mul as
// well as the non-broadcast Mul.
inline void MulElementwise(int size, const ArithmeticParams& params,
                           const uint8_t* input1_data,
                           const uint8_t* input2_data, uint8_t* output_data) {
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
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

template <typename T>
inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const T* input1_data,
                const RuntimeShape& input2_shape, const T* input2_data,
                const RuntimeShape& output_shape, T* output_data) {
  T output_activation_min;
  T output_activation_max;
  GetActivationParams(params, &output_activation_min, &output_activation_max);

  const int flat_size =
      MatchingExtendedShapeFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax<T>(
        input1_data[i] * input2_data[i], output_activation_min,
        output_activation_max);
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape,
                const std::complex<float>* input1_data,
                const RuntimeShape& input2_shape,
                const std::complex<float>* input2_data,
                const RuntimeShape& output_shape,
                std::complex<float>* output_data) {
  const int flat_size =
      MatchingExtendedShapeFlatSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = input1_data[i] * input2_data[i];
  }
}

inline void Mul(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const uint8_t* input1_data,
                const RuntimeShape& input2_shape, const uint8_t* input2_data,
                const RuntimeShape& output_shape, uint8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingExtendedShapeFlatSize(input1_shape, input2_shape, output_shape);

  MulElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T, typename F>
void BroadcastMulRecursiveDimensions(
    const ArithmeticParams& params, int dimension, const T* input1_data,
    const T* input2_data, T* output_data, size_t* input1_offset_p,
    size_t* input2_offset_p, size_t* output_offset,
    const NdArrayDesc<kMaxMulBroadcastDim>& desc1,
    const NdArrayDesc<kMaxMulBroadcastDim>& desc2,
    const int32_t extended_output_shape_dims[kMaxMulBroadcastDim],
    F binary_func) {
  if (dimension == kMaxMulBroadcastDim - 1) {
    for (int c = 0; c < extended_output_shape_dims[dimension]; ++c) {
      const T input1_val = input1_data[*input1_offset_p];
      const T input2_val = input2_data[*input2_offset_p];
      output_data[*output_offset] = binary_func(params, input1_val, input2_val);
      *input1_offset_p += desc1.strides[dimension];
      *input2_offset_p += desc2.strides[dimension];
      ++(*output_offset);
    }
  } else {
    for (int a = 0; a < extended_output_shape_dims[dimension]; ++a) {
      size_t input1_offset_c = *input1_offset_p;
      size_t input2_offset_c = *input2_offset_p;
      BroadcastMulRecursiveDimensions(
          params, dimension + 1, input1_data, input2_data, output_data,
          &input1_offset_c, &input2_offset_c, output_offset, desc1, desc2,
          extended_output_shape_dims, binary_func);
      *input1_offset_p += desc1.strides[dimension];
      *input2_offset_p += desc2.strides[dimension];
    }
  }
}

inline void BroadcastMul6DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const uint8_t* input1_data,
                               const RuntimeShape& input2_shape,
                               const uint8_t* input2_data,
                               const RuntimeShape& output_shape,
                               uint8_t* output_data) {
  NdArrayDesc<kMaxMulBroadcastDim> desc1;
  NdArrayDesc<kMaxMulBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxMulBroadcastDim, output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxMulBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastMulRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const uint8_t input1_val,
         const uint8_t input2_val) {
        const int32_t offsetted_input1_val = params.input1_offset + input1_val;
        const int32_t offsetted_input2_val = params.input2_offset + input2_val;
        const int32_t unclamped_result =
            params.output_offset +
            MultiplyByQuantizedMultiplier(
                offsetted_input1_val * offsetted_input2_val,
                params.output_multiplier, params.output_shift);
        const int32_t clamped_output = std::min(
            params.quantized_activation_max,
            std::max(params.quantized_activation_min, unclamped_result));
        return static_cast<uint8_t>(clamped_output);
      });
}

template <typename T,
          // For unquantized mul on small integers, explicitly set to true.
          bool enable_for_short_integers = false>
inline typename std::enable_if<
    !is_small_integer<T>::value || enable_for_short_integers, void>::type
BroadcastMul6DSlow(const ArithmeticParams& params,
                   const RuntimeShape& unextended_input1_shape,
                   const T* input1_data,
                   const RuntimeShape& unextended_input2_shape,
                   const T* input2_data,
                   const RuntimeShape& unextended_output_shape,
                   T* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 6);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 6);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 6);
  NdArrayDesc<kMaxMulBroadcastDim> desc1;
  NdArrayDesc<kMaxMulBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxMulBroadcastDim, unextended_output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxMulBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

  // In Tensorflow, the dimensions are canonically named (batch_number, row,
  // col, channel), with extents (batches, height, width, depth), with the
  // trailing dimension changing most rapidly (channels has the smallest
  // stride, typically 1 element).
  //
  // In generated C code, we store arrays with the dimensions reversed. The
  // first dimension has smallest stride.
  //
  // We name our variables by their Tensorflow convention, but generate C code
  // nesting loops such that the innermost loop has the smallest stride for
  // the best cache behavior.
  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastMulRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const T input1_val,
         const T input2_val) {
        T output_activation_min;
        T output_activation_max;
        GetActivationParams(params, &output_activation_min,
                            &output_activation_max);
        return ActivationFunctionWithMinMax<T>(input1_val * input2_val,
                                               output_activation_min,
                                               output_activation_max);
      });
}

inline void BroadcastMul6DSlow(const ArithmeticParams& params,
                               const RuntimeShape& unextended_input1_shape,
                               const std::complex<float>* input1_data,
                               const RuntimeShape& unextended_input2_shape,
                               const std::complex<float>* input2_data,
                               const RuntimeShape& unextended_output_shape,
                               std::complex<float>* output_data) {
  TFLITE_DCHECK_LE(unextended_input1_shape.DimensionsCount(), 6);
  TFLITE_DCHECK_LE(unextended_input2_shape.DimensionsCount(), 6);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 6);

  NdArrayDesc<kMaxMulBroadcastDim> desc1;
  NdArrayDesc<kMaxMulBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(unextended_input1_shape,
                                      unextended_input2_shape, &desc1, &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxMulBroadcastDim, unextended_output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxMulBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastMulRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const std::complex<float> input1_val,
         const std::complex<float> input2_val) {
        return input1_val * input2_val;
      });
}

template <typename T>
inline void BroadcastMul4DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  return BroadcastMul6DSlow(params, input1_shape, input1_data, input2_shape,
                            input2_data, output_shape, output_data);
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_MUL_H_
