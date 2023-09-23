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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_

#include <stdint.h>

#include <algorithm>
#include <limits>

#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Maximum dimension supported by the broadcast sub operation.
constexpr int kMaxSubBroadcastDim = 6;

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const float* input1_data,
                            const RuntimeShape& input2_shape,
                            const float* input2_data,
                            const RuntimeShape& output_shape,
                            float* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.float_activation_min,
        params.float_activation_max);
  }
}

inline void SubNonBroadcast(const ArithmeticParams& params,
                            const RuntimeShape& input1_shape,
                            const int32_t* input1_data,
                            const RuntimeShape& input2_shape,
                            const int32_t* input2_data,
                            const RuntimeShape& output_shape,
                            int32_t* output_data) {
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], params.quantized_activation_min,
        params.quantized_activation_max);
  }
}

template <typename T, typename F>
void BroadcastSubRecursiveDimensions(
    const ArithmeticParams& params, int dimension, const T* input1_data,
    const T* input2_data, T* output_data, size_t* input1_offset_p,
    size_t* input2_offset_p, size_t* output_offset,
    const NdArrayDesc<kMaxSubBroadcastDim>& desc1,
    const NdArrayDesc<kMaxSubBroadcastDim>& desc2,
    const int32_t extended_output_shape_dims[kMaxSubBroadcastDim],
    F binary_func) {
  if (dimension == kMaxSubBroadcastDim - 1) {
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
      BroadcastSubRecursiveDimensions(
          params, dimension + 1, input1_data, input2_data, output_data,
          &input1_offset_c, &input2_offset_c, output_offset, desc1, desc2,
          extended_output_shape_dims, binary_func);
      *input1_offset_p += desc1.strides[dimension];
      *input2_offset_p += desc2.strides[dimension];
    }
  }
}

// TODO(b/151345304): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
template <typename T>
void BroadcastSubSlow(const ArithmeticParams& params,
                      const RuntimeShape& input1_shape, const T* input1_data,
                      const RuntimeShape& input2_shape, const T* input2_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/T");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), kMaxSubBroadcastDim);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), kMaxSubBroadcastDim);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), kMaxSubBroadcastDim);
  NdArrayDesc<kMaxSubBroadcastDim> desc1;
  NdArrayDesc<kMaxSubBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxSubBroadcastDim, output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxSubBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

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
  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastSubRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const T input1_val,
         const T input2_val) {
        T activation_min, activation_max;
        GetActivationParams(params, &activation_min, &activation_max);
        return ActivationFunctionWithMinMax<T>(input1_val - input2_val,
                                               activation_min, activation_max);
      });
}

inline void BroadcastSub16POTSlow(const ArithmeticParams& params,
                                  const RuntimeShape& input1_shape,
                                  const int16_t* input1_data,
                                  const RuntimeShape& input2_shape,
                                  const int16_t* input2_data,
                                  const RuntimeShape& output_shape,
                                  int16_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSub16POTSlow/int16_t");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), kMaxSubBroadcastDim);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), kMaxSubBroadcastDim);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), kMaxSubBroadcastDim);
  NdArrayDesc<kMaxSubBroadcastDim> desc1;
  NdArrayDesc<kMaxSubBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxSubBroadcastDim, output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxSubBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

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
  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastSubRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const int16_t input1_val,
         const int16_t input2_val) {
        const int32_t scaled_input1_val =
            gemmlowp::RoundingDivideByPOT(input1_val, -params.input1_shift);
        const int32_t scaled_input2_val =
            gemmlowp::RoundingDivideByPOT(input2_val, -params.input2_shift);
        const int32_t raw_output = scaled_input1_val - scaled_input2_val;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        return static_cast<int16_t>(clamped_output);
      });
}

template <typename T>
void BroadcastQuantSubSlow(const ArithmeticParams& params,
                           const RuntimeShape& input1_shape,
                           const T* input1_data,
                           const RuntimeShape& input2_shape,
                           const T* input2_data,
                           const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastQuantSubSlow/T");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), kMaxSubBroadcastDim);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), kMaxSubBroadcastDim);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), kMaxSubBroadcastDim);
  NdArrayDesc<kMaxSubBroadcastDim> desc1;
  NdArrayDesc<kMaxSubBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxSubBroadcastDim, output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxSubBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

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
  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastSubRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const T input1_val,
         const T input2_val) {
        const int32_t shifted_input1_val =
            (params.input1_offset + input1_val) * (1 << params.left_shift);
        const int32_t shifted_input2_val =
            (params.input2_offset + input2_val) * (1 << params.left_shift);
        const int32_t scaled_input1_val =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                shifted_input1_val, params.input1_multiplier,
                params.input1_shift);
        const int32_t scaled_input2_val =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                shifted_input2_val, params.input2_multiplier,
                params.input2_shift);
        const int32_t raw_sub = scaled_input1_val - scaled_input2_val;
        const int32_t raw_output =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                raw_sub, params.output_multiplier, params.output_shift) +
            params.output_offset;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        return static_cast<T>(clamped_output);
      });
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
template <typename T>
inline void SubElementwise(int size, const ArithmeticParams& params,
                           const T* input1_data, const T* input2_data,
                           T* output_data) {
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
    const int32_t raw_sub = scaled_input1_val - scaled_input2_val;
    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sub, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<T>(clamped_output);
  }
}

inline void Sub(const ArithmeticParams& params,
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
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_GE(params.input1_offset, -128);
  TFLITE_DCHECK_GE(params.input2_offset, -128);
  // offset = -quantization_params.zero_point in PrepareGeneralSubOp().
  // So it's maximum can be 128 not 127.
  TFLITE_DCHECK_LE(params.input1_offset, 128);
  TFLITE_DCHECK_LE(params.input2_offset, 128);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void Sub(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int16_t* input1_data,
                const RuntimeShape& input2_shape, const int16_t* input2_data,
                const RuntimeShape& output_shape, int16_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  TFLITE_DCHECK_EQ(params.input1_offset, 0);
  TFLITE_DCHECK_EQ(params.input2_offset, 0);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  NdArrayDesc<kMaxSubBroadcastDim> desc1;
  NdArrayDesc<kMaxSubBroadcastDim> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(kMaxSubBroadcastDim, output_shape);
  // Cache output shape dimensions.
  int32_t extended_output_shape_dims[kMaxSubBroadcastDim];
  std::memcpy(extended_output_shape_dims, extended_output_shape.DimsData(),
              sizeof(extended_output_shape_dims));

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
  size_t input1_offset = 0;
  size_t input2_offset = 0;
  size_t output_offset = 0;
  BroadcastSubRecursiveDimensions(
      params, 0, input1_data, input2_data, output_data, &input1_offset,
      &input2_offset, &output_offset, desc1, desc2, extended_output_shape_dims,
      [](const ArithmeticParams& params, const T input1_val,
         const T input2_val) { return input1_val - input2_val; });
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                int32_t* activation_min,
                                int32_t* activation_max) {
  *activation_min = params.quantized_activation_min;
  *activation_max = params.quantized_activation_max;
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                float* activation_min, float* activation_max) {
  *activation_min = params.float_activation_min;
  *activation_max = params.float_activation_max;
}

inline void SetActivationMinMax(const ArithmeticParams& params,
                                int64_t* activation_min,
                                int64_t* activation_max) {
  *activation_min = params.int64_activation_min;
  *activation_max = params.int64_activation_max;
}

template <typename T>
inline void SubWithActivation(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("SubWithActivation");
  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);
  T activation_min, activation_max;
  SetActivationMinMax(params, &activation_min, &activation_max);

  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = ActivationFunctionWithMinMax(
        input1_data[i] - input2_data[i], activation_min, activation_max);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_SUB_H_
