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

// TODO(b/151345304): We can implement BroadcastSub on buffers of arbitrary
// dimensionality if the runtime code does a single loop over one dimension
// that handles broadcasting as the base case. The code generator would then
// generate max(D1, D2) nested for loops.
// TODO(b/151345101): BroadcastSub is intentionally duplicated from
// reference_ops.h. Once an optimized version is implemented and NdArrayDesc<T>
// is no longer referenced in this file, move NdArrayDesc<T> from types.h to
// reference_ops.h.
template <int N = 5>
inline void BroadcastSubSlow(const ArithmeticParams& params,
                             const RuntimeShape& input1_shape,
                             const float* input1_data,
                             const RuntimeShape& input2_shape,
                             const float* input2_data,
                             const RuntimeShape& output_shape,
                             float* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/float");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.float_activation_min, params.float_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
inline void BroadcastSubSlow(const ArithmeticParams& params,
                             const RuntimeShape& input1_shape,
                             const uint8_t* input1_data,
                             const RuntimeShape& input2_shape,
                             const uint8_t* input2_data,
                             const RuntimeShape& output_shape,
                             uint8_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/uint8_t");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
    const int32_t input1_val =
        params.input1_offset + input1_data[SubscriptToIndex(desc1, indexes)];
    const int32_t input2_val =
        params.input2_offset + input2_data[SubscriptToIndex(desc2, indexes)];
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
    output_data[SubscriptToIndex(output_desc, indexes)] =
        static_cast<uint8_t>(clamped_output);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
inline void BroadcastSubSlow(const ArithmeticParams& params,
                             const RuntimeShape& input1_shape,
                             const int32_t* input1_data,
                             const RuntimeShape& input2_shape,
                             const int32_t* input2_data,
                             const RuntimeShape& output_shape,
                             int32_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/int32_t");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.quantized_activation_min, params.quantized_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
inline void BroadcastSubSlow(const ArithmeticParams& params,
                             const RuntimeShape& input1_shape,
                             const int8_t* input1_data,
                             const RuntimeShape& input2_shape,
                             const int8_t* input2_data,
                             const RuntimeShape& output_shape,
                             int8_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/int8_t");
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
    const int32_t input1_val =
        params.input1_offset + input1_data[SubscriptToIndex(desc1, indexes)];
    const int32_t input2_val =
        params.input2_offset + input2_data[SubscriptToIndex(desc2, indexes)];
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
    output_data[SubscriptToIndex(output_desc, indexes)] =
        static_cast<int8_t>(clamped_output);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <int N = 5>
void BroadcastSubSlow(const ArithmeticParams& params,
                      const RuntimeShape& input1_shape,
                      const int64_t* input1_data,
                      const RuntimeShape& input2_shape,
                      const int64_t* input2_data,
                      const RuntimeShape& output_shape, int64_t* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/int64_t");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.int64_activation_min, params.int64_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

template <typename T, int N = 5>
void BroadcastSubSlow(const ArithmeticParams& params,
                      const RuntimeShape& input1_shape, const T* input1_data,
                      const RuntimeShape& input2_shape, const T* input2_data,
                      const RuntimeShape& output_shape, T* output_data) {
  ruy::profiler::ScopeLabel label("BroadcastSubSlow/templated");
  TFLITE_DCHECK_LE(input1_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(input2_shape.DimensionsCount(), N);
  TFLITE_DCHECK_LE(output_shape.DimensionsCount(), N);
  NdArrayDesc<N> desc1;
  NdArrayDesc<N> desc2;
  NdArrayDesc<N> output_desc;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  CopyDimsToDesc(RuntimeShape::ExtendedShape(N, output_shape), &output_desc);

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
  auto sub_func = [&](int indexes[N]) {
    output_data[SubscriptToIndex(output_desc, indexes)] =
        ActivationFunctionWithMinMax(
            input1_data[SubscriptToIndex(desc1, indexes)] -
                input2_data[SubscriptToIndex(desc2, indexes)],
            params.quantized_activation_min, params.quantized_activation_max);
  };
  NDOpsHelper<N>(output_desc, sub_func);
}

// Element-wise Sub that can often be used for inner loop of broadcast sub as
// well as the non-broadcast sub.
inline void SubElementwise(int size, const ArithmeticParams& params,
                           const uint8_t* input1_data,
                           const uint8_t* input2_data, uint8_t* output_data) {
  TFLITE_DCHECK_GT(params.input1_offset, -256);
  TFLITE_DCHECK_GT(params.input2_offset, -256);
  TFLITE_DCHECK_LT(params.input1_offset, 256);
  TFLITE_DCHECK_LT(params.input2_offset, 256);

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
    output_data[i] = static_cast<uint8_t>(clamped_output);
  }
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void SubElementwise(int size, const ArithmeticParams& params,
                           const int8_t* input1_data, const int8_t* input2_data,
                           int8_t* output_data) {
  const int32_t int8_max_value = std::numeric_limits<int8_t>::max();
  TFLITE_DCHECK_GE(params.input1_offset, -1 * int8_max_value);
  TFLITE_DCHECK_GE(params.input2_offset, -1 * int8_max_value);
  TFLITE_DCHECK_LE(params.input1_offset, int8_max_value);
  TFLITE_DCHECK_LE(params.input2_offset, int8_max_value);

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
    output_data[i] = static_cast<int8_t>(clamped_output);
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

  const int32_t int8_max_value = std::numeric_limits<int8_t>::max();
  TFLITE_DCHECK_GE(params.input1_offset, -1 * int8_max_value);
  TFLITE_DCHECK_GE(params.input2_offset, -1 * int8_max_value);
  TFLITE_DCHECK_LE(params.input1_offset, int8_max_value);
  TFLITE_DCHECK_LE(params.input2_offset, int8_max_value);
  SubElementwise(flat_size, params, input1_data, input2_data, output_data);
}

template <typename T>
void Sub(const ArithmeticParams& params, const RuntimeShape& input1_shape,
         const T* input1_data, const RuntimeShape& input2_shape,
         const T* input2_data, const RuntimeShape& output_shape,
         T* output_data) {
  NdArrayDesc<4> desc1;
  NdArrayDesc<4> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(4, output_shape);

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
  for (int b = 0; b < extended_output_shape.Dims(0); ++b) {
    for (int y = 0; y < extended_output_shape.Dims(1); ++y) {
      for (int x = 0; x < extended_output_shape.Dims(2); ++x) {
        for (int c = 0; c < extended_output_shape.Dims(3); ++c) {
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              input1_data[SubscriptToIndex(desc1, b, y, x, c)] -
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
        }
      }
    }
  }
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
