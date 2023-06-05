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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

inline void CheckArithmeticParams(const ArithmeticParams& params) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  TFLITE_DCHECK_GE(-params.input1_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_GE(-params.input2_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(-params.input1_offset, std::numeric_limits<int8_t>::max());
  TFLITE_DCHECK_LE(-params.input2_offset, std::numeric_limits<int8_t>::max());
}

// TODO(b/270589088): move to a more appropriate file (b/270589088#comment2)
template <typename T>
void ElementWise(int size, const ArithmeticParams& params, const T* input1_data,
                 const T* input2_data, T* output_data,
                 void (*check_arithmetic_params)(const ArithmeticParams&),
                 T (*binary_func)(T, T, const ArithmeticParams&)) {
  CheckArithmeticParams(params);
  for (int i = 0; i < size; ++i) {
    output_data[i] = binary_func(input1_data[i], input2_data[i], params);
  }
}
// TODO(b/270589088): move to a more appropriate file. (b/270589088#comment2)
template <typename T>
void BroadcastBinaryFunction6DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data,
    void (*check_arithmetic_params)(const ArithmeticParams&),
    T (*binary_func)(T, T, const ArithmeticParams&)) {
  NdArrayDesc<6> desc1;
  NdArrayDesc<6> desc2;
  NdArrayDescsForElementwiseBroadcast(input1_shape, input2_shape, &desc1,
                                      &desc2);
  const RuntimeShape extended_output_shape =
      RuntimeShape::ExtendedShape(6, output_shape);

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
  size_t input1_offset_a = 0;
  size_t input2_offset_a = 0;
  size_t output_offset_a = 0;
  for (int a = 0; a < extended_output_shape.Dims(0); ++a) {
    size_t input1_offset_d = input1_offset_a;
    size_t input2_offset_d = input2_offset_a;
    size_t output_offset_d = output_offset_a;
    for (int d = 0; d < extended_output_shape.Dims(1); ++d) {
      size_t input1_offset_b = input1_offset_d;
      size_t input2_offset_b = input2_offset_d;
      size_t output_offset_b = output_offset_d;
      for (int b = 0; b < extended_output_shape.Dims(2); ++b) {
        size_t input1_offset_y = input1_offset_b;
        size_t input2_offset_y = input2_offset_b;
        size_t output_offset_y = output_offset_b;
        for (int y = 0; y < extended_output_shape.Dims(3); ++y) {
          size_t input1_offset_x = input1_offset_y;
          size_t input2_offset_x = input2_offset_y;
          size_t output_offset_x = output_offset_y;
          for (int x = 0; x < extended_output_shape.Dims(4); ++x) {
            size_t input1_offset_c = input1_offset_x;
            size_t input2_offset_c = input2_offset_x;
            size_t output_offset_c = output_offset_x;
            for (int c = 0; c < extended_output_shape.Dims(5); ++c) {
              output_data[output_offset_c] =
                  binary_func(input1_data[input1_offset_c],
                              input2_data[input2_offset_c], params);
              input1_offset_c += desc1.strides[5];
              input2_offset_c += desc2.strides[5];
              ++output_offset_c;
            }
            input1_offset_x += desc1.strides[4];
            input2_offset_x += desc2.strides[4];
            output_offset_x += extended_output_shape.Dims(5);
          }
          input1_offset_y += desc1.strides[3];
          input2_offset_y += desc2.strides[3];
          output_offset_y +=
              extended_output_shape.Dims(4) * extended_output_shape.Dims(5);
        }
        input1_offset_b += desc1.strides[2];
        input2_offset_b += desc2.strides[2];
        output_offset_b += extended_output_shape.Dims(3) *
                           extended_output_shape.Dims(4) *
                           extended_output_shape.Dims(5);
      }
      input1_offset_d += desc1.strides[1];
      input2_offset_d += desc2.strides[1];
      output_offset_d +=
          extended_output_shape.Dims(2) * extended_output_shape.Dims(3) *
          extended_output_shape.Dims(4) * extended_output_shape.Dims(5);
    }
    input1_offset_a += desc1.strides[0];
    input2_offset_a += desc2.strides[0];
    output_offset_a +=
        extended_output_shape.Dims(1) * extended_output_shape.Dims(2) *
        extended_output_shape.Dims(3) * extended_output_shape.Dims(4) *
        extended_output_shape.Dims(5);
  }
}

template <typename T>
void BroadcastBinaryFunction4DSlow(
    const ArithmeticParams& params, const RuntimeShape& input1_shape,
    const T* input1_data, const RuntimeShape& input2_shape,
    const T* input2_data, const RuntimeShape& output_shape, T* output_data,
    void (*check_arithmetic_params)(const ArithmeticParams&),
    T (*binary_func)(T, T, const ArithmeticParams&)) {
  BroadcastBinaryFunction6DSlow(params, input1_shape, input1_data, input2_shape,
                                input2_data, output_shape, output_data,
                                check_arithmetic_params, binary_func);
}

inline int8_t AddFunc(int8_t x, int8_t y, const ArithmeticParams& params) {
  const int32_t input1_val = params.input1_offset + x;
  const int32_t input2_val = params.input2_offset + y;
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
  return static_cast<int8_t>(clamped_output);
}

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const int8_t* input1_data, const int8_t* input2_data,
                           int8_t* output_data) {
  ElementWise(size, params, input1_data, input2_data, output_data,
              CheckArithmeticParams, AddFunc);
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  CheckArithmeticParams(params);

  const int flat_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastAdd6DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const int8_t* input1_data,
                               const RuntimeShape& input2_shape,
                               const int8_t* input2_data,
                               const RuntimeShape& output_shape,
                               int8_t* output_data) {
  BroadcastBinaryFunction6DSlow(params, input1_shape, input1_data, input2_shape,
                                input2_data, output_shape, output_data,
                                CheckArithmeticParams, AddFunc);
}

inline void BroadcastAdd4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const int8_t* input1_data,
                               const RuntimeShape& input2_shape,
                               const int8_t* input2_data,
                               const RuntimeShape& output_shape,
                               int8_t* output_data) {
  BroadcastBinaryFunction6DSlow(params, input1_shape, input1_data, input2_shape,
                                input2_data, output_shape, output_data,
                                CheckArithmeticParams, AddFunc);
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_H_
