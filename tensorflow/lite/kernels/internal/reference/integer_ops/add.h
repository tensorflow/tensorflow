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

#include <limits>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

// Element-wise add that can often be used for inner loop of broadcast add as
// well as the non-broadcast add.
inline void AddElementwise(int size, const ArithmeticParams& params,
                           const int8_t* input1_data, const int8_t* input2_data,
                           int8_t* output_data) {
  const int32_t int8_max_value = std::numeric_limits<int8_t>::max();
  TFLITE_DCHECK_GE(params.input1_offset, -1 * int8_max_value);
  TFLITE_DCHECK_GE(params.input2_offset, -1 * int8_max_value);
  TFLITE_DCHECK_LE(params.input1_offset, int8_max_value);
  TFLITE_DCHECK_LE(params.input2_offset, int8_max_value);

  for (int i = 0; i < size; ++i) {
    const int32 input1_val = params.input1_offset + input1_data[i];
    const int32 input2_val = params.input2_offset + input2_data[i];
    const int32 shifted_input1_val = input1_val * (1 << params.left_shift);
    const int32 shifted_input2_val = input2_val * (1 << params.left_shift);
    const int32 scaled_input1_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input1_val, params.input1_multiplier, params.input1_shift);
    const int32 scaled_input2_val =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_input2_val, params.input2_multiplier, params.input2_shift);
    const int32 raw_sum = scaled_input1_val + scaled_input2_val;
    const int32 raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32 clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

inline void Add(const ArithmeticParams& params,
                const RuntimeShape& input1_shape, const int8_t* input1_data,
                const RuntimeShape& input2_shape, const int8_t* input2_data,
                const RuntimeShape& output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  const int flat_size =
      MatchingFlatSize(input1_shape, input2_shape, output_shape);

  const int32_t int8_max_value = std::numeric_limits<int8_t>::max();
  TFLITE_DCHECK_GE(params.input1_offset, -1 * int8_max_value);
  TFLITE_DCHECK_GE(params.input2_offset, -1 * int8_max_value);
  TFLITE_DCHECK_LE(params.input1_offset, int8_max_value);
  TFLITE_DCHECK_LE(params.input2_offset, int8_max_value);
  AddElementwise(flat_size, params, input1_data, input2_data, output_data);
}

inline void BroadcastAdd4DSlow(const ArithmeticParams& params,
                               const RuntimeShape& input1_shape,
                               const int8_t* input1_data,
                               const RuntimeShape& input2_shape,
                               const int8_t* input2_data,
                               const RuntimeShape& output_shape,
                               int8_t* output_data) {
  gemmlowp::ScopedProfilingLabel label("BroadcastAdd4DSlow/int8");
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
          const int32_t input1_val =
              params.input1_offset +
              input1_data[SubscriptToIndex(desc1, b, y, x, c)];
          const int32_t input2_val =
              params.input2_offset +
              input2_data[SubscriptToIndex(desc2, b, y, x, c)];
          const int32_t shifted_input1_val =
              input1_val * (1 << params.left_shift);
          const int32_t shifted_input2_val =
              input2_val * (1 << params.left_shift);
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
          output_data[Offset(extended_output_shape, b, y, x, c)] =
              static_cast<int8_t>(clamped_output);
        }
      }
    }
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_H_
