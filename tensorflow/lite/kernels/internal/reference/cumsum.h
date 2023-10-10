/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace tflite {
namespace reference_ops {

template <typename T>
inline void CumSum(const T* input_data, const RuntimeShape& shape, int32_t axis,
                   bool exclusive, bool reverse, T* output_data) {
  const int32_t rank = shape.DimensionsCount();
  TFLITE_DCHECK_GE(rank, 1);
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, rank);

  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (i < axis)
      inner *= shape.Dims(i);
    else if (i > axis)
      outer *= shape.Dims(i);
    else
      depth = shape.Dims(i);
  }

  for (size_t outer_index = 0; outer_index < outer; outer_index++) {
    size_t outer_index_adj;
    if (reverse)
      outer_index_adj = (outer - 1) - outer_index;
    else
      outer_index_adj = outer_index;
    for (size_t inner_index = 0; inner_index < inner; inner_index++) {
      T accumulator = 0;
      size_t inner_index_adj;
      if (reverse)
        inner_index_adj = (inner - 1) - inner_index;
      else
        inner_index_adj = inner_index;
      for (size_t depth_index = 0; depth_index < depth; depth_index++) {
        size_t depth_index_adj;
        if (reverse)
          depth_index_adj = (depth - 1) - depth_index;
        else
          depth_index_adj = depth_index;

        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;

        if (exclusive) {
          output_data[index] = accumulator;
          accumulator += input_data[index];
        } else {
          accumulator += input_data[index];
          output_data[index] = accumulator;
        }
      }
    }
  }
}

//
// Quantized INT8 CUMSUM
//
inline void CumSum(const ArithmeticParams& params, const int8_t* input_data,
                   const RuntimeShape& shape, int32_t axis, bool exclusive,
                   bool reverse, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  // All inputs should have same zero-point and scale, this is checked during
  // Prepare stage.
  TFLITE_DCHECK_GE(-params.input1_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(-params.input1_offset, std::numeric_limits<int8_t>::max());

  const int32_t rank = shape.DimensionsCount();
  TFLITE_DCHECK_GE(rank, 1);
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, rank);

  size_t inner = 1;
  size_t outer = 1;
  size_t depth = 1;
  for (int32_t i = 0; i < rank; i++) {
    if (i < axis)
      inner *= shape.Dims(i);
    else if (i > axis)
      outer *= shape.Dims(i);
    else
      depth = shape.Dims(i);
  }

  for (size_t outer_index = 0; outer_index < outer; outer_index++) {
    size_t outer_index_adj;
    if (reverse)
      outer_index_adj = (outer - 1) - outer_index;
    else
      outer_index_adj = outer_index;
    for (size_t inner_index = 0; inner_index < inner; inner_index++) {
      int32_t accumulator = params.input1_offset;  // accumulator = 0
      accumulator *= (1 << params.left_shift);
      accumulator = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          accumulator, params.input1_multiplier, params.input1_shift);

      size_t inner_index_adj;
      if (reverse)
        inner_index_adj = (inner - 1) - inner_index;
      else
        inner_index_adj = inner_index;

      for (size_t depth_index = 0; depth_index < depth; depth_index++) {
        size_t depth_index_adj;
        if (reverse)
          depth_index_adj = (depth - 1) - depth_index;
        else
          depth_index_adj = depth_index;

        size_t index = outer_index_adj;
        index += inner_index_adj * depth * outer;
        index += depth_index_adj * outer;

        const int32_t y = params.input1_offset + input_data[index];
        const int32_t shifted_y = y * (1 << params.left_shift);
        const int32_t scaled_y = MultiplyByQuantizedMultiplierSmallerThanOneExp(
            shifted_y, params.input1_multiplier, params.input1_shift);

        int32_t scaled_output;
        if (exclusive) {
          scaled_output = accumulator;
          accumulator += scaled_y;
        } else {
          accumulator += scaled_y;
          scaled_output = accumulator;
        }

        const int32_t raw_output =
            MultiplyByQuantizedMultiplierSmallerThanOneExp(
                scaled_output, params.output_multiplier, params.output_shift) +
            params.output_offset;
        const int32_t clamped_output =
            std::min(params.quantized_activation_max,
                     std::max(params.quantized_activation_min, raw_output));
        output_data[index] = static_cast<int8_t>(clamped_output);
      }
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_CUMSUM_H_
