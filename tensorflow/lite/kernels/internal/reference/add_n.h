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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_N_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_N_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"

namespace tflite {
namespace reference_ops {

// T is expected to be either float or int.
template <typename T>
inline void AddN(const RuntimeShape& input_shape, const size_t num_inputs,
                 const T* const* input_data, T* output_data) {
  // All inputs and output should have the same shape, this is checked during
  // Prepare stage.
  const size_t size = input_shape.FlatSize();
  for (size_t i = 0; i < size; ++i) {
    T x = 0;
    for (size_t j = 0; j < num_inputs; ++j) {
      x += input_data[j][i];
    }
    output_data[i] = x;
  }
}

inline void AddN(const ArithmeticParams& params,
                 const RuntimeShape& input_shape, const size_t num_inputs,
                 const int8_t* const* input_data, int8_t* output_data) {
  TFLITE_DCHECK_LE(params.quantized_activation_min,
                   params.quantized_activation_max);
  // Input offset is negative input zero point. Activation tensors are
  // asymmetric quantized so they span the full int8 range.
  // All inputs should have same zero-point and scale, this is checked during
  // Prepare stage.
  TFLITE_DCHECK_GE(-params.input1_offset, std::numeric_limits<int8_t>::min());
  TFLITE_DCHECK_LE(-params.input1_offset, std::numeric_limits<int8_t>::max());

  // All inputs and output should have the same shape, this is checked during
  // Prepare stage.
  const size_t size = input_shape.FlatSize();
  for (size_t i = 0; i < size; ++i) {
    // accumulate in scaled_x before clamping to avoid overflow
    const int32_t x = params.input1_offset;  // x = 0
    const int32_t shifted_x = x * (1 << params.left_shift);
    int32_t scaled_x = MultiplyByQuantizedMultiplierSmallerThanOneExp(
        shifted_x, params.input1_multiplier, params.input1_shift);

    for (size_t j = 0; j < num_inputs; ++j) {
      const int32_t y = params.input1_offset + input_data[j][i];
      const int32_t shifted_y = y * (1 << params.left_shift);
      int32_t scaled_y = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          shifted_y, params.input1_multiplier, params.input1_shift);
      scaled_x += scaled_y;
    }

    const int32_t raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            scaled_x, params.output_multiplier, params.output_shift) +
        params.output_offset;
    const int32_t clamped_output =
        std::min(params.quantized_activation_max,
                 std::max(params.quantized_activation_min, raw_output));
    output_data[i] = static_cast<int8_t>(clamped_output);
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_ADD_N_H_
