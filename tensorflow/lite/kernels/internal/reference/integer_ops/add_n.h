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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_N_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_N_H_

#include <limits>
#include "public/gemmlowp.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

// T is expected to be either int8 or uint8.
template <typename T>
inline void AddN(const AddNParams* params, const RuntimeShape& input_shape,
                 const size_t num_inputs, T* const* input_data,
                 T* output_data) {
  // All inputs and output should have the same shape, this is checked during
  // Prepare stage.
  const size_t size = input_shape.FlatSize();
  const T int8_max_value = std::numeric_limits<T>::max();
  const T int8_min_value = std::numeric_limits<T>::min();

  for (int i = 0; i < size; ++i) {
    int32 raw_sum = 0;
    for (int j = 0; j < num_inputs; ++j) {
      TFLITE_DCHECK_GE(params->inputs[j].offset, int8_min_value);
      TFLITE_DCHECK_LE(params->inputs[j].offset, int8_max_value);

      const int32 input1_val = params->inputs[j].offset + input_data[j][i];
      const int32 shifted_input1_val = input1_val * (1 << params->left_shift);
      const int32 scaled_input1_val =
          MultiplyByQuantizedMultiplierSmallerThanOneExp(
              shifted_input1_val, params->inputs[j].multiplier,
              params->inputs[j].shift);
      raw_sum += scaled_input1_val;
    }

    const int32 raw_output =
        MultiplyByQuantizedMultiplierSmallerThanOneExp(
            raw_sum, params->output.multiplier, params->output.shift) +
        params->output.offset;
    output_data[i] = static_cast<T>(raw_output);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ADD_N_H_
