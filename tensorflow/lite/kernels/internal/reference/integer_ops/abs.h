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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ABS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ABS_H_

#include <cmath>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace reference_integer_ops {

template <typename T>
inline void Abs(const AbsParams& params, const RuntimeShape& input_shape,
                const T* input_data, const RuntimeShape& output_shape,
                T* output_data) {
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  const int32_t q_min_val = static_cast<int32_t>(std::numeric_limits<T>::min());
  const int32_t q_max_val = static_cast<int32_t>(std::numeric_limits<T>::max());
  TFLITE_DCHECK_GE(params.input_offset, q_min_val);
  TFLITE_DCHECK_LE(params.input_offset, q_max_val);

  for (int i = 0; i < flat_size; ++i) {
    const int32 input_val = std::abs(params.input_offset + input_data[i]);
    const int32 unclamped_result =
        params.output_offset +
        MultiplyByQuantizedMultiplier(input_val, params.output_multiplier,
                                      params.output_shift);
    const int32 clamped_output = clamp(unclamped_result, q_min_val, q_max_val);
    output_data[i] = static_cast<T>(clamped_output);
  }
}

}  // namespace reference_integer_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_INTEGER_OPS_ABS_H_
