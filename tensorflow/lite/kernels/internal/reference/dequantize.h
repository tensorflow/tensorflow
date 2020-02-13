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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEQUANTIZE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEQUANTIZE_H_

#include <limits.h>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

// Dequantizes into a float without rounding.
template <typename InputT, typename OutputT>
inline void Dequantize(const tflite::DequantizationParams& op_params,
                       const RuntimeShape& input_shape,
                       const InputT* input_data,
                       const RuntimeShape& output_shape, OutputT* output_data) {
  int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int32 val = input_data[i];
    const OutputT result = static_cast<OutputT>(scale * (val - zero_point));
    output_data[i] = result;
  }
}

// Dequantizes into an integer with rounding.
template <typename InputT, typename OutputT>
inline void DequantizeInteger(const tflite::DequantizationParams& op_params,
                              const RuntimeShape& input_shape,
                              const InputT* input_data,
                              const RuntimeShape& output_shape,
                              OutputT* output_data) {
  int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    const int32 val = input_data[i];
    const OutputT result =
        static_cast<OutputT>(round(scale * (val - zero_point)));
    output_data[i] = result;
  }
}

}  // namespace reference_ops

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_DEQUANTIZE_H_
