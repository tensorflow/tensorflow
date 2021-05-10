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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_L2NORMALIZATION_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_L2NORMALIZATION_H_

#include <algorithm>
#include <cmath>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const float* input_data,
                            const RuntimeShape& output_shape,
                            float* output_data, float epsilon = 1e-6) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  for (int i = 0; i < outer_size; ++i) {
    float squared_l2_norm = 0;
    for (int c = 0; c < depth; ++c) {
      const float val = input_data[depth * i + c];
      squared_l2_norm += val * val;
    }
    float l2_norm = std::sqrt(squared_l2_norm);
    l2_norm = std::max(l2_norm, epsilon);
    for (int c = 0; c < depth; ++c) {
      output_data[depth * i + c] = input_data[depth * i + c] / l2_norm;
    }
  }
}

inline void L2Normalization(const tflite::L2NormalizationParams& op_params,
                            const RuntimeShape& input_shape,
                            const uint8_t* input_data,
                            const RuntimeShape& output_shape,
                            uint8_t* output_data) {
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int32_t input_zero_point = op_params.input_zero_point;

  for (int i = 0; i < outer_size; ++i) {
    int32_t square_l2_norm = 0;
    for (int c = 0; c < depth; c++) {
      int32_t diff = input_data[depth * i + c] - input_zero_point;
      square_l2_norm += diff * diff;
    }
    int32_t inv_l2norm_multiplier;
    int inv_l2norm_shift;
    GetInvSqrtQuantizedMultiplierExp(square_l2_norm, kReverseShift,
                                     &inv_l2norm_multiplier, &inv_l2norm_shift);
    for (int c = 0; c < depth; c++) {
      int32_t diff = input_data[depth * i + c] - input_zero_point;
      int32_t rescaled_diff = MultiplyByQuantizedMultiplierSmallerThanOneExp(
          128 * diff, inv_l2norm_multiplier, inv_l2norm_shift);
      int32_t unclamped_output_val = 128 + rescaled_diff;
      int32_t output_val =
          std::min(static_cast<int32_t>(255),
                   std::max(static_cast<int32_t>(0), unclamped_output_val));
      output_data[depth * i + c] = static_cast<uint8_t>(output_val);
    }
  }
}

}  // namespace reference_ops
}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_L2NORMALIZATION_H_
