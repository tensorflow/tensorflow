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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_QUANTIZE_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_QUANTIZE_H_

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

namespace reference_ops {

template <typename InputT, typename OutputT>
inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                           const RuntimeShape& input_shape,
                           const InputT* input_data,
                           const RuntimeShape& output_shape,
                           OutputT* output_data) {
  const int32 zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32 min_val = std::numeric_limits<OutputT>::min();
  static constexpr int32 max_val = std::numeric_limits<OutputT>::max();

  for (int i = 0; i < flat_size; i++) {
    const InputT val = input_data[i];
    int32 unclamped =
        static_cast<int32>(TfLiteRound(val / static_cast<float>(scale))) +
        zero_point;
    int32 clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

}  // namespace reference_ops

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_QUANTIZE_H_
