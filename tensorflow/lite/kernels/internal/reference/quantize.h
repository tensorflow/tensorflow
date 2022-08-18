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
#include <vector>

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
  const int32_t zero_point = op_params.zero_point;
  const double scale = op_params.scale;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  static constexpr int32_t min_val = std::numeric_limits<OutputT>::min();
  static constexpr int32_t max_val = std::numeric_limits<OutputT>::max();

  for (int i = 0; i < flat_size; i++) {
    const InputT val = input_data[i];
    int32_t unclamped =
        static_cast<int32_t>(TfLiteRound(val / static_cast<float>(scale))) +
        zero_point;
    int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[i] = clamped;
  }
}

// Quantizes per-channel.
template <typename InputT, typename OutputT>
inline void PerChannelQuantize(
    const tflite::PerChannelQuantizationParams& op_params,
    const RuntimeShape& input_shape, const InputT* input_data,
    const RuntimeShape& output_shape, OutputT* output_data) {
  // Ensure flat size is same.
  MatchingFlatSize(input_shape, output_shape);

  const int32_t* zero_point = op_params.zero_point;
  const float* scale = op_params.scale;
  const int32_t quantized_dimension = op_params.quantized_dimension;
  const int32_t num_dims = input_shape.DimensionsCount();
  const int32_t* dims_data = input_shape.DimsData();
  std::vector<int> current_dim(num_dims, 0);
  static constexpr int32_t min_val = std::numeric_limits<OutputT>::min();
  static constexpr int32_t max_val = std::numeric_limits<OutputT>::max();

  do {
    size_t offset =
        ReducedOutputOffset(num_dims, reinterpret_cast<const int*>(dims_data),
                            current_dim.data(), 0, nullptr);
    const InputT val = input_data[offset];
    const int channel = current_dim[quantized_dimension];
    int32_t unclamped = static_cast<int32_t>(TfLiteRound(
                            val / static_cast<float>(scale[channel]))) +
                        zero_point[channel];
    int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
    output_data[offset] = static_cast<OutputT>(clamped);
  } while (NextIndex(num_dims, reinterpret_cast<const int*>(dims_data),
                     current_dim.data()));
}

}  // namespace reference_ops

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_QUANTIZE_H_
