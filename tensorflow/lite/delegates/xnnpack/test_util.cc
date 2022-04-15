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

#include "tensorflow/lite/delegates/xnnpack/test_util.h"

#include <algorithm>
#include <limits>

#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {
namespace xnnpack {

int8_t QuantizeInt8(float value, int32_t zero_point, float scale) {
  static constexpr int32_t min_val = std::numeric_limits<int8_t>::min();
  static constexpr int32_t max_val = std::numeric_limits<int8_t>::max();

  int32_t unclamped =
      static_cast<int32_t>(TfLiteRound(value / scale)) + zero_point;
  int32_t clamped = std::min(std::max(unclamped, min_val), max_val);
  return static_cast<int8_t>(clamped);
}

void QuantizeInt8PerChannel(const float* scale, const int64_t* zero_point,
                            int32_t quantized_dimension,
                            const float* input_data, int8_t* output_data,
                            const std::vector<int32_t>& shape) {
  const int32_t num_dims = shape.size();
  const int32_t* dims_data = shape.data();
  std::vector<int> current_dim(num_dims, 0);

  do {
    const size_t offset =
        ReducedOutputOffset(num_dims, reinterpret_cast<const int*>(dims_data),
                            current_dim.data(), 0, nullptr);
    const int channel_idx = current_dim[quantized_dimension];
    output_data[offset] = QuantizeInt8(
        input_data[offset], zero_point[channel_idx], scale[channel_idx]);
  } while (NextIndex(num_dims, reinterpret_cast<const int*>(dims_data),
                     current_dim.data()));
}

float GetInt8QuantizationScaleFromMinMax(float min, float max) {
  static constexpr int8_t qmin_val = std::numeric_limits<int8_t>::min();
  static constexpr int8_t qmax_val = std::numeric_limits<int8_t>::max();
  static constexpr float qmin_float = qmin_val;
  static constexpr float qmax_float = qmax_val;

  float range = max - min;
  if (range == 0) {
    range = std::max(std::abs(max), std::abs(min));
  }
  return range / (qmax_float - qmin_float);
}

float GetInt8QuantizationScale(const std::vector<float>& data) {
  return GetInt8QuantizationScaleFromMinMax(
      *std::max_element(data.begin(), data.end()),
      *std::min_element(data.begin(), data.end()));
}

std::vector<float> GetInt8QuantizationScalePerChannel(
    const float* data, int32_t quantized_dimension,
    const std::vector<int32_t>& shape) {
  const int32_t num_dims = shape.size();
  const int32_t* dims_data = shape.data();
  const int32_t channel_dim_size = shape[quantized_dimension];
  std::vector<float> min(channel_dim_size, std::numeric_limits<float>::max());
  std::vector<float> max(channel_dim_size, std::numeric_limits<float>::min());
  std::vector<int> current_dim(num_dims, 0);

  do {
    const size_t offset =
        ReducedOutputOffset(num_dims, reinterpret_cast<const int*>(dims_data),
                            current_dim.data(), 0, nullptr);
    const int channel_idx = current_dim[quantized_dimension];
    const float val = data[offset];
    min[channel_idx] = std::min(min[channel_idx], val);
    max[channel_idx] = std::max(max[channel_idx], val);
  } while (NextIndex(num_dims, reinterpret_cast<const int*>(dims_data),
                     current_dim.data()));

  std::vector<float> scale;
  scale.reserve(channel_dim_size);
  for (int32_t idx = 0; idx < channel_dim_size; ++idx) {
    scale.push_back(GetInt8QuantizationScaleFromMinMax(min[idx], max[idx]));
  }
  return scale;
}

}  // namespace xnnpack
}  // namespace tflite
