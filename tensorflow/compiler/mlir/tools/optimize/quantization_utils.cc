/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/tools/optimize/quantization_utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/mlir/tools/safe_cast.h"
#include "xla/tsl/platform/status.h"
#include "tensorflow/core/framework/tensor_shape.h"

namespace tflite_migration {
namespace optimize {
namespace utils {

namespace {

const int8_t kMinQuantizedValue8bit = -127;
const int8_t kMaxQuantizedValue8bit = 127;

// const int8_t kMinQuantizedValue4bit = -7;
// const int8_t kMaxQuantizedValue4bit = 7;

// The maximum number of dimensions supported in per-channel quantization.
constexpr int kPerChannelMaxDim = 4;
}  // namespace

template <class BiasType>
std::vector<BiasType> SymmetricBiasQuantize(const float* data,
                                            uint64_t num_elements,
                                            const std::vector<float>& scales) {
  std::vector<BiasType> buffer(num_elements);
  const BiasType kScale = std::numeric_limits<BiasType>::max();
  float scaling_factor_inv_per_layer = (scales[0] == 0) ? 0 : 1.0 / scales[0];

  for (int32_t idx = 0; idx < num_elements; idx++) {
    float scaling_factor_inv =
        scales.size() == 1 ? scaling_factor_inv_per_layer
                           : ((scales[idx] == 0) ? 0 : 1.0 / scales[idx]);
    const BiasType quantized_value =
        tools::SafeCast<BiasType>(std::round(data[idx] * scaling_factor_inv));
    buffer[idx] = std::min(kScale, std::max(-kScale, quantized_value));
  }
  return buffer;
}

template std::vector<std::int32_t> SymmetricBiasQuantize<std::int32_t>(
    const float* data, uint64_t num_elements, const std::vector<float>& scales);

template std::vector<std::int64_t> SymmetricBiasQuantize<std::int64_t>(
    const float* data, uint64_t num_elements, const std::vector<float>& scales);

std::vector<int16_t> SymmetricQuantizeFloatsToInt16(const float* data,
                                                    uint64_t num_elements,
                                                    float scaling_factor) {
  // Compute the inverse of scale.
  const float scaling_factor_inv =
      (scaling_factor == 0) ? 0 : 1.0 / scaling_factor;
  std::vector<int16_t> buffer(num_elements);
  const int32_t kScale = std::numeric_limits<int16_t>::max();

  for (size_t i = 0; i < num_elements; i++) {
    const int32_t quantized_value =
        static_cast<int32_t>(std::round(data[i] * scaling_factor_inv));
    buffer[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
  return buffer;
}

void SymmetricPerChannelQuantizeValues(const float* const input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value) {
  // Quantize the values.
  int indices[kPerChannelMaxDim];
  tensorflow::TensorShape unextended_shape;
  TF_CHECK_OK(tensorflow::TensorShapeUtils::MakeShape(absl::MakeSpan(dimension),
                                                      &unextended_shape));
  tensorflow::TensorShape shape;
  for (int i = 0; i < kPerChannelMaxDim - unextended_shape.dims(); ++i) {
    TF_CHECK_OK(shape.AddDimWithStatus(1));
  }
  TF_CHECK_OK(shape.AppendShapeWithStatus(unextended_shape));
  channel_dim_index += kPerChannelMaxDim - unextended_shape.dims();

  for (indices[0] = 0; indices[0] < shape.dim_size(0); indices[0]++) {
    for (indices[1] = 0; indices[1] < shape.dim_size(1); indices[1]++) {
      for (indices[2] = 0; indices[2] < shape.dim_size(2); indices[2]++) {
        for (indices[3] = 0; indices[3] < shape.dim_size(3); indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          int index = 0;
          int current_stride = 1;
          for (int i = kPerChannelMaxDim - 1; i >= 0; --i) {
            index += indices[i] * current_stride;
            current_stride *= shape.dim_size(i);
          }
          const float val = input[index];
          const int32_t quantized_value =
              static_cast<int32_t>(std::round(val * scales_inv[channel_idx]));
          output_value->at(index) = std::min<int8_t>(
              kMaxQuantizedValue8bit,
              std::max<int8_t>(kMinQuantizedValue8bit, quantized_value));
        }
      }
    }
  }
}

}  // namespace utils
}  // namespace optimize
}  // namespace tflite_migration
