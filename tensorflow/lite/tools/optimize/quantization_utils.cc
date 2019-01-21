/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/optimize/quantization_utils.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/round.h"
#include "tensorflow/lite/kernels/internal/types.h"

#include <cmath>
#include <cstdint>

namespace tflite {
namespace optimize {
namespace utils {

namespace {
const int8_t kMinQuantizedValue = -127;
const int8_t kMaxQuantizedValue = 127;
}  // namespace

TfLiteStatus NumElements(const TensorT& tensor, uint64_t* num_elements) {
  if (tensor.shape.empty()) {
    return kTfLiteError;
  }
  *num_elements = 1;
  for (const uint64_t dim : tensor.shape) {
    *num_elements *= dim;
  }
  return kTfLiteOk;
}

// Nudge min and max so that floating point 0 falls exactly on a quantized
// value, returning the nudges scale and zero_point.
//
// Although this code originates from FakeQuantization in quantized training,
// we may deviate from that implementation as we please since we do not fine
// tune the weights with quantized training.
void GetAsymmetricQuantizationParams(
    float min, float max, const int quant_min, const int quant_max,
    QuantizationParametersT* quantization_params) {
  const float quant_min_float = static_cast<float>(quant_min);
  const float quant_max_float = static_cast<float>(quant_max);
  // Adjust the boundaries to guarantee 0 is included.
  min = std::min(static_cast<float>(min), 0.0f);
  max = std::max(static_cast<float>(max), 0.0f);
  const float scale = (max - min) / (quant_max_float - quant_min_float);
  const float zero_point_from_min = quant_min_float - min / scale;
  int64_t zero_point;
  if (zero_point_from_min < quant_min_float) {
    zero_point = static_cast<int64_t>(quant_min);
  } else if (zero_point_from_min > quant_max_float) {
    zero_point = static_cast<int64_t>(quant_max);
  } else {
    zero_point = static_cast<int64_t>(std::round(zero_point_from_min));
  }
  quantization_params->min = std::vector<float>(1, min);
  quantization_params->max = std::vector<float>(1, max);
  quantization_params->scale = std::vector<float>(1, scale);
  quantization_params->zero_point = std::vector<int64_t>(1, zero_point);
}

// Per-channel quantize a tensor at the given index and returns both scales and
// quantized values.
void SymmetricPerChannelQuantization(const float* const input,
                                     const std::vector<int>& dimension,
                                     int32_t channel_dim_index,
                                     std::vector<float>* output_scales,
                                     std::vector<int8_t>* output_value) {
  const int32_t channel_dim_size = dimension[channel_dim_index];
  std::vector<float> min_vals(channel_dim_size);
  std::vector<float> max_vals(channel_dim_size);
  std::vector<bool> has_min_max_value(channel_dim_size, false);
  int indices[4];
  RuntimeShape tensor_dims{dimension[0], dimension[1], dimension[2],
                           dimension[3]};

  // Compute min max ranges per channel
  for (indices[0] = 0; indices[0] < dimension[0]; indices[0]++) {
    for (indices[1] = 0; indices[1] < dimension[1]; indices[1]++) {
      for (indices[2] = 0; indices[2] < dimension[2]; indices[2]++) {
        for (indices[3] = 0; indices[3] < dimension[3]; indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          const float val = input[Offset(tensor_dims, indices)];
          if (has_min_max_value[channel_idx]) {
            if (min_vals[channel_idx] > val) {
              min_vals[channel_idx] = val;
            } else if (max_vals[channel_idx] < val) {
              max_vals[channel_idx] = val;
            }
          } else {
            min_vals[channel_idx] = val;
            max_vals[channel_idx] = val;
            has_min_max_value[channel_idx] = true;
          }
        }
      }
    }
  }

  // Calculate scales per channel
  std::vector<float> scale_invs(channel_dim_size);
  const float half_scale = kMaxQuantizedValue;
  for (size_t channel_idx = 0; channel_idx < channel_dim_size; channel_idx++) {
    const float half_range = std::max(std::abs(min_vals[channel_idx]),
                                      std::abs(max_vals[channel_idx]));
    output_scales->at(channel_idx) = half_range / half_scale;
    if (half_range == 0) {
      scale_invs[channel_idx] = 0;
    } else {
      scale_invs[channel_idx] = half_scale / half_range;
    }
  }

  // Quantize the values.
  const uint64_t num_elements_per_channel =
      output_scales->size() / channel_dim_size;
  std::vector<int8_t> quantized_buffer(num_elements_per_channel);
  memset(indices, 0, 4 * sizeof(int));
  for (indices[0] = 0; indices[0] < dimension[0]; indices[0]++) {
    for (indices[1] = 0; indices[1] < dimension[1]; indices[1]++) {
      for (indices[2] = 0; indices[2] < dimension[2]; indices[2]++) {
        for (indices[3] = 0; indices[3] < dimension[3]; indices[3]++) {
          int channel_idx = indices[channel_dim_index];
          int index = Offset(tensor_dims, indices);
          const float val = input[index];
          const int32_t quantized_value =
              static_cast<int32_t>(TfLiteRound(val * scale_invs[channel_idx]));
          output_value->at(index) = std::min<int8_t>(
              kMaxQuantizedValue,
              std::max<int8_t>(kMinQuantizedValue, quantized_value));
        }
      }
    }
  }
}

}  // namespace utils
}  // namespace optimize
}  // namespace tflite
