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
// This file is the MLIR copy of part of
// third_party/tensorflow/lite/tools/optimize/quantization_utils.h as part of
// the effort to decouple TFLite from MLIR.

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_QUANTIZATION_UTILS_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_QUANTIZATION_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/status/status.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"

namespace mlir {
namespace lite {
namespace toco_legacy {

using tflite::ModelT;
using tflite::QuantizationParametersT;
using tflite::TensorT;
using tflite::TensorType;

// LINT.IfChange(num_elements)
// Returns the number of elements in the given tensor.
absl::Status NumElements(const TensorT& tensor, uint64_t* num_elements);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:num_elements)

// LINT.IfChange(fill_per_channel_min_max)
// Populates the max and min values for per channel quantization.
absl::Status FillPerChannelMinMax(const float* input,
                                  const std::vector<int32_t>& dimension,
                                  int32_t channel_dim_index,
                                  QuantizationParametersT* quantization_params);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:fill_per_channel_min_max)

// LINT.IfChange(symmetric_per_channel_quantization)
// Per-channel quantize a tensor at the given index and returns both scales and
// quantized values.
// Parameters:
// - tensor is the tensor to be quantized, needed to access associated
//   quantization parameters
// - input is the float input data to be quantized.
// - channel_dim_index is the channel index within "dimension".
//   dimension[channel_dim_index] gives the number of channels.
// - output_scale is the output scale, the size of which equals the number of
//   channels.
// - output_value is the output data, the size of which equals the number of
//   inputs.
absl::Status SymmetricPerChannelQuantization(TensorT* tensor,
                                             const float* input,
                                             int32_t channel_dim_index,
                                             std::vector<float>* output_scales,
                                             std::vector<int8_t>* output_value);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:symmetric_per_channel_quantization)

// LINT.IfChange(symmetric_per_channel_quantize_values)
// Quantize the values given an array of scales.
void SymmetricPerChannelQuantizeValues(const float* input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:symmetric_per_channel_quantize_values)

// LINT.IfChange(symmetric_quantize_tensor)
// Quantizes tensor using symmetric quantization with the min and max elements
// of the tensor.
absl::Status SymmetricQuantizeTensor(ModelT* model, TensorT* tensor);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:symmetric_quantize_tensor)

// LINT.IfChange(symmetric_quantize_tensor_per_channel)
// Quantizes tensor with per channel.
absl::Status SymmetricQuantizeTensorPerChannel(ModelT* model, TensorT* tensor,
                                               int32_t channel_dim_index);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:symmetric_quantize_tensor_per_channel)

// LINT.IfChange(quantize_tensor_float16)
// Quantizes tensor to float16.
absl::Status QuantizeTensorFloat16(ModelT* model, TensorT* tensor);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:quantize_tensor_float16)

// LINT.IfChange(add_quantization_params)
absl::Status AddQuantizationParams(const std::vector<float>& scales,
                                   const std::vector<int64_t>& zero_point,
                                   int quantized_dimension,
                                   const uint8_t* buffer_data,
                                   size_t buffer_size, TensorType output_type,
                                   ModelT* model, TensorT* tensor);
// LINT.ThenChange(//tensorflow/lite/tools/optimize/quantization_utils.h:add_quantization_params)

}  // namespace toco_legacy
}  // namespace lite
}  // namespace mlir

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_QUANTIZATION_LITE_TOCO_LEGACY_QUANTIZATION_UTILS_H_
