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
#ifndef TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_

#include <cstdint>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace utils {

// Returns the number of elements in the given tensor.
TfLiteStatus NumElements(const TensorT& tensor, uint64_t* num_elements);

// Populates the scale and zero point for quantization parameters.
//
// Nudges min and max so that floating point 0 falls exactly on a quantized
// value, returning the nudges scale and zero_point.
void GetAsymmetricQuantizationParams(
    float min, float max, const int quant_min, const int quant_max,
    QuantizationParametersT* quantization_params);

// Per-channel quantize a tensor at the given index and returns both scales and
// quantized values.
// Parameters:
// - input is the float input data to be quantized.
// - dimension is the dimension of the input data. Only supports dimension of
//   size 4.
// - channel_dim_index is the channel index within "dimension".
//   dimension[channel_dim_index] gives the number of channels.
// - output_scale is the output scale, the size of which equals the number of
//   channels.
// - output_value is the output data, the size of which equals the number of
//   inputs.
void SymmetricPerChannelQuantization(const float* const input,
                                     const std::vector<int>& dimension,
                                     int32_t channel_dim_index,
                                     std::vector<float>* output_scales,
                                     std::vector<int8_t>* output_value);

// Quantize the values given an array of scales.
void SymmetricPerChannelQuantizeValues(const float* const input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value);

// Quantizes tensor using symmetric quantization with the min and max elements
// of the tensor.
TfLiteStatus SymmetricQuantizeTensor(ModelT* model, TensorT* tensor);

// Add quantization parameters.
TfLiteStatus AddQuantizationParams(const std::vector<float>& scales,
                                   const std::vector<int64_t>& zero_point,
                                   int quantized_dimension,
                                   const uint8_t* buffer_data,
                                   size_t buffer_size, TensorType output_type,
                                   ModelT* model, TensorT* tensor);

// Quantize tensor with per channel.
TfLiteStatus SymmetricQuantizeTensorPerChannel(ModelT* model, TensorT* tensor,
                                               int32_t channel_dim_index);

// Symmetrically quantized the bias for per-layer ops (i.e. FullyConnected).
TfLiteStatus SymmetricPerLayerBiasQuantize(ModelT* model, TensorT* tensor,
                                           float input_scale,
                                           float weight_scale);

// Symmetrically quantizes the bias for ops like Conv and DepthwiseConv.
// The scale of bias if weight_per_channel_scale[channel] * input_scale
TfLiteStatus SymmetricPerChannelBiasQuantize(ModelT* model, TensorT* tensor,
                                             float input_scale,
                                             const float* weight_scales,
                                             int number_of_dimension,
                                             int dimension_index);

// Quantize weight with or without per channel.
TfLiteStatus QuantizeWeight(ModelT* model, TensorT* tensor, bool per_channel,
                            int per_axis_index);

// Quantize activation.
void QuantizeActivation(TensorT* tensor);

}  // namespace utils
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
