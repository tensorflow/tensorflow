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
#include <vector>

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/core/api/error_reporter.h"
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

// Populates the single total max and min values for a tensor.
void FillSingleMinMax(const float* const input, const uint64_t input_size,
                      QuantizationParametersT* quantization_params);

// Populates the max and min values for per channel quantization.
TfLiteStatus FillPerChannelMinMax(const float* const input,
                                  const std::vector<int>& dimension,
                                  int32_t channel_dim_index,
                                  QuantizationParametersT* quantization_params,
                                  ErrorReporter* error_reporter);

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
TfLiteStatus SymmetricPerChannelQuantization(TensorT* tensor,
                                             const float* const input,
                                             int32_t channel_dim_index,
                                             std::vector<float>* output_scales,
                                             std::vector<int8_t>* output_value,
                                             ErrorReporter* error_reporter);

// Quantize the values given an array of scales.
void SymmetricPerChannelQuantizeValues(const float* const input,
                                       const std::vector<float>& scales_inv,
                                       const std::vector<int32_t>& dimension,
                                       int32_t channel_dim_index,
                                       std::vector<int8_t>* output_value,
                                       TfLiteType type = kTfLiteNoType);

// Quantizes tensor using symmetric quantization with the min and max elements
// of the tensor.
TfLiteStatus SymmetricQuantizeTensor(ModelT* model, TensorT* tensor);

// Quantizes tensor to float16.
TfLiteStatus QuantizeTensorFloat16(ModelT* model, TensorT* tensor);

// Add quantization parameters.
TfLiteStatus AddQuantizationParams(const std::vector<float>& scales,
                                   const std::vector<int64_t>& zero_point,
                                   int quantized_dimension,
                                   const uint8_t* buffer_data,
                                   size_t buffer_size, TensorType output_type,
                                   ModelT* model, TensorT* tensor,
                                   ErrorReporter* error_reporter);

// Populates the scales vector based on max and min values of quant_params
TfLiteStatus GetSymmetricScalesFromMaxMin(QuantizationParametersT* quant_params,
                                          std::vector<float>* scales,
                                          ErrorReporter* error_reporter);

// Adjusts scale of weights if incompatible with bias scale and likely to
// cause overflow.
TfLiteStatus AdjustWeightsForBiasScale(QuantizationParametersT* quant_params,
                                       const float* bias_data,
                                       const size_t bias_size,
                                       const float input_scale,
                                       ErrorReporter* error_reporter);

// Quantizes tensor with per channel.
TfLiteStatus SymmetricQuantizeTensorPerChannel(ModelT* model, TensorT* tensor,
                                               int32_t channel_dim_index,
                                               ErrorReporter* error_reporter);

// Symmetrically quantizes float to 16bits.
TfLiteStatus SymmetricQuantizeFloatsToInt16(ModelT* model, TensorT* tensor,
                                            float scaling_factor,
                                            ErrorReporter* error_reporter);

std::vector<int16_t> SymmetricQuantizeFloatsToInt16(const float* data,
                                                    uint64_t num_elements,
                                                    float scaling_factor);

// Symmetrically quantizes the bias for per-layer ops (i.e. FullyConnected).
template <typename BiasType>
TfLiteStatus SymmetricPerLayerBiasQuantize(ModelT* model, TensorT* tensor,
                                           float scaling_factor,
                                           ErrorReporter* error_reporter);

// Symmetrically quantizes the bias for ops like Conv and DepthwiseConv.
// The scale of bias if weight_per_channel_scale[channel] * input_scale.
template <typename BiasType>
TfLiteStatus SymmetricPerChannelBiasQuantize(ModelT* model, TensorT* tensor,
                                             float input_scale,
                                             const float* weight_scales,
                                             int number_of_dimension,
                                             ErrorReporter* error_reporter);

template <typename BiasType>
std::vector<BiasType> SymmetricBiasQuantize(const float* data,
                                            uint64_t num_elements,
                                            const std::vector<float>& scales);

// Quantize weight with or without per channel.
TfLiteStatus QuantizeWeight(ModelT* model, TensorT* tensor, bool per_channel,
                            int per_axis_index, ErrorReporter* error_reporter);

// Get effective scale by combining input scale, intermediate scale and factors.
float GetEffectiveScale(ModelT* model, SubGraphT* subgraph, int op_idx,
                        std::vector<int> input_index,
                        std::vector<int> intermediate_index,
                        std::vector<float> factors);

// Return quantization parameters depending on activations type.
TfLiteStatus GetQuantizationParams(TensorT* tensor, TensorType activations_type,
                                   QuantizationParametersT* quantization_params,
                                   ErrorReporter* error_reporter);

// Quantize activation.
TfLiteStatus QuantizeActivation(TensorT* tensor, TensorType activations_type,
                                ErrorReporter* error_reporter);

// Quantize activation to 16bit.
TfLiteStatus QuantizeActivationToInt16(TensorT* tensor, float scale);

// Get the power of two scale for min and max for symmetric quantization case.
int GetPowerOfTwoScale(float min, float max);

}  // namespace utils
}  // namespace optimize
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_OPTIMIZE_QUANTIZATION_UTILS_H_
