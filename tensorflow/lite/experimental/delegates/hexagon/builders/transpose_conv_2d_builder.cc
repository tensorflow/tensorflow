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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/transpose_conv_2d_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

constexpr uint8_t k8BitSignFlipConstant = 0x80;
// 1/1024 ~ 0.0009766 is a restriction set by Hexagon's kernels.
// TODO(b/151103818): Figure out a way to retrieve this constant reliably.
constexpr float kHexagonMinRelativeScale = 0.0009766f;

}  // namespace

TfLiteStatus TransposeConv2dOpBuilder::ProcessPerChannelQuantizedWeights(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context, float* weights_min, float* weights_max) {
  const auto& weights_tensor = context->tensors[inputs->data[1]];
  TfLiteAffineQuantization* weights_quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(
          weights_tensor.quantization.params);

  // Retrieve channel scales.
  num_scale_values_ = weights_quant_params->scale->size;
  // Normalize the scales as expected by Hexagon.
  scales_data_ = weights_quant_params->scale->data;
  std::vector<float> normalized_scales;
  normalized_scales.reserve(num_scale_values_);
  float scale_max = 0.0;
  for (int i = 0; i < num_scale_values_; ++i) {
    normalized_scales.push_back(scales_data_[i]);
    if (scales_data_[i] > scale_max) {
      scale_max = scales_data_[i];
    }
  }
  if (scale_max == 0.0) {
    TF_LITE_KERNEL_LOG(context, "Scale max is zero for: %s",
                       weights_tensor.name);
    return kTfLiteError;
  }
  for (int i = 0; i < num_scale_values_; ++i) {
    normalized_scales[i] =
        std::max(normalized_scales[i] / scale_max, kHexagonMinRelativeScale);
  }
  // Add node for channel scales data.
  const std::vector<int> scales_shape = {1, 1, 1, num_scale_values_};
  channel_scales_node_ = graph_builder_->AddConstNodeWithData(
      scales_shape.data(), reinterpret_cast<char*>(normalized_scales.data()),
      normalized_scales.size() * sizeof(normalized_scales[0]));
  *weights_min = -128 * scale_max;
  *weights_max = 127 * scale_max;
  return kTfLiteOk;
}

TfLiteStatus TransposeConv2dOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  static std::vector<int> quant_bound_shape = {1, 1, 1, 1};
  int tensor_id;

  // DATA TENSOR.
  tensor_id = inputs->data[2];
  const auto& data_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  float data_min = 0;
  float data_max = 0;
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(data_tensor, &data_min, &data_max));
  auto* data_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&data_min),
      sizeof(data_min));
  auto* data_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&data_max),
      sizeof(data_max));

  // WEIGHTS.
  tensor_id = inputs->data[1];
  const auto& weights_tensor = context->tensors[tensor_id];
  if (weights_tensor.allocation_type != kTfLiteMmapRo) {
    context->ReportError(
        context, "Weights tensor doesn't have correct allocation type: %s",
        weights_tensor.name);
    return kTfLiteError;
  }
  int filter_batch_size, filter_height_size, filter_width_size,
      filter_depth_size;
  GetDims(&filter_batch_size, &filter_height_size, &filter_width_size,
          &filter_depth_size, weights_tensor.dims);
  weight_shape_ = {filter_batch_size, filter_height_size, filter_width_size,
                   filter_depth_size};
  // Weights tensor could be int8 even for per-tensor quantization.
  // Therefore, we look at the number of scale values to check if it is
  // per-channel quantized.
  TfLiteAffineQuantization* weights_quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(
          weights_tensor.quantization.params);
  const bool is_per_channel_quant = weights_quant_params->scale->size > 1;

  OpBuilder* const_weights_node;
  if (weights_tensor.type == kTfLiteInt8) {
    std::vector<uint8_t> weights_data(NumElements(&weights_tensor));
    const int8_t* original_data = weights_tensor.data.int8;
    // Flip bits on the weight values so that the int8 values are treated
    // as uint8.
    for (int i = 0; i < NumElements(&weights_tensor); ++i) {
      weights_data[i] = original_data[i] ^ k8BitSignFlipConstant;
    }
    const_weights_node = graph_builder_->AddConstNodeWithData(
        weight_shape_.data(), reinterpret_cast<char*>(weights_data.data()),
        weights_data.size() * sizeof(weights_data[0]));
  } else {
    const_weights_node = graph_builder_->AddConstNodeWithData(
        weight_shape_.data(), weights_tensor.data.raw, weights_tensor.bytes);
  }
  graph_builder_->AddTensorWithID(tensor_id, const_weights_node->GetID(), 0);
  AddInput(TensorID(const_weights_node->GetID(), 0));

  // Handle weights quantization.
  float weights_min = 0;
  float weights_max = 0;
  if (is_per_channel_quant) {
    ProcessPerChannelQuantizedWeights(inputs, outputs, context, &weights_min,
                                      &weights_max);
  } else {
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        weights_tensor, &weights_min, &weights_max));
  }
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&weights_min),
      sizeof(weights_min));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&weights_max),
      sizeof(weights_max));

  // Min/max inputs for data & weights tensors.
  AddInput(TensorID(data_min_const->GetID(), 0));
  AddInput(TensorID(data_max_const->GetID(), 0));
  AddInput(TensorID(weights_min_const->GetID(), 0));
  AddInput(TensorID(weights_max_const->GetID(), 0));

  // Output dims are required to compute padding.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  // PADDING & STRIDE.
  // Hexagon TransposeConv requires an explicit padding tensor. So we compute
  // the same using stride, input & output info.
  const TfLiteTransposeConvParams* params =
      reinterpret_cast<const TfLiteTransposeConvParams*>(builtin_data_);
  int unused_output_height, unused_output_width;
  TfLitePaddingValues padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, output_height_size,
      output_width_size, filter_height_size, filter_width_size, params->padding,
      &unused_output_height, &unused_output_width);
  std::vector<int> padding_tensor = {padding.height, padding.height,
                                     padding.width, padding.width};
  std::vector<int> padding_tensor_shape = {1, 1, 2, 2};
  auto* padding_const = graph_builder_->AddConstNodeWithData(
      padding_tensor_shape.data(),
      reinterpret_cast<char*>(padding_tensor.data()), (sizeof(int) * 4));
  AddInput(TensorID(padding_const->GetID(), 0));

  // Stride shape.
  int stride_height = params->stride_height;
  int stride_width = params->stride_width;
  static int dummy = 0;
  stride_shape_ = {1, stride_height, stride_width, 1};
  auto* stride_node = graph_builder_->AddConstNodeWithData(
      stride_shape_.data(), reinterpret_cast<char*>(&dummy), sizeof(dummy));
  AddInput(TensorID(stride_node->GetID(), 0));

  // BIAS.
  // TFLite's TransposeConv doesn't have a bias input, so we just feed in 0s.
  std::vector<int> bias_data(output_depth_size, 0);
  // Hexagon's conv ops require bias as a [1, 1, 1, dout] tensor.
  bias_shape_ = {1, 1, 1, output_depth_size};
  auto* bias_const = graph_builder_->AddConstNodeWithData(
      bias_shape_.data(), reinterpret_cast<char*>(bias_data.data()),
      sizeof(bias_data[0]) * bias_data.size());
  float zero_bound = 0;
  auto* bias_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&zero_bound),
      sizeof(zero_bound));
  auto* bias_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&zero_bound),
      sizeof(zero_bound));
  AddInput(TensorID(bias_const->GetID(), 0));
  AddInput(TensorID(bias_min_const->GetID(), 0));
  AddInput(TensorID(bias_max_const->GetID(), 0));

  // Output quantization.
  float output_min = 0;
  float output_max = 0;
  ComputeMinAndMaxQuantValues(context->tensors[outputs->data[0]], &output_min,
                              &output_max);
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&output_min),
      sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), reinterpret_cast<char*>(&output_max),
      sizeof(output_max));
  AddInput(TensorID(output_min_const->GetID(), 0));
  AddInput(TensorID(output_max_const->GetID(), 0));

  // Channel scales, if this op is per-channel quantized.
  if (channel_scales_node_ != nullptr) {
    AddInput(TensorID(channel_scales_node_->GetID(), 0));
  }

  // Hexagon outputs for this node.
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  return kTfLiteOk;
}

TfLiteStatus TransposeConv2dOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

TransposeConv2dOpBuilder::~TransposeConv2dOpBuilder() {}

OpBuilder* CreateTransposeConv2DBuilder(GraphBuilder* graph_builder,
                                        int op_type) {
  return new TransposeConv2dOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
