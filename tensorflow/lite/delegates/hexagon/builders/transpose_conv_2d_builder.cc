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
#include "tensorflow/lite/delegates/hexagon/builders/transpose_conv_2d_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
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

TfLiteStatus TransposeConv2dOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // DATA TENSOR.
  int tensor_id = inputs->data[2];
  const auto& data_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // WEIGHTS.
  tensor_id = inputs->data[1];
  const auto& weights_tensor = context->tensors[tensor_id];
  if (weights_tensor.allocation_type != kTfLiteMmapRo) {
    TF_LITE_KERNEL_LOG(
        context, "Weights tensor doesn't have correct allocation type: %s",
        weights_tensor.name);
    return kTfLiteError;
  }
  int filter_batch_size, filter_height_size, filter_width_size,
      filter_depth_size;
  GetDims(&filter_batch_size, &filter_height_size, &filter_width_size,
          &filter_depth_size, weights_tensor.dims);
  // Weights tensor could be int8 even for per-tensor quantization.
  // Therefore, we look at the number of scale values to check if it is
  // per-channel quantized.
  TfLiteAffineQuantization* weights_quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(
          weights_tensor.quantization.params);
  const bool is_per_channel_quant = weights_quant_params->scale->size > 1;
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // Handle weights quantization.
  float weights_min = 0;
  float weights_max = 0;
  if (is_per_channel_quant) {
    ProcessPerChannelQuantizedWeights(weights_tensor, context, &weights_min,
                                      &weights_max, graph_builder_,
                                      &per_channel_quant_);
  } else {
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        weights_tensor, &weights_min, &weights_max));
  }
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_min), sizeof(weights_min));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_max), sizeof(weights_max));

  // Min/max inputs for data & weights tensors.
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, data_tensor));
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
  const bool has_bias = inputs->size == 4;
  OpBuilder* bias_const = nullptr;
  OpBuilder* bias_min_const = nullptr;
  OpBuilder* bias_max_const = nullptr;
  if (!has_bias) {
    // If the TFLite node does not have a bias, we simply feed in 0s.
    std::vector<int> bias_data(output_depth_size, 0);
    bias_shape_ = {1, 1, 1, output_depth_size};
    bias_const = graph_builder_->AddConstNodeWithData(
        bias_shape_.data(), reinterpret_cast<char*>(bias_data.data()),
        sizeof(bias_data[0]) * bias_data.size());
    float zero_bound = 0;
    bias_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&zero_bound), sizeof(zero_bound));
    bias_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&zero_bound), sizeof(zero_bound));
  } else {
    const auto& bias_tensor = context->tensors[inputs->data[3]];
    if (bias_tensor.allocation_type != kTfLiteMmapRo) {
      TF_LITE_KERNEL_LOG(context,
                         "Bias tensor doesn't have correct allocation type: %s",
                         bias_tensor.name);
      return kTfLiteError;
    }
    float bias_min = 0;
    float bias_max = 0;
    if (per_channel_quant_.channel_scales_node != nullptr) {
      ProcessPerChannelQuantizedBias(
          data_tensor, bias_tensor, inputs->data[3], context, &bias_min,
          &bias_max, graph_builder_, &per_channel_quant_, &bias_const);
    } else {
      bias_const =
          graph_builder_->AddConstNodeWithData(inputs->data[3], bias_tensor);
      TF_LITE_ENSURE_STATUS(
          ComputeMinAndMaxQuantValues(bias_tensor, &bias_min, &bias_max));
    }

    bias_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&bias_min), sizeof(bias_min));
    bias_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&bias_max), sizeof(bias_max));
  }
  AddInput(TensorID(bias_const->GetID(), 0));
  AddInput(TensorID(bias_min_const->GetID(), 0));
  AddInput(TensorID(bias_max_const->GetID(), 0));

  // Output quantization.
  TF_LITE_ENSURE_STATUS(
      ComputeAndAddMinAndMax(context, context->tensors[outputs->data[0]]));

  // Channel scales, if this op is per-channel quantized.
  if (per_channel_quant_.channel_scales_node != nullptr) {
    AddInput(TensorID(per_channel_quant_.channel_scales_node->GetID(), 0));
  }

  // Hexagon outputs for this node.
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);

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
