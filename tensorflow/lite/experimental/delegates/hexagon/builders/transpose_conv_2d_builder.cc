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
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus TransposeConv2dOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  static std::vector<int> quant_bound_shape = {1, 1, 1, 1};
  int tensor_id;

  // Input data tensor.
  tensor_id = inputs->data[2];
  const auto& data_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      data_tensor, &data_min_, &data_max_, std::numeric_limits<uint8_t>::min(),
      std::numeric_limits<uint8_t>::max()));
  auto* data_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&data_min_, sizeof(data_min_));
  auto* data_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&data_max_, sizeof(data_max_));

  // Weights tensor
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
  auto* const_weights_node = graph_builder_->AddConstNodeWithData(
      weight_shape_.data(), (char*)weights_tensor.data.raw,
      weights_tensor.bytes);
  graph_builder_->AddTensorWithID(tensor_id, const_weights_node->GetID(), 0);
  AddInput(TensorID(const_weights_node->GetID(), 0));
  ComputeMinAndMaxQuantValues(weights_tensor, &weights_min_, &weights_max_,
                              std::numeric_limits<uint8_t>::min(),
                              std::numeric_limits<uint8_t>::max());
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&weights_min_, sizeof(weights_min_));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&weights_max_, sizeof(weights_max_));

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
      padding_tensor_shape.data(), (char*)padding_tensor.data(),
      (sizeof(int) * 4));
  AddInput(TensorID(padding_const->GetID(), 0));

  // Stride shape.
  int stride_height = params->stride_height;
  int stride_width = params->stride_width;
  static int dummy = 0;
  stride_shape_ = {1, stride_height, stride_width, 1};
  auto* stride_node = graph_builder_->AddConstNodeWithData(
      stride_shape_.data(), (char*)&dummy, sizeof(dummy));
  AddInput(TensorID(stride_node->GetID(), 0));

  // TFLite's TransposeConv doesn't have a bias input, so we just feed in 0s.
  std::vector<int> bias_data(output_depth_size);
  // Hexagon's conv ops require bias as a [1, 1, 1, dout] tensor.
  bias_shape_ = {1, 1, 1, output_depth_size};
  auto* bias_const = graph_builder_->AddConstNodeWithData(
      bias_shape_.data(), (char*)bias_data.data(),
      sizeof(bias_data[0]) * bias_data.size());
  bias_min_ = 0;
  bias_max_ = 0;
  auto* bias_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&bias_min_, sizeof(bias_min_));
  auto* bias_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&bias_max_, sizeof(bias_max_));
  AddInput(TensorID(bias_const->GetID(), 0));
  AddInput(TensorID(bias_min_const->GetID(), 0));
  AddInput(TensorID(bias_max_const->GetID(), 0));

  // Output min/max.
  ComputeMinAndMaxQuantValues(context->tensors[outputs->data[0]], &output_min_,
                              &output_max_, std::numeric_limits<uint8_t>::min(),
                              std::numeric_limits<uint8_t>::max());
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&output_min_, sizeof(output_min_));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape.data(), (char*)&output_max_, sizeof(output_max_));
  AddInput(TensorID(output_min_const->GetID(), 0));
  AddInput(TensorID(output_max_const->GetID(), 0));

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
