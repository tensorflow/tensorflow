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
#include "tensorflow/lite/delegates/hexagon/builders/pool_2d_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus Pool2dOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  // Input data tensor.
  int tensor_id = inputs->data[0];
  const auto& data_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, data_tensor));

  const TfLitePoolParams* pool_params =
      reinterpret_cast<const TfLitePoolParams*>(builtin_data_);

  // Padding type.
  if (pool_params->padding == kTfLitePaddingSame) {
    SetPaddingType(NN_PAD_SAME);
  } else if (pool_params->padding == kTfLitePaddingValid) {
    SetPaddingType(NN_PAD_VALID);
  }

  // Pooling window (filter) width/height as inputs.
  static int dummy = 0;
  filter_shape_ = {1, pool_params->filter_height, pool_params->filter_width, 1};
  auto* filter_node = graph_builder_->AddConstNodeWithData(
      filter_shape_.data(), (char*)&dummy, sizeof(dummy));
  AddInput(TensorID(filter_node->GetID(), 0));
  // Stride width/height as inputs.
  stride_shape_ = {1, pool_params->stride_height, pool_params->stride_width, 1};
  auto* stride_node = graph_builder_->AddConstNodeWithData(
      stride_shape_.data(), (char*)&dummy, sizeof(dummy));
  AddInput(TensorID(stride_node->GetID(), 0));

  // Hexagon outputs for this node.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  if (op_node_.op_type == OP_QuantizedMaxPool_8) {
    node_output_ = AddOutput(sizeof(uint8_t), 4,
                             {output_batch_size, output_height_size,
                              output_width_size, output_depth_size});
    AddOutput(sizeof(float), 4, kScalarShape);
    AddOutput(sizeof(float), 4, kScalarShape);
  } else {
    // Hexagon's AvgPool output has different min/max bounds than what TFLite
    // expects. Therefore, we add a Requantize op to correct the ranges.
    TensorID pool_out = AddOutput(sizeof(uint8_t), 4,
                                  {output_batch_size, output_height_size,
                                   output_width_size, output_depth_size});
    const auto& pool_out_min = AddOutput(sizeof(float), 4, kScalarShape);
    const auto& pool_out_max = AddOutput(sizeof(float), 4, kScalarShape);

    // Output min/max for requantization.
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        context->tensors[outputs->data[0]], &output_min_, &output_max_));
    auto* output_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, (char*)&output_min_, sizeof(output_min_));
    auto* output_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, (char*)&output_max_, sizeof(output_max_));

    auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
    requantize_op->SetOpType(OP_Requantize_8to8);
    requantize_op->AddInput(pool_out);
    requantize_op->AddInput(pool_out_min);
    requantize_op->AddInput(pool_out_max);
    requantize_op->AddInput(TensorID(output_min_const->GetID(), 0));
    requantize_op->AddInput(TensorID(output_max_const->GetID(), 0));
    node_output_ =
        requantize_op->AddOutput(sizeof(uint8_t), 4,
                                 {output_batch_size, output_height_size,
                                  output_width_size, output_depth_size});
    requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
    requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  }

  return kTfLiteOk;
}

TfLiteStatus Pool2dOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

Pool2dOpBuilder::~Pool2dOpBuilder() {}

OpBuilder* CreatePool2DBuilder(GraphBuilder* graph_builder, int op_type) {
  return new Pool2dOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
