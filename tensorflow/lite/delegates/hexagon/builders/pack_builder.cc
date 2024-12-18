/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/delegates/hexagon/builders/pack_builder.h"

#include <stdint.h>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

int GetAxis(int axis, const TfLiteIntArray* inputs, TfLiteContext* context) {
  auto& input_tensor = context->tensors[inputs->data[0]];
  // Handle -ve axis.
  if (axis < 0) {
    axis += input_tensor.dims->size + 1;
  }
  // We need to adjust the axis to be as if the inputs are of rank 4, since
  // we represent tensors in Hexagon of rank 4.
  return (4 - input_tensor.dims->size) + axis - 1;
}

}  // namespace
TfLiteStatus PackOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                             const TfLiteIntArray* outputs,
                                             TfLiteContext* context) {
  auto* params = reinterpret_cast<TfLitePackParams*>(builtin_data_);
  int axis = GetAxis(params->axis, inputs, context);
  // Add axis
  auto* axis_node = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&axis), sizeof(axis));
  AddInput(TensorID(axis_node->GetID(), 0));

  // Add all input tensors.
  minima_.reserve(inputs->size);
  maxima_.reserve(inputs->size);
  int tensor_id = -1;
  float data_min, data_max;
  for (int i = 0; i < inputs->size; ++i) {
    tensor_id = inputs->data[i];
    auto& input_tensor = context->tensors[tensor_id];
    AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
    TF_LITE_ENSURE_STATUS(
        ComputeMinAndMaxQuantValues(input_tensor, &data_min, &data_max));
    minima_.push_back(data_min);
    maxima_.push_back(data_max);
  }

  // Minima tensors.
  for (int i = 0; i < minima_.size(); ++i) {
    auto* data_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&minima_[i]), sizeof(minima_[i]));
    AddInput(TensorID(data_min_const->GetID(), 0));
  }

  // Maxima tensors.
  for (int i = 0; i < maxima_.size(); ++i) {
    auto* data_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&maxima_[i]), sizeof(maxima_[i]));
    AddInput(TensorID(data_max_const->GetID(), 0));
  }

  // Hexagon outputs for this node.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  TensorID pack_out = AddOutput(sizeof(uint8_t), 4,
                                {output_batch_size, output_height_size,
                                 output_width_size, output_depth_size});

  // Output min/max for requantization.
  float output_min, output_max;
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min, &output_max));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));

  const auto& pack_out_min = AddOutput(sizeof(float), 4, kScalarShape);
  const auto& pack_out_max = AddOutput(sizeof(float), 4, kScalarShape);

  // Requantize output to the expected min/max.
  auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
  requantize_op->SetOpType(OP_Requantize_8to8);
  requantize_op->AddInput(pack_out);
  requantize_op->AddInput(pack_out_min);
  requantize_op->AddInput(pack_out_max);
  requantize_op->AddInput(TensorID(output_min_const->GetID(), 0));
  requantize_op->AddInput(TensorID(output_max_const->GetID(), 0));
  node_output_ =
      requantize_op->AddOutput(sizeof(uint8_t), 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
  requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  return kTfLiteOk;
}

TfLiteStatus PackOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                            TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

OpBuilder* CreatePackBuilder(GraphBuilder* graph_builder, int op_type) {
  return new PackOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
