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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/min_max_builder.h"

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus MinMaxOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  static int scalar_shape[] = {1, 1, 1, 1};
  int a_tensor_id;
  int b_tensor_id;

  // Input tensors a and b.
  a_tensor_id = inputs->data[0];
  b_tensor_id = inputs->data[1];
  const auto& a_tensor = context->tensors[a_tensor_id];
  const auto& b_tensor = context->tensors[b_tensor_id];
  if (a_tensor.allocation_type == kTfLiteMmapRo)
    graph_builder_->AddConstNodeWithData(a_tensor_id, a_tensor);
  if (b_tensor.allocation_type == kTfLiteMmapRo)
    graph_builder_->AddConstNodeWithData(b_tensor_id, b_tensor);
  AddInput(graph_builder_->GetHexagonTensorId(a_tensor_id));
  AddInput(graph_builder_->GetHexagonTensorId(b_tensor_id));

  // Add Inputs A & B min/max
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(a_tensor, &a_input_min_, &a_input_max_));
  auto* a_input_min_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&a_input_min_),
      sizeof(a_input_min_));
  auto* a_input_max_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&a_input_max_),
      sizeof(a_input_max_));
  AddInput(TensorID(a_input_min_const->GetID(), 0));
  AddInput(TensorID(a_input_max_const->GetID(), 0));

  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(b_tensor, &b_input_min_, &b_input_max_));
  auto* b_input_min_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&b_input_min_),
      sizeof(b_input_min_));
  auto* b_input_max_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&b_input_max_),
      sizeof(b_input_max_));
  AddInput(TensorID(b_input_min_const->GetID(), 0));
  AddInput(TensorID(b_input_max_const->GetID(), 0));

  // Add output min/max
  const int output_tensor_id = outputs->data[0];
  const auto& output_tensor = context->tensors[output_tensor_id];
  float output_min, output_max;
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(output_tensor, &output_min, &output_max));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  AddInput(TensorID(output_min_const->GetID(), 0));
  AddInput(TensorID(output_max_const->GetID(), 0));

  // Add outputs.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  return kTfLiteOk;
}

TfLiteStatus MinMaxOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

OpBuilder* CreateMinMaxBuilder(GraphBuilder* graph_builder, int op_type) {
  return new MinMaxOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
