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
#include "tensorflow/lite/delegates/hexagon/builders/reduce_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ReduceOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  // Input data tensor.
  int tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  ComputeMinAndMaxQuantValues(input_tensor, &input_min_, &input_max_);
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_min_), sizeof(input_min_));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_max_), sizeof(input_max_));

  // Min/max values for input tensor.
  AddInput(TensorID(input_min_const->GetID(), 0));
  AddInput(TensorID(input_max_const->GetID(), 0));

  // Axes tensor should be constant.
  tensor_id = inputs->data[1];
  const auto& axes_tensor = context->tensors[tensor_id];
  if (axes_tensor.allocation_type == kTfLiteMmapRo) {
    // If the axes input is a constant, bake it into the Hexagon graph as a
    // Const node.
    auto* const_axes_node =
        graph_builder_->AddConstNodeWithData(tensor_id, axes_tensor);
    AddInput(TensorID(const_axes_node->GetID(), 0));
  } else {
    context->ReportError(context, "Reduction op doesn't have constant axis");
    return kTfLiteError;
  }

  auto& output_tensor = context->tensors[outputs->data[0]];
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_tensor.dims);

  float output_min = -1, output_max = -1;
  ComputeMinAndMaxQuantValues(output_tensor, &output_min, &output_max);
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  // Min/max values for output tensor.
  AddInput(TensorID(output_min_const->GetID(), 0));
  AddInput(TensorID(output_max_const->GetID(), 0));

  // Add outputs
  size_t output_element_size = 0;
  TF_LITE_ENSURE_STATUS(
      GetSizeOfType(context, output_tensor.type, &output_element_size));
  auto mean_output = AddOutput(output_element_size, 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
  auto mean_out_min = AddOutput(output_element_size, 4, kScalarShape);
  auto mean_out_max = AddOutput(output_element_size, 4, kScalarShape);
  // Mean op doesn't honor the passed min/max for output, so we need
  // to add requantize.
  auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
  requantize_op->SetOpType(OP_Requantize_8to8);
  requantize_op->AddInput(mean_output);
  requantize_op->AddInput(mean_out_min);
  requantize_op->AddInput(mean_out_max);
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

TfLiteStatus ReduceOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

ReduceOpBuilder::~ReduceOpBuilder() {}

OpBuilder* CreateReduceBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ReduceOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
