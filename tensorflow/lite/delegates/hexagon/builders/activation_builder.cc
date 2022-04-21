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
#include "tensorflow/lite/delegates/hexagon/builders/activation_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ActivationOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // Input data tensor.
  int tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input_tensor));

  if (op_node_.op_type == OP_QuantizedReluX_8) {
    auto* relu_value_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&relu_value_),
        sizeof(relu_value_));
    AddInput(TensorID(relu_value_const->GetID(), 0));
  }

  // Hexagon outputs for this node.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  auto activation_output = AddOutput(sizeof(uint8_t), 4,
                                     {output_batch_size, output_height_size,
                                      output_width_size, output_depth_size});
  auto activation_output_min = AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  auto activation_output_max = AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  float output_min = -1, output_max = -1;
  // Output min/max for requantization.
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min, &output_max));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, (char*)&output_min, sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, (char*)&output_max, sizeof(output_max));

  auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
  requantize_op->SetOpType(OP_Requantize_8to8);
  requantize_op->AddInput(activation_output);
  requantize_op->AddInput(activation_output_min);
  requantize_op->AddInput(activation_output_max);
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

TfLiteStatus ActivationOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

ActivationOpBuilder::~ActivationOpBuilder() {}

OpBuilder* CreateActivationBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ActivationOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
