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
#include "tensorflow/lite/delegates/hexagon/builders/arithmetic_builder.h"

#include <stdint.h>

#include <limits>

#include "hexagon/hexagon_nn_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ArithmeticOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // First input data tensor.
  int tensor_id = inputs->data[0];
  const auto& input1_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // Second input data tensor.
  tensor_id = inputs->data[1];
  const auto& input2_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // Inputs min/max
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input1_tensor));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input2_tensor));

  // Output details.
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min_, &output_max_));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min_), sizeof(output_min_));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max_), sizeof(output_max_));
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  if (op_node_.op_type == OP_QuantizedAdd_8p8to8 && output_max_ != 0) {
    // Hexagon's QuantizedAdd supports output min/max as input.
    AddInput(TensorID(output_min_const->GetID(), 0));
    AddInput(TensorID(output_max_const->GetID(), 0));
  }

  if (op_node_.op_type == OP_QuantizedMul_8x8to32) {
    const auto& math_out = AddOutput(sizeof(int), 4,
                                     {output_batch_size, output_height_size,
                                      output_width_size, output_depth_size});
    const auto& math_out_min = AddOutput(sizeof(float), 4, kScalarShape);
    const auto& math_out_max = AddOutput(sizeof(float), 4, kScalarShape);

    auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
    requantize_op->SetOpType(OP_Requantize_32to8);
    requantize_op->AddInput(math_out);
    requantize_op->AddInput(math_out_min);
    requantize_op->AddInput(math_out_max);
    requantize_op->AddInput(TensorID(output_min_const->GetID(), 0));
    requantize_op->AddInput(TensorID(output_max_const->GetID(), 0));
    node_output_ =
        requantize_op->AddOutput(sizeof(uint8_t), 4,
                                 {output_batch_size, output_height_size,
                                  output_width_size, output_depth_size});
    requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
    requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  } else {
    auto result_out = AddOutput(sizeof(uint8_t), 4,
                                {output_batch_size, output_height_size,
                                 output_width_size, output_depth_size});
    auto result_min = AddOutput(sizeof(float), 4, kScalarShape);
    auto result_max = AddOutput(sizeof(float), 4, kScalarShape);

    auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
    requantize_op->SetOpType(OP_Requantize_8to8);
    requantize_op->AddInput(result_out);
    requantize_op->AddInput(result_min);
    requantize_op->AddInput(result_max);
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

TfLiteStatus ArithmeticOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

ArithmeticOpBuilder::~ArithmeticOpBuilder() {}

OpBuilder* CreateArithmeticBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ArithmeticOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
