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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/arithmetic_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ArithmeticOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  static int quant_bound_shape[] = {1, 1, 1, 1};
  int tensor_id;

  // First input data tensor.
  tensor_id = inputs->data[0];
  const auto& input1_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(input1_tensor, &input1_min_, &input1_max_,
                                  std::numeric_limits<uint8_t>::min(),
                                  std::numeric_limits<uint8_t>::max()));
  auto* input1_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input1_min_),
      sizeof(input1_min_));
  auto* input1_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input1_max_),
      sizeof(input1_max_));

  // Second input data tensor.
  tensor_id = inputs->data[1];
  const auto& input2_tensor = context->tensors[tensor_id];
  // TODO(karimnosseir): Have this as util to generalize to all ops.
  if (input2_tensor.allocation_type == kTfLiteMmapRo) {
    auto* const_input_node =
        graph_builder_->AddConstNodeWithData(tensor_id, input2_tensor);
    graph_builder_->AddTensorWithID(tensor_id, const_input_node->GetID(), 0);
  }
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(input2_tensor, &input2_min_, &input2_max_,
                                  std::numeric_limits<uint8_t>::min(),
                                  std::numeric_limits<uint8_t>::max()));
  auto* input2_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input2_min_),
      sizeof(input2_min_));
  auto* input2_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input2_max_),
      sizeof(input2_max_));

  // Min/max values for input tensors.
  AddInput(TensorID(input1_min_const->GetID(), 0));
  AddInput(TensorID(input1_max_const->GetID(), 0));
  AddInput(TensorID(input2_min_const->GetID(), 0));
  AddInput(TensorID(input2_max_const->GetID(), 0));

  // Output min/max as inputs, only if it's an Add node.
  if (op_node_.op_type == OP_QuantizedAdd_8p8to8) {
    output_min_ = 0;
    output_max_ = 0;
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        context->tensors[outputs->data[0]], &output_min_, &output_max_,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max()));
    if (output_max_ != 0) {
      auto* output_min_const = graph_builder_->AddConstNodeWithData(
          quant_bound_shape, reinterpret_cast<char*>(&output_min_),
          sizeof(output_min_));
      auto* output_max_const = graph_builder_->AddConstNodeWithData(
          quant_bound_shape, reinterpret_cast<char*>(&output_max_),
          sizeof(output_max_));
      AddInput(TensorID(output_min_const->GetID(), 0));
      AddInput(TensorID(output_max_const->GetID(), 0));
    }
  }

  // Hexagon outputs for this node.
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
