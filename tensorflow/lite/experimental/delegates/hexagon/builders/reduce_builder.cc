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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/reduce_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ReduceOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  static int quant_bound_shape[] = {1, 1, 1, 1};
  int tensor_id;

  // Input data tensor.
  tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  ComputeMinAndMaxQuantValues(input_tensor, &input_min_, &input_max_,
                              std::numeric_limits<uint8_t>::min(),
                              std::numeric_limits<uint8_t>::max());
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input_min_),
      sizeof(input_min_));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&input_max_),
      sizeof(input_max_));

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

  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  // Hexagon's sum-reduction outputs int32, so we shrink it down to UInt8.
  if (op_node_.op_type == OP_QuantizedSum_8to32) {
    const auto& reduce_out = AddOutput(sizeof(int32_t), 4,
                                       {output_batch_size, output_height_size,
                                        output_width_size, output_depth_size});
    const auto& reduce_out_min = AddOutput(sizeof(float), 4, {1, 1, 1, 1});
    const auto& reduce_out_max = AddOutput(sizeof(float), 4, {1, 1, 1, 1});

    auto* quantize_output_op = graph_builder_->AddNode();
    quantize_output_op->SetOpType(OP_QuantizeDownAndShrinkRange_32to8);
    quantize_output_op->AddInput(reduce_out);
    quantize_output_op->AddInput(reduce_out_min);
    quantize_output_op->AddInput(reduce_out_max);
    node_output_ =
        quantize_output_op->AddOutput(sizeof(uint8_t), 4,
                                      {output_batch_size, output_height_size,
                                       output_width_size, output_depth_size});
    quantize_output_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
    quantize_output_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  } else {
    node_output_ = AddOutput(sizeof(uint8_t), 4,
                             {output_batch_size, output_height_size,
                              output_width_size, output_depth_size});
    AddOutput(sizeof(float), 4, {1, 1, 1, 1});
    AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  }

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
