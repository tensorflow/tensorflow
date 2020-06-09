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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/quantize_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus QuantizeOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                                 const TfLiteIntArray* outputs,
                                                 TfLiteContext* context) {
  // Input.
  float input_min = 0;
  float input_max = 0;
  const auto& input_tensor = context->tensors[inputs->data[0]];
  ComputeMinAndMaxQuantValues(input_tensor, &input_min, &input_max);
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_min), sizeof(input_min));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_max), sizeof(input_max));

  // Output.
  float output_min = 0;
  float output_max = 0;
  const auto& output_tensor = context->tensors[outputs->data[0]];
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(output_tensor, &output_min, &output_max));
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_tensor.dims);
  auto* requantized_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* requantized_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));

  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));
  AddInput(TensorID(input_min_const->GetID(), 0));
  AddInput(TensorID(input_max_const->GetID(), 0));
  AddInput(TensorID(requantized_min_const->GetID(), 0));
  AddInput(TensorID(requantized_max_const->GetID(), 0));

  // Hexagon outputs for this node.
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus QuantizeOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

QuantizeOpBuilder::~QuantizeOpBuilder() {}

OpBuilder* CreateQuantizeBuilder(GraphBuilder* graph_builder, int op_type) {
  return new QuantizeOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
