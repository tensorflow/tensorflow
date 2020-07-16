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
#include "tensorflow/lite/delegates/hexagon/builders/split_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus SplitOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                              const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  const int input_tensor_id = inputs->data[1];
  const auto& input_tensor = context->tensors[input_tensor_id];

  // Axis tensor.
  const int axis_tensor_id = inputs->data[0];
  const auto& axis = context->tensors[axis_tensor_id];
  if (axis.allocation_type != kTfLiteMmapRo) {
    context->ReportError(context,
                         "Axis tensor doesn't have correct allocation type: %s",
                         axis.name);
    return kTfLiteError;
  }
  // We pad Hexagon tensor dimensions with 1 if dims.size < 4.
  // (4 - input_tensor.dims->size) helps maps the input axis value in such
  // cases.
  int axis_value = axis.data.i32[0] + (4 - input_tensor.dims->size);
  if (axis_value < 0) {
    axis_value += input_tensor.dims->size;
  }
  auto* input_axis_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&axis_value), sizeof(int));
  AddInput(TensorID(input_axis_const->GetID(), 0));

  // Input data tensor & min/max.
  AddInput(graph_builder_->GetHexagonTensorId(input_tensor_id));
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(input_tensor, &input_min_, &input_max_));
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_min_), sizeof(input_min_));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_max_), sizeof(input_max_));
  AddInput(TensorID(input_min_const->GetID(), 0));
  AddInput(TensorID(input_max_const->GetID(), 0));

  // Output data tensors.
  for (int i = 0; i < outputs->size; ++i) {
    int output_batch_size, output_height_size, output_width_size,
        output_depth_size;
    GetDims(&output_batch_size, &output_height_size, &output_width_size,
            &output_depth_size, context->tensors[outputs->data[i]].dims);
    TensorID output = AddOutput(sizeof(uint8_t), 4,
                                {output_batch_size, output_height_size,
                                 output_width_size, output_depth_size});
    node_outputs_.push_back(output);
  }
  // For Hexagon output min/max.
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus SplitOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                             TfLiteContext* context) {
  for (int i = 0; i < node_outputs_.size(); ++i) {
    graph_builder_->AddTensorWithID(outputs->data[i], node_outputs_[i].first,
                                    node_outputs_[i].second);
  }
  return kTfLiteOk;
}

SplitOpBuilder::~SplitOpBuilder() {}

OpBuilder* CreateSplitBuilder(GraphBuilder* graph_builder, int op_type) {
  return new SplitOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
