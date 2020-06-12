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
#include "tensorflow/lite/delegates/hexagon/builders/arg_min_max_builder.h"

#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ArgMinMaxOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                                  const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
  if (inputs->size != 2) {
    context->ReportError(context, "Expecting 2 inputs %d != 2\n", inputs->size);
    return kTfLiteError;
  }

  // Input data tensor.
  int input_tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[input_tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(input_tensor_id));

  // Axis tensor.
  const int axis_tensor_id = inputs->data[1];
  const auto& axis = context->tensors[axis_tensor_id];
  if (axis.allocation_type != kTfLiteMmapRo) {
    context->ReportError(context,
                         "Axis tensor doesn't have correct allocation type: %s",
                         axis.name);
    return kTfLiteError;
  }

  int axis_value = axis.data.i32[0];
  if (axis_value < 0) {
    axis_value += input_tensor.dims->size;
  }
  auto* input_axis_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&axis_value), sizeof(int));
  AddInput(TensorID(input_axis_const->GetID(), 0));

  // Compute Min/Max
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(input_tensor, &input_min_, &input_max_));
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_min_), sizeof(input_min_));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_max_), sizeof(input_max_));

  AddInput(TensorID(input_min_const->GetID(), 0));
  AddInput(TensorID(input_max_const->GetID(), 0));

  // Output Node
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  size_t output_element_size = 0;
  TF_LITE_ENSURE_STATUS(GetSizeOfType(
      context, context->tensors[outputs->data[0]].type, &output_element_size));
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(output_element_size, 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});

  return kTfLiteOk;
}

TfLiteStatus ArgMinMaxOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                 TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

ArgMinMaxOpBuilder::~ArgMinMaxOpBuilder() {}

OpBuilder* CreateArgMinMaxOpBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ArgMinMaxOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
