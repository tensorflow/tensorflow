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
#include "tensorflow/lite/delegates/hexagon/builders/strided_slice_builder.h"

#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {}  // namespace

TfLiteStatus StridedSliceOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // Input data tensor.
  const auto& input_tensor = context->tensors[inputs->data[0]];
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));
  // Begin/End/Step.
  const auto& begin_tensor = context->tensors[inputs->data[1]];
  const auto& end_tensor = context->tensors[inputs->data[2]];
  const auto& step_tensor = context->tensors[inputs->data[3]];
  auto begins_node =
      graph_builder_->AddConstNodeWithData(inputs->data[1], begin_tensor);
  auto ends_node =
      graph_builder_->AddConstNodeWithData(inputs->data[2], end_tensor);
  auto steps_node =
      graph_builder_->AddConstNodeWithData(inputs->data[3], step_tensor);
  AddInput(TensorID(begins_node->GetID(), 0));
  AddInput(TensorID(ends_node->GetID(), 0));
  AddInput(TensorID(steps_node->GetID(), 0));
  // Begin/End/Shrink-Axis masks.
  // Hexagon's op always expects bits at 0, 1, 2 & 3 to correspond to BHWD.
  // So we have to left-shift the mask by (4 - begins.size()).
  const TfLiteStridedSliceParams* params =
      reinterpret_cast<const TfLiteStridedSliceParams*>(builtin_data_);
  int begin_mask = params->begin_mask;
  int end_mask = params->end_mask;
  int shrink_axis_mask = params->shrink_axis_mask;
  int original_mask_size = input_tensor.dims->size;
  begin_mask = begin_mask << (4 - original_mask_size);
  end_mask = end_mask << (4 - original_mask_size);
  shrink_axis_mask = shrink_axis_mask << (4 - original_mask_size);
  auto* begin_mask_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&begin_mask), sizeof(begin_mask));
  AddInput(TensorID(begin_mask_const->GetID(), 0));
  auto* end_mask_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&end_mask), sizeof(end_mask));
  AddInput(TensorID(end_mask_const->GetID(), 0));
  auto* shrink_axis_mask_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&shrink_axis_mask),
      sizeof(shrink_axis_mask));
  AddInput(TensorID(shrink_axis_mask_const->GetID(), 0));

  // Input min/max
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input_tensor));

  // Slice outputs.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus StridedSliceOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

OpBuilder* CreateStridedSliceBuilder(GraphBuilder* graph_builder, int op_type) {
  return new StridedSliceOpBuilder(graph_builder, op_type);
}
}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
