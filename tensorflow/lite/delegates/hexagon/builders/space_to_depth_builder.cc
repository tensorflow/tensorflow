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
#include "tensorflow/lite/delegates/hexagon/builders/space_to_depth_builder.h"

#include <stdint.h>

#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus SpaceToDepthOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // Input tensor.
  int tensor_id = inputs->data[0];

  // Block size.
  const TfLiteSpaceToDepthParams* space_to_depth_params =
      reinterpret_cast<const TfLiteSpaceToDepthParams*>(builtin_data_);
  block_size_ = space_to_depth_params->block_size;
  auto* block_size_node = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&block_size_), sizeof(int));

  // All inputs.
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  AddInput(TensorID(block_size_node->GetID(), 0));
  TF_LITE_ENSURE_STATUS(
      ComputeAndAddMinAndMax(context, context->tensors[tensor_id]));

  // Hexagon outputs for this node.
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

TfLiteStatus SpaceToDepthOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

SpaceToDepthOpBuilder::~SpaceToDepthOpBuilder() {}

OpBuilder* CreateSpaceToDepthBuilder(GraphBuilder* graph_builder, int op_type) {
  return new SpaceToDepthOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
