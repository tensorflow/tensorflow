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
#include "tensorflow/lite/delegates/hexagon/builders/reshape_builder.h"

#include <stdint.h>

#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

void PopulateOutputShapeFromTensor(const TfLiteTensor* shape_tensor,
                                   std::vector<int>* output_shape) {
  for (int i = 0; i < shape_tensor->dims->data[0]; ++i) {
    output_shape->push_back(shape_tensor->data.i32[i]);
  }
}

void PopulateShapeFromParam(const TfLiteReshapeParams* params,
                            std::vector<int>* output_shape) {
  // The function is returned above this line if the shape tensor is usable.
  // Now fallback to the shape parameter in `TfLiteReshapeParams`.
  int num_dimensions = params->num_dimensions;
  if (num_dimensions == 1 && params->shape[0] == 0) {
    // Legacy tflite models use a shape parameter of [0] to indicate scalars,
    // so adjust accordingly. TODO(b/111614235): Allow zero-sized buffers during
    // toco conversion.
    num_dimensions = 0;
  }
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->push_back(params->shape[i]);
  }
}
}  // namespace

TfLiteStatus ReshapeOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                                const TfLiteIntArray* outputs,
                                                TfLiteContext* context) {
  // Input data tensor.
  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));

  // Output shape.
  TfLiteTensor* shape_tensor = nullptr;
  bool output_shape_is_dynamic = false;
  if (inputs->size == 2) {
    shape_tensor = &context->tensors[inputs->data[1]];
    bool is_shape_tensor =
        (shape_tensor->dims->size == 1 && shape_tensor->type == kTfLiteInt32);
    // If tensor shape is dynamic, pass it along directly.
    if (shape_tensor->allocation_type != kTfLiteMmapRo && is_shape_tensor) {
      output_shape_is_dynamic = true;
      AddInput(graph_builder_->GetHexagonTensorId(inputs->data[1]));
    }
    if (!is_shape_tensor) {
      shape_tensor = nullptr;
    }
  }
  if (!output_shape_is_dynamic) {
    if (shape_tensor) {
      PopulateOutputShapeFromTensor(shape_tensor, &output_shape_);
    } else {
      const TfLiteReshapeParams* reshape_params =
          reinterpret_cast<const TfLiteReshapeParams*>(builtin_data_);
      PopulateShapeFromParam(reshape_params, &output_shape_);
    }
    int num_elements_in_shape = static_cast<int>(output_shape_.size());
    output_shape_shape_ = {1, 1, 1, num_elements_in_shape};
    auto* shape_node = graph_builder_->AddConstNodeWithData(
        output_shape_shape_.data(),
        reinterpret_cast<char*>(output_shape_.data()),
        sizeof(int) * num_elements_in_shape);
    AddInput(TensorID(shape_node->GetID(), 0));
  }

  // Hexagon output for this node.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});

  return kTfLiteOk;
}

TfLiteStatus ReshapeOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

ReshapeOpBuilder::~ReshapeOpBuilder() {}

OpBuilder* CreateReshapeBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ReshapeOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
