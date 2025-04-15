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
#include "tensorflow/lite/delegates/hexagon/builders/quantize_builder.h"

#include <stdint.h>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus QuantizeOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                                 const TfLiteIntArray* outputs,
                                                 TfLiteContext* context) {
  const auto& input_tensor = context->tensors[inputs->data[0]];
  const auto& output_tensor = context->tensors[outputs->data[0]];
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_tensor.dims);

  AddInput(graph_builder_->GetHexagonTensorId(inputs->data[0]));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input_tensor));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, output_tensor));

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
