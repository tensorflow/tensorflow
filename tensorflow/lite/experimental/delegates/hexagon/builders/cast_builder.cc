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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/cast_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus CastOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                             const TfLiteIntArray* outputs,
                                             TfLiteContext* context) {
  static int scalar_shape[] = {1, 1, 1, 1};

  // Should be only 1 tensor that is cast in-place.
  if (inputs->size != 1 || outputs->size != 1) {
    TF_LITE_KERNEL_LOG(context, "Cast supports a single tensor");
    return kTfLiteError;
  } else if (inputs->data[0] != outputs->data[0]) {
    TF_LITE_KERNEL_LOG(context, "input & output should be same for Cast");
    return kTfLiteError;
  }

  int tensor_id = inputs->data[0];
  const auto& tensor = context->tensors[tensor_id];
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor.dims);
  float min_value = 0;
  float max_value = 0;
  if (tensor.quantization.type ==
      TfLiteQuantizationType::kTfLiteAffineQuantization) {
    // Casting doesn't require min/max, so populate only if available.
    TF_LITE_ENSURE_STATUS(
        ComputeMinAndMaxQuantValues(tensor, &min_value, &max_value));
  }
  auto* min_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&min_value), sizeof(min_value));
  auto* max_const = graph_builder_->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&max_value), sizeof(max_value));

  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  AddInput(TensorID(min_const->GetID(), 0));
  AddInput(TensorID(max_const->GetID(), 0));
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {batch_size, height_size, width_size, depth_size});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  return kTfLiteOk;
}

TfLiteStatus CastOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                            TfLiteContext* context) {
  // Should be only 1 output.
  // Cast tensor already exists in the graph, so we need to overwrite it with
  // the new TensorID.
  if (!graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                       node_output_.second,
                                       /*overwrite*/ true)) {
    TF_LITE_KERNEL_LOG(context, "Could not register Cast output.");
    return kTfLiteError;
  }

  return kTfLiteOk;
}

CastOpBuilder::~CastOpBuilder() {}

OpBuilder* CreateCastBuilder(GraphBuilder* graph_builder, int op_type) {
  return new CastOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
