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
#include "tensorflow/lite/delegates/hexagon/builders/mirror_pad_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus MirrorPadOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                                  const TfLiteIntArray* outputs,
                                                  TfLiteContext* context) {
  // Input data tensor.
  int tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // Padding tensor.
  // Should be a constant.
  tensor_id = inputs->data[1];
  const auto& padding_tensor = context->tensors[tensor_id];
  if (padding_tensor.dims->size != 2 || padding_tensor.dims->data[0] > 4 ||
      padding_tensor.dims->data[1] != 2) {
    TF_LITE_KERNEL_LOG(context, "Invalid padding tensor shape");
    return kTfLiteError;
  }
  paddings_shape_ = {1, 1, 4, 2};
  std::vector<int> padding_data(8, 0);
  // Hexagon always expects padding data for each dimension in order {b, h, w,
  // d}. This start value ensures we pad the non-relevant dimensions with 0.
  int padding_data_start = 8 - padding_tensor.dims->data[0] * 2;
  for (int i = 0; i < padding_tensor.dims->data[0] * 2; ++i) {
    padding_data[padding_data_start + i] = padding_tensor.data.i32[i];
  }
  auto* const_padding_node = graph_builder_->AddConstNodeWithData(
      paddings_shape_.data(), reinterpret_cast<char*>(padding_data.data()),
      padding_data.size() * sizeof(padding_data[0]));
  AddInput(TensorID(const_padding_node->GetID(), 0));
  // Padding type.
  const TfLiteMirrorPaddingParams* params =
      reinterpret_cast<const TfLiteMirrorPaddingParams*>(builtin_data_);
  if (params->mode == kTfLiteMirrorPaddingReflect) {
    SetPaddingType(NN_PAD_MIRROR_REFLECT);
  } else if (params->mode == kTfLiteMirrorPaddingSymmetric) {
    SetPaddingType(NN_PAD_MIRROR_SYMMETRIC);
  }

  // Min/max values for input tensor.
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input_tensor));

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

TfLiteStatus MirrorPadOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                                 TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

MirrorPadOpBuilder::~MirrorPadOpBuilder() {}

OpBuilder* CreateMirrorPadBuilder(GraphBuilder* graph_builder, int op_type) {
  return new MirrorPadOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
