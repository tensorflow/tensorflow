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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/resize_bilinear_builder.h"

#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ResizeBilinearOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  if (inputs->size != 2) {
    context->ReportError(context, "Expecting 2 inputs %d != 2\n", inputs->size);
    return kTfLiteError;
  }

  // Input data tensor.
  int input_tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[input_tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(input_tensor_id));

  const auto& size_tensor = context->tensors[inputs->data[1]];
  if (!IsConstantTensor(&size_tensor)) {
    context->ReportError(context,
                         "Hexagon Delegate doesn't support dynamic shape.\n");
    return kTfLiteError;
  }
  // dims tensor.
  const int dims_shape[] = {1, 1, 1, 2};
  std::vector<int> dims = {size_tensor.data.i32[0], size_tensor.data.i32[1]};
  auto* dims_const = graph_builder_->AddConstNodeWithData(
      dims_shape, reinterpret_cast<char*>(dims.data()),
      sizeof(int) * dims.size());
  AddInput(TensorID(dims_const->GetID(), 0));

  // Input min/max
  TF_LITE_ENSURE_OK(context, ComputeMinAndMaxQuantValues(
                                 input_tensor, &input_min_, &input_max_));
  auto* input_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_min_), sizeof(input_min_));
  auto* input_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&input_max_), sizeof(input_max_));

  AddInput(TensorID(input_min_const->GetID(), 0));
  AddInput(TensorID(input_max_const->GetID(), 0));
  // Align Corners
  const TfLiteResizeBilinearParams* params =
      reinterpret_cast<const TfLiteResizeBilinearParams*>(builtin_data_);
  int align_corners = params->align_corners ? 1 : 0;
  auto* align_corners_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&align_corners),
      sizeof(align_corners));
  AddInput(TensorID(align_corners_const->GetID(), 0));

  // Output
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  auto resize_bilinear_out = AddOutput(sizeof(uint8_t), 4,
                                       {output_batch_size, output_height_size,
                                        output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);
  node_output_ = resize_bilinear_out;

  return kTfLiteOk;
}

TfLiteStatus ResizeBilinearOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

ResizeBilinearOpBuilder::~ResizeBilinearOpBuilder() {}

OpBuilder* CreateResizeBilinearOpBuilder(GraphBuilder* graph_builder,
                                         int op_type) {
  return new ResizeBilinearOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
