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
#include "tensorflow/lite/delegates/hexagon/builders/reduce_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ReduceOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  // Input data tensor.
  int tensor_id = inputs->data[0];
  const auto& input_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, input_tensor));

  // Axes tensor should be constant.
  tensor_id = inputs->data[1];
  const auto& axes_tensor = context->tensors[tensor_id];


  if (axes_tensor.allocation_type == kTfLiteMmapRo) {

    // TfLiteTensor* new_axes_tensor = new TfLiteTensor();
    // TF_LITE_ENSURE_STATUS(TfLiteTensorCopy(&axes_tensor, new_axes_tensor));
    // context->ReportError(context, "Axes tensor has type %d", axes_tensor.type);
    // TF_LITE_ASSERT(axes_tensor.type == kTfLiteInt32);

    auto axes_orig = GetTensorData<int32_t>(&axes_tensor);
    auto axes_tensor_size = axes_tensor.dims->size;
    // context->ReportError(context, "Axes tensor has size %d", axes_tensor_size);
    // for (int i = 0; i < axes_tensor_size; i++){
    //   context->ReportError(context, "Axes tensor at dim %d is %d", i, axes_orig[i]);
    // }
    auto num_axes = axes_tensor.dims->size;

    // Increase the constant axis by the number of dimensions we're padding
    int num_dimensions_padding = 4 - input_tensor.dims->size;
    // context->ReportError(context, "Input tensor has %d dims", input_tensor.dims->size);
    // context->ReportError(context, "Therefore will pad by %d (4 - num_dims)",
    //                      num_dimensions_padding);

    TfLiteTensor* new_tensor = new TfLiteTensor();
    // context->ReportError(context, "initialized new tensor");

    new_tensor->type = axes_tensor.type;
    new_tensor->bytes = axes_tensor.bytes;
    new_tensor->data = TfLitePtrUnion{};
    new_tensor->dims = TfLiteIntArrayCopy(axes_tensor.dims);
    new_tensor->buffer_handle = axes_tensor.buffer_handle;
    new_tensor->data_is_stale = axes_tensor.data_is_stale;
    new_tensor->delegate = axes_tensor.delegate;

    new_tensor->data.data = (void *) new int32_t[axes_tensor_size];

    auto axes = const_cast<int32_t*>(GetTensorData<int32_t>(new_tensor));

    for (int i = 0; i < axes_tensor_size; i++) {
      axes[i] = axes_orig[i];
      // context->ReportError(context, "Padding axes at index %d of total %d", i, axes_tensor_size);
      // context->ReportError(context, "Before padding: %d", axes[i]);
      // We always left-pad by ones so these should be valid.
      axes[i] += num_dimensions_padding;
      // context->ReportError(context, "After padding: %d", axes[i]);
    }

    // context->tensors[tensor_id] = *new_tensor;

    // If the axes input is a constant, bake it into the Hexagon graph as a
    // Const node.
    auto* const_axes_node =
        graph_builder_->AddConstNodeWithData(tensor_id, *new_tensor);

    AddInput(TensorID(const_axes_node->GetID(), 0));

    // TfLiteTensorDataFree(&axes_tensor.data);
    // TfLiteTensorFree(const_cast<TfLiteTensor*>(&axes_tensor));

  } else {
    TF_LITE_KERNEL_LOG(context, "Reduction op doesn't have constant axis");
    return kTfLiteError;
  }

  auto& output_tensor = context->tensors[outputs->data[0]];
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, output_tensor.dims);

  float output_min = -1, output_max = -1;
  ComputeMinAndMaxQuantValues(output_tensor, &output_min, &output_max);
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  // Min/max values for output tensor.
  AddInput(TensorID(output_min_const->GetID(), 0));
  AddInput(TensorID(output_max_const->GetID(), 0));

  // Add outputs
  size_t output_element_size = 0;
  TF_LITE_ENSURE_STATUS(
      GetSizeOfType(context, output_tensor.type, &output_element_size));
  auto mean_output = AddOutput(output_element_size, 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
  auto mean_out_min = AddOutput(output_element_size, 4, kScalarShape);
  auto mean_out_max = AddOutput(output_element_size, 4, kScalarShape);
  // Mean op doesn't honor the passed min/max for output, so we need
  // to add requantize.
  auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
  requantize_op->SetOpType(OP_Requantize_8to8);
  requantize_op->AddInput(mean_output);
  requantize_op->AddInput(mean_out_min);
  requantize_op->AddInput(mean_out_max);
  requantize_op->AddInput(TensorID(output_min_const->GetID(), 0));
  requantize_op->AddInput(TensorID(output_max_const->GetID(), 0));
  node_output_ =
      requantize_op->AddOutput(sizeof(uint8_t), 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
  requantize_op->AddOutput(sizeof(float), 4, kScalarShape);
  requantize_op->AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus ReduceOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);

  return kTfLiteOk;
}

ReduceOpBuilder::~ReduceOpBuilder() {}

OpBuilder* CreateReduceBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ReduceOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
