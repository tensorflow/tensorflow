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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/concat_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
TfLiteStatus ConcatOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  static int quant_bound_shape[] = {1, 1, 1, 1};

  // Only axis 3 is supported.
  const TfLiteConcatenationParams* concat_params =
      reinterpret_cast<const TfLiteConcatenationParams*>(builtin_data_);
  auto* axis_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, (char*)&concat_params->axis,
      sizeof(concat_params->axis));
  AddInput(TensorID(axis_const->GetID(), 0));

  int tensor_id;

  // Input data tensors.
  // input_bound_minimum & input_bound_maximum track the minimum & maximum
  // min/max bounds across all inputs.
  float input_bound_minimum = std::numeric_limits<float>::max();
  float input_bound_maximum = std::numeric_limits<float>::min();
  input_minima_.reserve(inputs->size);
  input_maxima_.reserve(inputs->size);
  for (int i = 0; i < inputs->size; ++i) {
    tensor_id = inputs->data[i];
    float data_min, data_max;
    const auto& data_tensor = context->tensors[tensor_id];
    AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        data_tensor, &data_min, &data_max, std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max()));
    input_minima_.push_back(data_min);
    input_maxima_.push_back(data_max);
    if (data_min < input_bound_minimum) input_bound_minimum = data_min;
    if (data_max > input_bound_maximum) input_bound_maximum = data_max;
  }

  // Minima tensors.
  for (int i = 0; i < input_minima_.size(); ++i) {
    auto* data_min_const = graph_builder_->AddConstNodeWithData(
        quant_bound_shape, reinterpret_cast<char*>(&input_minima_[i]),
        sizeof(input_minima_[i]));
    AddInput(TensorID(data_min_const->GetID(), 0));
  }

  // Maxima tensors.
  for (int i = 0; i < input_minima_.size(); ++i) {
    auto* data_max_const = graph_builder_->AddConstNodeWithData(
        quant_bound_shape, reinterpret_cast<char*>(&input_maxima_[i]),
        sizeof(input_maxima_[i]));
    AddInput(TensorID(data_max_const->GetID(), 0));
  }

  // Hexagon outputs for this node.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  // We requantize the output from concat to the range expected by TFLite.
  // Otherwise, we see accuracy issues for cases where the inputs have different
  // min/max bounds.
  TensorID concat_out = AddOutput(sizeof(uint8_t), 4,
                                  {output_batch_size, output_height_size,
                                   output_width_size, output_depth_size});
  const auto& concat_out_min = AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  const auto& concat_out_max = AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  // Output min/max for requantization.
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min_, &output_max_,
      std::numeric_limits<uint8_t>::min(),
      std::numeric_limits<uint8_t>::max()));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, (char*)&output_min_, sizeof(output_min_));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, (char*)&output_max_, sizeof(output_max_));

  if (output_min_ == input_bound_minimum &&
      output_max_ == input_bound_maximum) {
    // If the input min/max (across all tensors) is same as the output min/max,
    // Hexagon's Requantize causes errors in InceptionV3.
    // TODO(b/150137234): Figure out why this is.
    node_output_ = concat_out;
  } else {
    auto* requantize_op = graph_builder_->AddNode(GetTFLiteNodeID());
    requantize_op->SetOpType(OP_Requantize_8to8);
    requantize_op->AddInput(concat_out);
    requantize_op->AddInput(concat_out_min);
    requantize_op->AddInput(concat_out_max);
    requantize_op->AddInput(TensorID(output_min_const->GetID(), 0));
    requantize_op->AddInput(TensorID(output_max_const->GetID(), 0));
    node_output_ =
        requantize_op->AddOutput(sizeof(uint8_t), 4,
                                 {output_batch_size, output_height_size,
                                  output_width_size, output_depth_size});
    requantize_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
    requantize_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  }

  return kTfLiteOk;
}

TfLiteStatus ConcatOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

ConcatOpBuilder::~ConcatOpBuilder() {}

OpBuilder* CreateConcatBuilder(GraphBuilder* graph_builder, int op_type) {
  return new ConcatOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
