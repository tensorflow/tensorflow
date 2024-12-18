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
#include <algorithm>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {

// Adds Rsqrt op to the Hexagon graph by constructing
// 1/Sqrt(input).
class RsqrtOpBuilder : public OpBuilder {
 public:
  explicit RsqrtOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  void AddNumerator();

  TensorID node_output_;
  TensorID numerator_out_;
  TensorID numerator_min_;
  TensorID numerator_max_;
  // Total number of elements in the input tensor.
  int num_elements_;
};

void RsqrtOpBuilder::AddNumerator() {
  // Numerator is a constant with value 1. We add it as float and quantize it.
  std::vector<uint8_t> numerator;
  // Hexagon NN Div implementation assumes output to be of shape as first
  // input, so it doesn't broadcast.
  // So here we create the constant numerator with value 1 to be of same
  // flattened shape as the denominator.
  numerator.resize(num_elements_);
  int flat_shape[] = {1, 1, 1, num_elements_};
  std::fill(numerator.begin(), numerator.end(), 0);
  float kNumeratorMin = 1.0, kNumeratorMax = 1.0;
  auto* const_node = graph_builder_->AddConstNodeWithData(
      flat_shape, reinterpret_cast<char*>(numerator.data()),
      sizeof(numerator[0]) * numerator.size());
  auto* numerator_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&kNumeratorMin),
      sizeof(kNumeratorMin));
  auto* numerator_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&kNumeratorMax),
      sizeof(kNumeratorMax));
  numerator_out_ = TensorID(const_node->GetID(), 0);
  numerator_min_ = TensorID(numerator_min_const->GetID(), 0);
  numerator_max_ = TensorID(numerator_max_const->GetID(), 0);
}

TfLiteStatus RsqrtOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                             TfLiteContext* context) {
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

TfLiteStatus RsqrtOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                              const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  const int tensor_id = inputs->data[0];
  const auto& tensor = context->tensors[tensor_id];
  float min_value = 0;
  float max_value = 0;
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size, tensor.dims);
  TF_LITE_ENSURE_STATUS(
      ComputeMinAndMaxQuantValues(tensor, &min_value, &max_value));
  num_elements_ = batch_size * height_size * width_size * depth_size;
  int flat_shape[] = {1, 1, 1, num_elements_};

  auto* min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&min_value), sizeof(min_value));
  auto* max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&max_value), sizeof(max_value));
  // Create SQRT op as denominator.
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));
  AddInput(TensorID(min_const->GetID(), 0));
  AddInput(TensorID(max_const->GetID(), 0));
  auto sqrt_output = AddOutput(
      sizeof(uint8_t), 4, {batch_size, height_size, width_size, depth_size});
  auto sqrt_output_min = AddOutput(sizeof(float), 4, kScalarShape);
  auto sqrt_output_max = AddOutput(sizeof(float), 4, kScalarShape);

  // Reshape result of Sqrt to be [1,1,1,NumElements] since Hexagon Div
  // has limitation on the shape of the tensor.
  const int reshape_shape[] = {1, 1, 1, 4};
  auto* target_shape_node = graph_builder_->AddConstNodeWithData(
      reshape_shape, reinterpret_cast<char*>(flat_shape),
      sizeof(flat_shape[0]) * 4);
  auto* reshape_op = graph_builder_->AddNode(GetTFLiteNodeID());
  reshape_op->SetOpType(OP_Reshape);
  reshape_op->AddInput(sqrt_output);
  reshape_op->AddInput(TensorID(target_shape_node->GetID(), 0));
  auto reshape_out = reshape_op->AddOutput(sizeof(uint8_t), 4, flat_shape);

  // Create the numerator and add to the graph.
  AddNumerator();

  // Fetch output details
  float output_min = -1, output_max = 1;
  // Output details.
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      context->tensors[outputs->data[0]], &output_min, &output_max));
  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  // Add Div op to compute 1/Sqrt
  auto* div_op = graph_builder_->AddNode(GetTFLiteNodeID());
  div_op->SetOpType(OP_QuantizedDiv_8);
  div_op->AddInput(numerator_out_);
  div_op->AddInput(reshape_out);
  div_op->AddInput(numerator_min_);
  div_op->AddInput(numerator_max_);
  div_op->AddInput(sqrt_output_min);
  div_op->AddInput(sqrt_output_max);
  div_op->AddInput(TensorID(output_min_const->GetID(), 0));
  div_op->AddInput(TensorID(output_max_const->GetID(), 0));

  auto div_output = div_op->AddOutput(sizeof(uint8_t), 4, flat_shape);
  div_op->AddOutput(sizeof(float), 4, kScalarShape);
  div_op->AddOutput(sizeof(float), 4, kScalarShape);

  // Reshape output back to the expected shape.
  int output_shape[] = {output_batch_size, output_height_size,
                        output_width_size, output_depth_size};
  target_shape_node = graph_builder_->AddConstNodeWithData(
      reshape_shape, reinterpret_cast<char*>(output_shape),
      sizeof(output_shape[0]) * 4);

  reshape_op = graph_builder_->AddNode(GetTFLiteNodeID());
  reshape_op->SetOpType(OP_Reshape);
  reshape_op->AddInput(div_output);
  reshape_op->AddInput(TensorID(target_shape_node->GetID(), 0));
  node_output_ = reshape_op->AddOutput(sizeof(uint8_t), 4, output_shape);
  return kTfLiteOk;
}

OpBuilder* CreateRSqrtOpBuilder(GraphBuilder* graph_builder, int op_type) {
  return new RsqrtOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
