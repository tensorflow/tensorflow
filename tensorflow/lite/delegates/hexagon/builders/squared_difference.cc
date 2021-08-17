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
#include "tensorflow/lite/delegates/hexagon/builders/op_builder.h"

namespace tflite {
namespace delegates {
namespace hexagon {
// Builder for SquaredDifference op by computing Mul(Sub(A,B), Sub(A,B))
class SquaredDifferenceOpBuilder : public OpBuilder {
 public:
  explicit SquaredDifferenceOpBuilder(GraphBuilder* graph_builder, int op_type)
      : OpBuilder(graph_builder, op_type) {}
  TfLiteStatus PopulateSubGraph(const TfLiteIntArray* inputs,
                                const TfLiteIntArray* outputs,
                                TfLiteContext* context) override;

  TfLiteStatus RegisterOutputs(const TfLiteIntArray* outputs,
                               TfLiteContext* context) override;

 private:
  TensorID node_output_;
};

TfLiteStatus SquaredDifferenceOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // We model Squared Diff as Mul(Sub(a,b), Sub(a,b))

  // Add first Sub op.
  const int tensor_a_index = inputs->data[0];
  const int tensor_b_index = inputs->data[1];
  const auto& tensor_a = context->tensors[tensor_a_index];
  const auto& tensor_b = context->tensors[tensor_b_index];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_a_index));
  AddInput(graph_builder_->GetHexagonTensorId(tensor_b_index));
  // Inputs min/max
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, tensor_a));
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, tensor_b));
  // Output details.
  float output_min = -1, output_max = -1;
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

  auto sub_out = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  auto sub_min = AddOutput(sizeof(float), 4, kScalarShape);
  auto sub_max = AddOutput(sizeof(float), 4, kScalarShape);

  // Add Mul
  auto* mul_op = graph_builder_->AddNode(GetTFLiteNodeID());
  mul_op->SetOpType(OP_QuantizedMul_8x8to8);
  mul_op->AddInput(sub_out);
  mul_op->AddInput(sub_out);
  mul_op->AddInput(sub_min);
  mul_op->AddInput(sub_max);
  mul_op->AddInput(sub_min);
  mul_op->AddInput(sub_max);
  mul_op->AddInput(TensorID(output_min_const->GetID(), 0));
  mul_op->AddInput(TensorID(output_max_const->GetID(), 0));
  node_output_ = mul_op->AddOutput(sizeof(uint8_t), 4,
                                   {output_batch_size, output_height_size,
                                    output_width_size, output_depth_size});
  mul_op->AddOutput(sizeof(float), 4, kScalarShape);
  mul_op->AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus SquaredDifferenceOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

OpBuilder* CreateSquaredDifferenceOpBuilder(GraphBuilder* graph_builder,
                                            int op_type) {
  return new SquaredDifferenceOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
