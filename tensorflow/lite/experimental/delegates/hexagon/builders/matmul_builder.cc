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
#include "tensorflow/lite/experimental/delegates/hexagon/builders/matmul_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
// The TFLite 'Fully-connected' quantized op corresponds to the following
// subgraph in Hexagon:
// Data (uint8), Weights (const, uint8) => MatMul => MatMul out (int32)
// Bias (const, int32) => Quantize => Bias (uint8)
// MatMul out (int32) => Quantize => MatMul out (uint8)
// MatMul out (uint8), Bias (uint8) => QuantizedAdd => Output (uint8)
// TODO(b/129276536): Add activation support.
TfLiteStatus MatMulOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  static int quant_bound_shape[] = {1, 1, 1, 1};

  // Data tensor.
  int data_tensor_id = inputs->data[0];
  const auto& data_tensor = context->tensors[data_tensor_id];
  TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
      data_tensor, &data_min_, &data_max_, std::numeric_limits<uint8_t>::min(),
      std::numeric_limits<uint8_t>::max()));
  auto* data_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&data_min_),
      sizeof(data_min_));
  auto* data_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&data_max_),
      sizeof(data_max_));

  // Weights vector.
  int weights_tensor_id = inputs->data[1];
  const auto& weights_tensor = context->tensors[weights_tensor_id];
  // TODO(srjoglekar): Abstract out.
  if (weights_tensor.allocation_type != kTfLiteMmapRo) {
    context->ReportError(
        context, "Weights tensor doesn't have correct allocation type: %s",
        weights_tensor.name);
    return kTfLiteError;
  }
  int batch_size, height_size, width_size, depth_size;
  // Hexagon lib expects the weight tensor in NHCW, TFLite uses NHWC.
  // Transpose NHWC -> NHCW
  GetDims(&batch_size, &height_size, &width_size, &depth_size,
          weights_tensor.dims);
  weights_shape_ = {batch_size, height_size, depth_size, width_size};
  RuntimeShape nhwc_shape({batch_size, height_size, width_size, depth_size});
  RuntimeShape nhcw_shape({batch_size, height_size, depth_size, width_size});
  std::vector<uint8_t> nhcw(NumElements(&weights_tensor));
  TransposeParams transpose_params;
  transpose_params.perm_count = 4;
  transpose_params.perm[0] = 0;
  transpose_params.perm[1] = 1;
  transpose_params.perm[2] = 3;
  transpose_params.perm[3] = 2;
  optimized_ops::Transpose<uint8_t>(transpose_params, nhwc_shape,
                                    weights_tensor.data.uint8, nhcw_shape,
                                    nhcw.data());
  auto* const_weights_node = graph_builder_->AddConstNodeWithData(
      weights_shape_.data(), reinterpret_cast<char*>(nhcw.data()),
      weights_tensor.bytes);
  graph_builder_->AddTensorWithID(weights_tensor_id,
                                  const_weights_node->GetID(), 0);
  ComputeMinAndMaxQuantValues(weights_tensor, &weights_min_, &weights_max_,
                              std::numeric_limits<uint8_t>::min(),
                              std::numeric_limits<uint8_t>::max());
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&weights_min_),
      sizeof(weights_min_));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&weights_max_),
      sizeof(weights_max_));

  // Data and weight tensors in required order.
  AddInput(graph_builder_->GetHexagonTensorId(data_tensor_id));
  AddInput(graph_builder_->GetHexagonTensorId(weights_tensor_id));
  AddInput(TensorID(data_min_const->GetID(), 0));
  AddInput(TensorID(data_max_const->GetID(), 0));
  AddInput(TensorID(weights_min_const->GetID(), 0));
  AddInput(TensorID(weights_max_const->GetID(), 0));

  // Outputs for the MatMul node, which are in int32 format.
  // Output shape should still be the same.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  const auto& matmul_out = AddOutput(sizeof(int32_t), 4,
                                     {output_batch_size, output_height_size,
                                      output_width_size, output_depth_size});
  const auto& matmul_out_min = AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  const auto& matmul_out_max = AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  // Quantize the MatMul output to quint8.
  auto* quantize_matmul_op = graph_builder_->AddNode(GetTFLiteNodeID());
  quantize_matmul_op->SetOpType(OP_QuantizeDownAndShrinkRange_32to8);
  quantize_matmul_op->AddInput(matmul_out);
  quantize_matmul_op->AddInput(matmul_out_min);
  quantize_matmul_op->AddInput(matmul_out_max);
  const auto& quantized_matmul_out =
      quantize_matmul_op->AddOutput(sizeof(uint8_t), 4,
                                    {output_batch_size, output_height_size,
                                     output_width_size, output_depth_size});
  const auto& quantized_matmul_out_min =
      quantize_matmul_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  const auto& quantized_matmul_out_max =
      quantize_matmul_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  // Bias tensor.
  int bias_tensor_id = inputs->data[2];
  const auto& bias_tensor = context->tensors[bias_tensor_id];
  auto* const_bias_node =
      graph_builder_->AddConstNodeWithData(bias_tensor_id, bias_tensor);
  graph_builder_->AddTensorWithID(bias_tensor_id, const_bias_node->GetID(), 0);
  ComputeMinAndMaxQuantValues(bias_tensor, &bias_min_, &bias_max_,
                              std::numeric_limits<int32_t>::min(),
                              std::numeric_limits<int32_t>::max());
  auto* bias_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&bias_min_),
      sizeof(bias_min_));
  auto* bias_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&bias_max_),
      sizeof(bias_max_));
  // Quantize bias
  auto* quantize_bias_op = graph_builder_->AddNode(GetTFLiteNodeID());
  quantize_bias_op->SetOpType(OP_QuantizeDownAndShrinkRange_32to8);
  quantize_bias_op->AddInput(
      graph_builder_->GetHexagonTensorId(bias_tensor_id));
  quantize_bias_op->AddInput(TensorID(bias_min_const->GetID(), 0));
  quantize_bias_op->AddInput(TensorID(bias_max_const->GetID(), 0));
  const auto& quantized_bias_out =
      quantize_bias_op->AddOutput(sizeof(uint8_t), 4,
                                  {output_batch_size, output_height_size,
                                   output_width_size, output_depth_size});
  const auto& quantized_bias_out_min =
      quantize_bias_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  const auto& quantized_bias_out_max =
      quantize_bias_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  // Output min/max.
  ComputeMinAndMaxQuantValues(context->tensors[outputs->data[0]], &output_min_,
                              &output_max_, std::numeric_limits<uint8_t>::min(),
                              std::numeric_limits<uint8_t>::max());

  auto* output_min_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&output_min_),
      sizeof(output_min_));
  auto* output_max_const = graph_builder_->AddConstNodeWithData(
      quant_bound_shape, reinterpret_cast<char*>(&output_max_),
      sizeof(output_max_));

  // MatMul + Bias.
  auto* bias_add_op = graph_builder_->AddNode(GetTFLiteNodeID());
  bias_add_op->SetOpType(OP_QuantizedAdd_8p8to8);
  bias_add_op->AddInput(quantized_matmul_out);
  bias_add_op->AddInput(quantized_bias_out);
  bias_add_op->AddInput(quantized_matmul_out_min);
  bias_add_op->AddInput(quantized_matmul_out_max);
  bias_add_op->AddInput(quantized_bias_out_min);
  bias_add_op->AddInput(quantized_bias_out_max);
  bias_add_op->AddInput(TensorID(output_min_const->GetID(), 0));
  bias_add_op->AddInput(TensorID(output_max_const->GetID(), 0));
  node_output_ = bias_add_op->AddOutput(sizeof(uint8_t), 4,
                                        {output_batch_size, output_height_size,
                                         output_width_size, output_depth_size});
  bias_add_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});
  bias_add_op->AddOutput(sizeof(float), 4, {1, 1, 1, 1});

  return kTfLiteOk;
}

TfLiteStatus MatMulOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

MatMulOpBuilder::~MatMulOpBuilder() {}

OpBuilder* CreateMatMulBuilder(GraphBuilder* graph_builder, int op_type) {
  return new MatMulOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
