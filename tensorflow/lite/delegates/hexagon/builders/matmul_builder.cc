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
#include "tensorflow/lite/delegates/hexagon/builders/matmul_builder.h"

#include <stdint.h>

#include <vector>

#include "hexagon/hexagon_nn_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {
void GetDims(int* batch_size, int* height_size, int* width_size,
             int* depth_size, const TfLiteIntArray* dims) {
  int* dim[] = {batch_size, height_size, width_size, depth_size};
  for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
  for (int i = 4 - dims->size; i < 4; ++i) {
    *dim[i] = dims->data[i - (4 - dims->size)];
  }
}

constexpr uint8_t k8BitSignFlipConstant = 0x80;

TfLiteStatus AddFullyConnectedHelper(const TfLiteIntArray* inputs,
                                     const TfLiteIntArray* outputs,
                                     const OpBuilder::TensorID weights_id,
                                     const OpBuilder::TensorID weights_min_id,
                                     const OpBuilder::TensorID weights_max_id,
                                     GraphBuilder* graph_builder,
                                     TfLiteContext* context,
                                     OpBuilder* matmul_op,
                                     OpBuilder::TensorID* node_output) {
  static int scalar_shape[] = {1, 1, 1, 1};
  // Data tensor.
  int data_tensor_id = inputs->data[0];
  const auto& data_tensor = context->tensors[data_tensor_id];
  float data_min, data_max;
  TF_LITE_ENSURE_STATUS(OpBuilder::ComputeMinAndMaxQuantValues(
      data_tensor, &data_min, &data_max));
  auto* data_min_const = graph_builder->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&data_min), sizeof(data_min));
  auto* data_max_const = graph_builder->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&data_max), sizeof(data_max));

  // Data and weight tensors in required order.
  matmul_op->AddInput(graph_builder->GetHexagonTensorId(data_tensor_id));
  matmul_op->AddInput(weights_id);
  matmul_op->AddInput(OpBuilder::TensorID(data_min_const->GetID(), 0));
  matmul_op->AddInput(OpBuilder::TensorID(data_max_const->GetID(), 0));
  matmul_op->AddInput(weights_min_id);
  matmul_op->AddInput(weights_max_id);

  // Outputs for the MatMul node, which are in int32 format.
  // Output shape should still be the same.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);
  const auto& matmul_out =
      matmul_op->AddOutput(sizeof(int), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  const auto& matmul_out_min =
      matmul_op->AddOutput(sizeof(float), 4, scalar_shape);
  const auto& matmul_out_max =
      matmul_op->AddOutput(sizeof(float), 4, scalar_shape);

  // Bias tensor.
  int bias_tensor_id = inputs->data[2];
  OpBuilder::TensorID matmul_and_bias_out = matmul_out,
                      matmul_and_bias_out_min = matmul_out_min,
                      matmul_and_bias_out_max = matmul_out_max;
  if (bias_tensor_id != -1) {
    const auto& bias_tensor = context->tensors[bias_tensor_id];
    float bias_min, bias_max;
    OpBuilder::ComputeMinAndMaxQuantValues(bias_tensor, &bias_min, &bias_max);
    auto* bias_min_const = graph_builder->AddConstNodeWithData(
        scalar_shape, reinterpret_cast<char*>(&bias_min), sizeof(bias_min));
    auto* bias_max_const = graph_builder->AddConstNodeWithData(
        scalar_shape, reinterpret_cast<char*>(&bias_max), sizeof(bias_max));

    // MatMul + Bias.
    auto* bias_add_op = graph_builder->AddNode(matmul_op->GetTFLiteNodeID());
    bias_add_op->SetOpType(OP_QuantizedBiasAdd_32p32to32);
    bias_add_op->AddInput(matmul_out);
    bias_add_op->AddInput(graph_builder->GetHexagonTensorId(bias_tensor_id));
    bias_add_op->AddInput(matmul_out_min);
    bias_add_op->AddInput(matmul_out_max);
    bias_add_op->AddInput(OpBuilder::TensorID(bias_min_const->GetID(), 0));
    bias_add_op->AddInput(OpBuilder::TensorID(bias_max_const->GetID(), 0));
    matmul_and_bias_out =
        bias_add_op->AddOutput(sizeof(int), 4,
                               {output_batch_size, output_height_size,
                                output_width_size, output_depth_size});
    matmul_and_bias_out_min =
        bias_add_op->AddOutput(sizeof(float), 4, scalar_shape);
    matmul_and_bias_out_max =
        bias_add_op->AddOutput(sizeof(float), 4, scalar_shape);
  }

  float output_min, output_max;
  // Quantize 32-bit result into 8-bit format using output tensor min/max.
  OpBuilder::ComputeMinAndMaxQuantValues(context->tensors[outputs->data[0]],
                                         &output_min, &output_max);
  auto* output_min_const = graph_builder->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&output_min), sizeof(output_min));
  auto* output_max_const = graph_builder->AddConstNodeWithData(
      scalar_shape, reinterpret_cast<char*>(&output_max), sizeof(output_max));
  auto* quantize_biasadd_op =
      graph_builder->AddNode(matmul_op->GetTFLiteNodeID());
  quantize_biasadd_op->SetOpType(OP_Requantize_32to8);
  quantize_biasadd_op->AddInput(matmul_and_bias_out);
  quantize_biasadd_op->AddInput(matmul_and_bias_out_min);
  quantize_biasadd_op->AddInput(matmul_and_bias_out_max);
  quantize_biasadd_op->AddInput(
      OpBuilder::TensorID(output_min_const->GetID(), 0));
  quantize_biasadd_op->AddInput(
      OpBuilder::TensorID(output_max_const->GetID(), 0));
  *node_output =
      quantize_biasadd_op->AddOutput(sizeof(uint8_t), 4,
                                     {output_batch_size, output_height_size,
                                      output_width_size, output_depth_size});
  quantize_biasadd_op->AddOutput(sizeof(float), 4, scalar_shape);
  quantize_biasadd_op->AddOutput(sizeof(float), 4, scalar_shape);
  return kTfLiteOk;
}

}  // namespace

// The TFLite 'Fully-connected' quantized op corresponds to the following
// subgraph in Hexagon:
// Data (8-bit), Weights (const, 8-bit) => MatMul => MatMul out (int32)
// MatMul out (int32), Bias (int32) => QuantizedBiasAdd => BiasAdd out (int32)
// BiasAdd out (int32) => Requantize_32to8 => Output (8-bit)
TfLiteStatus MatMulWithConstWeightsOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
  // Weights vector.
  int weights_tensor_id = inputs->data[1];
  const auto& weights_tensor = context->tensors[weights_tensor_id];
  if (weights_tensor.allocation_type != kTfLiteMmapRo) {
    TF_LITE_KERNEL_LOG(
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
  if (weights_tensor.type == kTfLiteInt8) {
    optimized_ops::Transpose<int8_t>(transpose_params, nhwc_shape,
                                     weights_tensor.data.int8, nhcw_shape,
                                     reinterpret_cast<int8_t*>(nhcw.data()));
    // Flip bits on the weight values so that the int8 values are treated
    // as uint8.
    for (int i = 0; i < nhcw.size(); ++i) {
      nhcw[i] = nhcw[i] ^ k8BitSignFlipConstant;
    }
  } else {
    optimized_ops::Transpose<uint8_t>(transpose_params, nhwc_shape,
                                      weights_tensor.data.uint8, nhcw_shape,
                                      nhcw.data());
  }
  auto* const_weights_node = graph_builder_->AddConstNodeWithData(
      weights_shape_.data(), reinterpret_cast<char*>(nhcw.data()),
      weights_tensor.bytes);
  graph_builder_->AddTensorWithID(weights_tensor_id,
                                  const_weights_node->GetID(), 0, true);
  ComputeMinAndMaxQuantValues(weights_tensor, &weights_min_, &weights_max_);
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_min_),
      sizeof(weights_min_));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_max_),
      sizeof(weights_max_));

  return AddFullyConnectedHelper(
      inputs, outputs, graph_builder_->GetHexagonTensorId(weights_tensor_id),
      TensorID(weights_min_const->GetID(), 0),
      TensorID(weights_max_const->GetID(), 0), graph_builder_, context, this,
      &node_output_);
}

TfLiteStatus MatMulWithConstWeightsOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

TfLiteStatus MatMulOpBuilder::PopulateSubGraph(const TfLiteIntArray* inputs,
                                               const TfLiteIntArray* outputs,
                                               TfLiteContext* context) {
  const int weights_tensor_id = inputs->data[1];
  const auto& weights_tensor = context->tensors[weights_tensor_id];
  int batch_size, height_size, width_size, depth_size;
  GetDims(&batch_size, &height_size, &width_size, &depth_size,
          weights_tensor.dims);
  weights_shape_ = {batch_size, height_size, depth_size, width_size};
  // Permutation for transposing.
  int permutation[] = {0, 1, 3, 2};
  const int permutation_shape[] = {1, 1, 1, 4};
  auto permutation_node = graph_builder_->AddConstNodeWithData(
      permutation_shape, reinterpret_cast<char*>(permutation),
      4 * sizeof(permutation[0]));
  AddInput(graph_builder_->GetHexagonTensorId(weights_tensor_id));
  AddInput(TensorID(permutation_node->GetID(), 0));

  ComputeMinAndMaxQuantValues(weights_tensor, &weights_min_, &weights_max_);
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_min_),
      sizeof(weights_min_));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_max_),
      sizeof(weights_max_));
  AddInput(TensorID(weights_min_const->GetID(), 0));
  AddInput(TensorID(weights_max_const->GetID(), 0));

  auto transposed_weights = AddOutput(sizeof(uint8_t), 4, weights_shape_);
  auto transposed_weights_min = AddOutput(sizeof(float), 4, kScalarShape);
  auto transposed_weights_max = AddOutput(sizeof(float), 4, kScalarShape);

  auto* matmul_op = graph_builder_->AddNode(GetTFLiteNodeID());
  matmul_op->SetOpType(OP_QuantizedMatMul_8x8to32);

  AddFullyConnected(inputs, outputs, transposed_weights, transposed_weights_min,
                    transposed_weights_max, context, matmul_op);
  return kTfLiteOk;
}

TfLiteStatus MatMulOpBuilder::AddFullyConnected(const TfLiteIntArray* inputs,
                                                const TfLiteIntArray* outputs,
                                                const TensorID weights_id,
                                                const TensorID weights_min_id,
                                                const TensorID weights_max_id,
                                                TfLiteContext* context,
                                                OpBuilder* matmul_op) {
  return AddFullyConnectedHelper(inputs, outputs, weights_id, weights_min_id,
                                 weights_max_id, graph_builder_, context,
                                 matmul_op, &node_output_);
}

TfLiteStatus MatMulOpBuilder::RegisterOutputs(const TfLiteIntArray* outputs,
                                              TfLiteContext* context) {
  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

OpBuilder* CreateMatMulWithConstWeightsOpBuilder(GraphBuilder* graph_builder,
                                                 int op_type) {
  return new MatMulWithConstWeightsOpBuilder(graph_builder, op_type);
}

OpBuilder* CreateMatMulOpBuilder(GraphBuilder* graph_builder, int op_type) {
  return new MatMulOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
