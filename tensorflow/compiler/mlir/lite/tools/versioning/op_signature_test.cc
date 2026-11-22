/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/tools/versioning/op_signature.h"

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/compiler/mlir/lite/core/absl_error_model_builder.h"
#include "tensorflow/compiler/mlir/lite/core/c/tflite_types.h"
#include "tensorflow/compiler/mlir/lite/schema/schema_generated.h"
#include "tensorflow/core/platform/resource_loader.h"

namespace tflite {

TEST(GetOpSignature, FlatBufferModel) {
  const std::string& full_path = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/testdata/add.bin");
  auto fb_model =
      mlir::TFL::FlatBufferModelAbslError::BuildFromFile(full_path.data());
  ASSERT_TRUE(fb_model);
  auto model = fb_model->GetModel();
  auto subgraphs = model->subgraphs();
  const SubGraph* subgraph = subgraphs->Get(0);
  const Operator* op1 = subgraph->operators()->Get(0);
  const OperatorCode* op_code1 =
      model->operator_codes()->Get(op1->opcode_index());
  OpSignature op_sig = GetOpSignature(op_code1, op1, subgraph, model);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_FALSE(op_sig.inputs[0].is_shape_dynamic);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.outputs[0].is_shape_dynamic);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  EXPECT_EQ(op_sig.version, 1);
  free(op_sig.builtin_data);

  const Operator* op2 = subgraph->operators()->Get(1);
  const OperatorCode* op_code2 =
      model->operator_codes()->Get(op2->opcode_index());
  op_sig = GetOpSignature(op_code2, op2, subgraph, model);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_FALSE(op_sig.inputs[0].is_shape_dynamic);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 4);
  EXPECT_FALSE(op_sig.outputs[0].is_shape_dynamic);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  EXPECT_EQ(op_sig.version, 1);
  free(op_sig.builtin_data);

  const std::string& full_path3 = tensorflow::GetDataDependencyFilepath(
      "tensorflow/compiler/mlir/lite/testdata/multi_signatures.bin");
  auto fb_model3 =
      mlir::TFL::FlatBufferModelAbslError::BuildFromFile(full_path3.data());
  ASSERT_TRUE(fb_model3);
  auto model3 = fb_model3->GetModel();
  auto subgraphs3 = model3->subgraphs();
  const SubGraph* subgraph3 = subgraphs3->Get(0);
  const Operator* op3 = subgraph3->operators()->Get(0);
  const OperatorCode* op_code3 =
      model3->operator_codes()->Get(op3->opcode_index());
  op_sig = GetOpSignature(op_code3, op3, subgraph3, model3);
  EXPECT_EQ(op_sig.op, BuiltinOperator_ADD);
  EXPECT_EQ(op_sig.inputs[0].type, kTfLiteFloat32);
  EXPECT_EQ(op_sig.inputs[0].dims.size(), 1);
  EXPECT_FALSE(op_sig.inputs[0].is_const);
  EXPECT_TRUE(op_sig.inputs[0].is_shape_dynamic);
  EXPECT_EQ(op_sig.outputs[0].type, kTfLiteFloat32);
  EXPECT_FALSE(op_sig.outputs[0].is_const);
  EXPECT_EQ(op_sig.outputs[0].dims.size(), 1);
  EXPECT_TRUE(op_sig.outputs[0].is_shape_dynamic);
  EXPECT_NE(op_sig.builtin_data, nullptr);
  EXPECT_EQ(op_sig.version, 1);
  free(op_sig.builtin_data);
}

// Builds a model with one FULLY_CONNECTED whose weight carries
// `BlockwiseQuantization` with the given `block_size` and `block_shape`.
std::vector<uint8_t> BuildBlockwiseWeightFullyConnectedModel(
    int32_t block_size, const std::vector<int32_t>& block_shape) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{})),
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>(4, 0))),
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>(8, 0)))};
  auto weight_details = CreateBlockwiseQuantization(
      builder, /*scales=*/2, /*zero_points=*/-1, block_size,
      block_shape.empty() ? 0 : builder.CreateVector(block_shape));
  auto weight_quant = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, weight_details.Union());
  std::vector<flatbuffers::Offset<Tensor>> tensors = {
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{1, 8}),
                   TensorType_INT8, /*buffer=*/0),
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{2, 8}),
                   TensorType_INT2, /*buffer=*/1, /*name=*/0, weight_quant),
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{2, 1}),
                   TensorType_FLOAT32, /*buffer=*/2),
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{1, 2}),
                   TensorType_INT8, /*buffer=*/0)};
  auto op = CreateOperator(builder, /*opcode_index=*/0,
                           builder.CreateVector(std::vector<int32_t>{0, 1}),
                           builder.CreateVector(std::vector<int32_t>{3}),
                           BuiltinOptions_FullyConnectedOptions,
                           CreateFullyConnectedOptions(builder).Union());
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(std::vector<int32_t>{3}),
      builder.CreateVector(std::vector<flatbuffers::Offset<Operator>>{op}));
  auto op_code = CreateOperatorCode(
      builder, BuiltinOperator_FULLY_CONNECTED, /*custom_code=*/0,
      /*version=*/1, BuiltinOperator_FULLY_CONNECTED);
  auto model =
      CreateModel(builder, /*version=*/3, builder.CreateVector({op_code}),
                  builder.CreateVector({subgraph}), /*description=*/0,
                  builder.CreateVector(buffers));
  FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

bool IsWeightBlockwiseQuantized(const std::vector<uint8_t>& buffer) {
  const Model* model = GetModel(buffer.data());
  const SubGraph* subgraph = model->subgraphs()->Get(0);
  const Operator* op = subgraph->operators()->Get(0);
  OpSignature op_sig = GetOpSignature(
      model->operator_codes()->Get(op->opcode_index()), op, subgraph, model);
  free(op_sig.builtin_data);
  return op_sig.ext_options.fully_connected.is_weight_blockwise_quantized;
}

TEST(GetOpSignature, FullyConnectedBlockwiseWeight) {
  // N-D blockwise quantization (non-empty `block_shape`).
  EXPECT_TRUE(IsWeightBlockwiseQuantized(
      BuildBlockwiseWeightFullyConnectedModel(/*block_size=*/0, {1, 8})));
  // The 4-bit hybrid blockwise quantization only sets `block_size`.
  EXPECT_FALSE(IsWeightBlockwiseQuantized(
      BuildBlockwiseWeightFullyConnectedModel(/*block_size=*/8, {})));
}

// Builds a model with one EMBEDDING_LOOKUP whose table carries
// `BlockwiseQuantization` with the given `block_size` and `block_shape`.
std::vector<uint8_t> BuildBlockwiseTableEmbeddingLookupModel(
    int32_t block_size, const std::vector<int32_t>& block_shape) {
  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>{})),
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>(4, 0))),
      CreateBuffer(builder, builder.CreateVector(std::vector<uint8_t>(8, 0)))};
  auto table_details = CreateBlockwiseQuantization(
      builder, /*scales=*/2, /*zero_points=*/-1, block_size,
      block_shape.empty() ? 0 : builder.CreateVector(block_shape));
  auto table_quant = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0, /*scale=*/0, /*zero_point=*/0,
      QuantizationDetails_BlockwiseQuantization, table_details.Union());
  std::vector<flatbuffers::Offset<Tensor>> tensors = {
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{1}),
                   TensorType_INT32, /*buffer=*/0),
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{2, 8}),
                   TensorType_INT2, /*buffer=*/1, /*name=*/0, table_quant),
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{2, 1}),
                   TensorType_FLOAT32, /*buffer=*/2),
      CreateTensor(builder, builder.CreateVector(std::vector<int32_t>{1, 8}),
                   TensorType_FLOAT32, /*buffer=*/0)};
  auto op = CreateOperator(builder, /*opcode_index=*/0,
                           builder.CreateVector(std::vector<int32_t>{0, 1}),
                           builder.CreateVector(std::vector<int32_t>{3}));
  auto subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors),
      builder.CreateVector(std::vector<int32_t>{0}),
      builder.CreateVector(std::vector<int32_t>{3}),
      builder.CreateVector(std::vector<flatbuffers::Offset<Operator>>{op}));
  auto op_code = CreateOperatorCode(
      builder, BuiltinOperator_EMBEDDING_LOOKUP, /*custom_code=*/0,
      /*version=*/1, BuiltinOperator_EMBEDDING_LOOKUP);
  auto model =
      CreateModel(builder, /*version=*/3, builder.CreateVector({op_code}),
                  builder.CreateVector({subgraph}), /*description=*/0,
                  builder.CreateVector(buffers));
  FinishModelBuffer(builder, model);
  return std::vector<uint8_t>(builder.GetBufferPointer(),
                              builder.GetBufferPointer() + builder.GetSize());
}

bool IsTableBlockwiseQuantized(const std::vector<uint8_t>& buffer) {
  const Model* model = GetModel(buffer.data());
  const SubGraph* subgraph = model->subgraphs()->Get(0);
  const Operator* op = subgraph->operators()->Get(0);
  OpSignature op_sig = GetOpSignature(
      model->operator_codes()->Get(op->opcode_index()), op, subgraph, model);
  free(op_sig.builtin_data);
  return op_sig.ext_options.embedding_lookup.is_blockwise_quantized;
}

TEST(GetOpSignature, EmbeddingLookupBlockwiseTable) {
  // N-D blockwise quantization (non-empty `block_shape`).
  EXPECT_TRUE(IsTableBlockwiseQuantized(
      BuildBlockwiseTableEmbeddingLookupModel(/*block_size=*/0, {1, 8})));
  // The legacy blockwise quantization only sets `block_size`.
  EXPECT_FALSE(IsTableBlockwiseQuantized(
      BuildBlockwiseTableEmbeddingLookupModel(/*block_size=*/8, {})));
}

}  // namespace tflite
