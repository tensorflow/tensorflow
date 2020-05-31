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
#include "tensorflow/lite/tools/optimize/modify_model_interface.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "absl/memory/memory.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace {

using ::testing::ElementsAreArray;

// Create a model with 1 quant, 1 FC, 1 dequant
std::unique_ptr<ModelT> CreateModelSingleInputOutput(const TensorType& quantization_type) {
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
  auto quant_op_code = absl::make_unique<OperatorCodeT>();
  auto quant_op = absl::make_unique<OperatorT>();
  auto fc_op_code = absl::make_unique<OperatorCodeT>();
  auto fc_op = absl::make_unique<OperatorT>();
  auto dequant_op_code = absl::make_unique<OperatorCodeT>();
  auto dequant_op = absl::make_unique<OperatorT>();

  model->subgraphs.push_back(std::move(subgraph));

  // Op code
  quant_op_code->builtin_code = BuiltinOperator_QUANTIZE;
  quant_op_code->version = 2;

  fc_op_code->builtin_code = BuiltinOperator_FULLY_CONNECTED;
  fc_op_code->version = 2;

  dequant_op_code->builtin_code = BuiltinOperator_DEQUANTIZE;
  dequant_op_code->version = 2;

  // Op.
  quant_op->opcode_index = 0;
  quant_op->inputs = {2};
  quant_op->outputs = {0};

  fc_op->opcode_index = 1;
  fc_op->inputs = {0};
  fc_op->outputs = {1};

  dequant_op->opcode_index = 2;
  dequant_op->inputs = {1};
  dequant_op->outputs = {3};

  model->subgraphs[0]->operators.push_back(std::move(quant_op));
  model->subgraphs[0]->operators.push_back(std::move(fc_op));
  model->subgraphs[0]->operators.push_back(std::move(dequant_op));

  model->operator_codes.push_back(std::move(quant_op_code));
  model->operator_codes.push_back(std::move(fc_op_code));
  model->operator_codes.push_back(std::move(dequant_op_code));

  // Model input/otuput.
  model->subgraphs[0]->inputs = {2};
  model->subgraphs[0]->outputs = {3};

  // Tensors. Float tensors are at the end of the tensor list.

  auto tensor_0 = absl::make_unique<TensorT>();
  tensor_0->quantization = absl::make_unique<QuantizationParametersT>();
  tensor_0->quantization->scale.push_back(0.35);
  tensor_0->quantization->zero_point.push_back(28);
  tensor_0->name = "tensor_0";
  tensor_0->shape = {};
  tensor_0->type = quantization_type;

  auto tensor_1 = absl::make_unique<TensorT>();
  tensor_1->quantization = absl::make_unique<QuantizationParametersT>();
  tensor_1->quantization->scale.push_back(0.12);
  tensor_1->quantization->zero_point.push_back(50);
  tensor_1->name = "tensor_1";
  tensor_1->shape = {};
  tensor_1->type = quantization_type;

  auto tensor_2 = absl::make_unique<TensorT>();
  tensor_2->name = "tensor_2";
  tensor_2->shape = {};
  tensor_2->type = TensorType_FLOAT32;

  auto tensor_3 = absl::make_unique<TensorT>();
  tensor_3->name = "tensor_3";
  tensor_3->shape = {};
  tensor_3->type = TensorType_FLOAT32;

  model->subgraphs[0]->tensors.push_back(std::move(tensor_0));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_1));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_2));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_3));

  // Buffer
  model->buffers.push_back(std::move(buffer));

  return model;
}

// Create a model with 2 quant, 1 FC, 2 dequant
// The model mimics the behavior of the quantize_model.cc.
std::unique_ptr<ModelT> CreateModelMultipleInputOutput(const TensorType& quantization_type) {
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
  auto quant_op_code = absl::make_unique<OperatorCodeT>();
  auto quant_op_1 = absl::make_unique<OperatorT>();
  auto quant_op_2 = absl::make_unique<OperatorT>();
  auto fc_op_code = absl::make_unique<OperatorCodeT>();
  auto fc_op = absl::make_unique<OperatorT>();
  auto dequant_op_code = absl::make_unique<OperatorCodeT>();
  auto dequant_op_1 = absl::make_unique<OperatorT>();
  auto dequant_op_2 = absl::make_unique<OperatorT>();

  model->subgraphs.push_back(std::move(subgraph));

  // Op code
  quant_op_code->builtin_code = BuiltinOperator_QUANTIZE;
  quant_op_code->version = 2;

  fc_op_code->builtin_code = BuiltinOperator_FULLY_CONNECTED;
  fc_op_code->version = 2;

  dequant_op_code->builtin_code = BuiltinOperator_DEQUANTIZE;
  dequant_op_code->version = 2;

  // Op.
  quant_op_1->opcode_index = 0;
  quant_op_1->inputs = {4};
  quant_op_1->outputs = {0};
  quant_op_2->opcode_index = 0;
  quant_op_2->inputs = {5};
  quant_op_2->outputs = {1};

  fc_op->opcode_index = 1;
  fc_op->inputs = {0, 1};
  fc_op->outputs = {2, 3};

  dequant_op_1->opcode_index = 2;
  dequant_op_1->inputs = {2};
  dequant_op_1->outputs = {6};
  dequant_op_2->opcode_index = 2;
  dequant_op_2->inputs = {3};
  dequant_op_2->outputs = {7};

  model->subgraphs[0]->operators.push_back(std::move(quant_op_1));
  model->subgraphs[0]->operators.push_back(std::move(quant_op_2));
  model->subgraphs[0]->operators.push_back(std::move(fc_op));
  model->subgraphs[0]->operators.push_back(std::move(dequant_op_1));
  model->subgraphs[0]->operators.push_back(std::move(dequant_op_2));

  model->operator_codes.push_back(std::move(quant_op_code));
  model->operator_codes.push_back(std::move(fc_op_code));
  model->operator_codes.push_back(std::move(dequant_op_code));

  // Model input/otuput.
  model->subgraphs[0]->inputs = {4, 5};
  model->subgraphs[0]->outputs = {6, 7};

  // Tensors
  auto tensor_0 = absl::make_unique<TensorT>();
  tensor_0->quantization = absl::make_unique<QuantizationParametersT>();
  tensor_0->quantization->scale.push_back(0.35);
  tensor_0->quantization->zero_point.push_back(28);
  tensor_0->name = "tensor_0";
  tensor_0->shape = {};
  tensor_0->type = quantization_type;

  auto tensor_1 = absl::make_unique<TensorT>();
  tensor_1->quantization = absl::make_unique<QuantizationParametersT>();
  tensor_1->quantization->scale.push_back(0.12);
  tensor_1->quantization->zero_point.push_back(50);
  tensor_1->name = "tensor_1";
  tensor_1->shape = {};
  tensor_1->type = quantization_type;

  auto tensor_2 = absl::make_unique<TensorT>();
  tensor_2->quantization = absl::make_unique<QuantizationParametersT>();
  tensor_2->quantization->scale.push_back(0.45);
  tensor_2->quantization->zero_point.push_back(28);
  tensor_2->name = "tensor_2";
  tensor_2->shape = {};
  tensor_2->type = quantization_type;

  auto tensor_3 = absl::make_unique<TensorT>();
  tensor_3->quantization = absl::make_unique<QuantizationParametersT>();
  tensor_3->quantization->scale.push_back(0.22);
  tensor_3->quantization->zero_point.push_back(50);
  tensor_3->name = "tensor_3";
  tensor_3->shape = {};
  tensor_3->type = quantization_type;

  auto tensor_4 = absl::make_unique<TensorT>();
  tensor_4->name = "tensor_4";
  tensor_4->shape = {};
  tensor_4->type = TensorType_FLOAT32;

  auto tensor_5 = absl::make_unique<TensorT>();
  tensor_5->name = "tensor_5";
  tensor_5->shape = {};
  tensor_5->type = TensorType_FLOAT32;

  auto tensor_6 = absl::make_unique<TensorT>();
  tensor_6->name = "tensor_6";
  tensor_6->shape = {};
  tensor_6->type = TensorType_FLOAT32;

  auto tensor_7 = absl::make_unique<TensorT>();
  tensor_7->name = "tensor_7";
  tensor_7->shape = {};
  tensor_7->type = TensorType_FLOAT32;

  model->subgraphs[0]->tensors.push_back(std::move(tensor_0));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_1));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_2));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_3));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_4));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_5));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_6));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_7));

  // Buffer
  model->buffers.push_back(std::move(buffer));

  return model;
}

// Create a model with 1 FC.
std::unique_ptr<ModelT> CreateFloatModel() {
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
  auto fc_op_code = absl::make_unique<OperatorCodeT>();
  auto fc_op = absl::make_unique<OperatorT>();

  model->subgraphs.push_back(std::move(subgraph));

  // Op code
  fc_op_code->builtin_code = BuiltinOperator_FULLY_CONNECTED;
  fc_op_code->version = 2;

  // Op.
  fc_op->opcode_index = 0;
  fc_op->inputs = {0};
  fc_op->outputs = {1};

  model->subgraphs[0]->operators.push_back(std::move(fc_op));
  model->operator_codes.push_back(std::move(fc_op_code));

  // Model input/otuput.
  model->subgraphs[0]->inputs = {0};
  model->subgraphs[0]->outputs = {1};

  // Tensors
  auto tensor_0 = absl::make_unique<TensorT>();
  tensor_0->name = "tensor_0";
  tensor_0->shape = {};
  tensor_0->type = TensorType_FLOAT32;

  auto tensor_1 = absl::make_unique<TensorT>();
  tensor_1->name = "tensor_1";
  tensor_1->shape = {};
  tensor_1->type = TensorType_FLOAT32;

  model->subgraphs[0]->tensors.push_back(std::move(tensor_0));
  model->subgraphs[0]->tensors.push_back(std::move(tensor_1));

  // Buffer
  model->buffers.push_back(std::move(buffer));

  return model;
}

struct ModelInterface:
  ::testing::TestWithParam<tflite::TensorType> {};

TEST_P(ModelInterface, SingleInputOutput) {
  TensorType quantization_type = GetParam();

  auto model = CreateModelSingleInputOutput(quantization_type);

  // Change model type.
  flatbuffers::FlatBufferBuilder builder;
  EXPECT_EQ(ModifyModelInterface(&builder, model.get(), quantization_type,
                                 quantization_type),
            kTfLiteOk);

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 3);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 2);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(model->subgraphs[0]->inputs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->inputs[0], 0);
  EXPECT_EQ(model->subgraphs[0]->outputs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->outputs[0], 1);
}

TEST_P(ModelInterface, MutipleInputOutput) {

  TensorType quantization_type = GetParam();

  auto model = CreateModelMultipleInputOutput(quantization_type);

  // Change model type.
  flatbuffers::FlatBufferBuilder builder;
  EXPECT_EQ(ModifyModelInterface(&builder, model.get(), quantization_type,
                                 quantization_type),
            kTfLiteOk);

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 3);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 4);
  EXPECT_EQ(model->subgraphs[0]->inputs.size(), 2);
  EXPECT_EQ(model->subgraphs[0]->inputs[0], 0);
  EXPECT_EQ(model->subgraphs[0]->inputs[1], 1);
  EXPECT_EQ(model->subgraphs[0]->outputs.size(), 2);
  EXPECT_EQ(model->subgraphs[0]->outputs[0], 2);
  EXPECT_EQ(model->subgraphs[0]->outputs[1], 3);
  EXPECT_EQ(model->buffers.size(), 1);

  // Tensors,
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "tensor_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, quantization_type);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[0]->quantization->scale[0],
                  0.35);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->quantization->zero_point[0], 28);

  EXPECT_EQ(model->subgraphs[0]->tensors[1]->name, "tensor_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[1]->type, quantization_type);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[1]->quantization->scale[0],
                  0.12);
  EXPECT_EQ(model->subgraphs[0]->tensors[1]->quantization->zero_point[0], 50);

  EXPECT_EQ(model->subgraphs[0]->tensors[2]->name, "tensor_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->type, quantization_type);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[2]->quantization->scale[0],
                  0.45);
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->quantization->zero_point[0], 28);

  EXPECT_EQ(model->subgraphs[0]->tensors[3]->name, "tensor_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->type, quantization_type);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[3]->quantization->scale[0],
                  0.22);
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->quantization->zero_point[0], 50);

  // Ops.
  EXPECT_EQ(model->subgraphs[0]->operators[0]->opcode_index, 1);
}

INSTANTIATE_TEST_SUITE_P(
  MultipleInputOutputTests,
  ModelInterface,
  ::testing::Values(TensorType_INT8, TensorType_INT16)
);

TEST(ModelInterface, Uint8SingleInputOutput) {
  auto model = CreateModelSingleInputOutput(TensorType_INT8);

  // Ops.
  EXPECT_EQ(model->subgraphs[0]->operators[0]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[1]->opcode_index, 1);
  EXPECT_EQ(model->subgraphs[0]->operators[2]->opcode_index, 2);

  // Change model type.
  flatbuffers::FlatBufferBuilder builder;
  EXPECT_EQ(ModifyModelInterface(&builder, model.get(), TensorType_UINT8,
                                 TensorType_UINT8),
            kTfLiteOk);

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 3);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 3);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 4);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(model->subgraphs[0]->tensors[2]->name, "tensor_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[2]->quantization->scale[0],
                  0.35);
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->quantization->zero_point[0], 156);

  EXPECT_EQ(model->subgraphs[0]->tensors[3]->name, "tensor_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[3]->quantization->scale[0],
                  0.12);
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->quantization->zero_point[0], 178);

  // Ops.
  EXPECT_EQ(model->subgraphs[0]->operators[0]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[1]->opcode_index, 1);
  EXPECT_EQ(model->subgraphs[0]->operators[2]->opcode_index, 0);
}

TEST(ModelInterface, Uint8MutipleInputOutput) {
  auto model = CreateModelMultipleInputOutput(TensorType_INT8);

  // Ops.
  EXPECT_EQ(model->subgraphs[0]->operators[0]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[1]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[2]->opcode_index, 1);
  EXPECT_EQ(model->subgraphs[0]->operators[3]->opcode_index, 2);
  EXPECT_EQ(model->subgraphs[0]->operators[4]->opcode_index, 2);

  // Change model type.
  flatbuffers::FlatBufferBuilder builder;
  EXPECT_EQ(ModifyModelInterface(&builder, model.get(), TensorType_UINT8,
                                 TensorType_UINT8),
            kTfLiteOk);

  // Verify results.
  // Model.
  EXPECT_EQ(model->operator_codes.size(), 3);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 5);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 8);
  EXPECT_EQ(model->subgraphs[0]->inputs.size(), 2);
  EXPECT_EQ(model->subgraphs[0]->inputs[0], 4);
  EXPECT_EQ(model->subgraphs[0]->inputs[1], 5);
  EXPECT_EQ(model->subgraphs[0]->outputs.size(), 2);
  EXPECT_EQ(model->subgraphs[0]->outputs[0], 6);
  EXPECT_EQ(model->subgraphs[0]->outputs[1], 7);
  EXPECT_EQ(model->buffers.size(), 1);

  // Tensors,
  EXPECT_EQ(model->subgraphs[0]->tensors[4]->name, "tensor_4");
  EXPECT_EQ(model->subgraphs[0]->tensors[4]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[4]->quantization->scale[0],
                  0.35);
  EXPECT_EQ(model->subgraphs[0]->tensors[4]->quantization->zero_point[0], 156);

  EXPECT_EQ(model->subgraphs[0]->tensors[5]->name, "tensor_5");
  EXPECT_EQ(model->subgraphs[0]->tensors[5]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[5]->quantization->scale[0],
                  0.12);
  EXPECT_EQ(model->subgraphs[0]->tensors[5]->quantization->zero_point[0], 178);

  EXPECT_EQ(model->subgraphs[0]->tensors[6]->name, "tensor_6");
  EXPECT_EQ(model->subgraphs[0]->tensors[6]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[6]->quantization->scale[0],
                  0.45);
  EXPECT_EQ(model->subgraphs[0]->tensors[6]->quantization->zero_point[0], 156);

  EXPECT_EQ(model->subgraphs[0]->tensors[7]->name, "tensor_7");
  EXPECT_EQ(model->subgraphs[0]->tensors[7]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[7]->quantization->scale[0],
                  0.22);
  EXPECT_EQ(model->subgraphs[0]->tensors[7]->quantization->zero_point[0], 178);

  // Ops.
  EXPECT_EQ(model->subgraphs[0]->operators[0]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[1]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[2]->opcode_index, 1);
  EXPECT_EQ(model->subgraphs[0]->operators[3]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[4]->opcode_index, 0);
}

TEST(ModelInterface, Float) {
  // Create the model.
  std::unique_ptr<ModelT> input_model_t = CreateFloatModel();
  flatbuffers::FlatBufferBuilder builder_temp;
  flatbuffers::Offset<Model> output_model_location =
      Model::Pack(builder_temp, input_model_t.get());
  FinishModelBuffer(builder_temp, output_model_location);
  const uint8_t* buffer_temp = builder_temp.GetBufferPointer();
  const Model* input_model = GetModel(buffer_temp);

  // Change model type.
  flatbuffers::FlatBufferBuilder builder;
  EXPECT_EQ(Uint8QuantizeModelInputsOutputs(&builder, input_model,
                                            {{"tensor_0", {0.4, 2}}},
                                            {{"tensor_1", {0.5, -5}}}),
            kTfLiteOk);

  const uint8_t* buffer = builder.GetBufferPointer();
  const Model* output_model = GetModel(buffer);
  std::unique_ptr<ModelT> model;
  model.reset(output_model->UnPack());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 3);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 3);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 4);
  EXPECT_EQ(model->buffers.size(), 1);

  // Ops.
  EXPECT_EQ(model->operator_codes[0]->builtin_code,
            BuiltinOperator_FULLY_CONNECTED);
  EXPECT_EQ(model->operator_codes[1]->builtin_code, BuiltinOperator_DEQUANTIZE);
  EXPECT_EQ(model->operator_codes[2]->builtin_code, BuiltinOperator_QUANTIZE);

  EXPECT_EQ(model->subgraphs[0]->operators[0]->opcode_index, 1);
  EXPECT_EQ(model->subgraphs[0]->operators[1]->opcode_index, 0);
  EXPECT_EQ(model->subgraphs[0]->operators[2]->opcode_index, 2);

  EXPECT_THAT(model->subgraphs[0]->operators[0]->inputs, ElementsAreArray({2}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({0}));
  EXPECT_THAT(model->subgraphs[0]->operators[1]->inputs, ElementsAreArray({0}));
  EXPECT_THAT(model->subgraphs[0]->operators[1]->outputs,
              ElementsAreArray({1}));
  EXPECT_THAT(model->subgraphs[0]->operators[2]->inputs, ElementsAreArray({1}));
  EXPECT_THAT(model->subgraphs[0]->operators[2]->outputs,
              ElementsAreArray({3}));

  // Tensors.
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "tensor_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->type, TensorType_FLOAT32);
  EXPECT_EQ(model->subgraphs[0]->tensors[1]->name, "tensor_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[1]->type, TensorType_FLOAT32);

  EXPECT_EQ(model->subgraphs[0]->tensors[2]->name, "tensor_0_uint8");
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[2]->quantization->scale[0], 0.4);
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->quantization->zero_point[0], 2);

  EXPECT_EQ(model->subgraphs[0]->tensors[3]->name, "tensor_1_uint8");
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->type, TensorType_UINT8);
  EXPECT_FLOAT_EQ(model->subgraphs[0]->tensors[3]->quantization->scale[0], 0.5);
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->quantization->zero_point[0], -5);
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
