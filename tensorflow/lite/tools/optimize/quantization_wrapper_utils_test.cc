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
#include "tensorflow/lite/tools/optimize/quantization_wrapper_utils.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace optimize {
namespace {

using ::testing::ElementsAreArray;

TEST(LstmPreprocess, Add2Tensors) {
  // Create a model with 1 lstm layer.
  auto model = absl::make_unique<ModelT>();
  auto subgraph = absl::make_unique<tflite::SubGraphT>();
  auto tensor = absl::make_unique<TensorT>();
  auto buffer = absl::make_unique<tflite::BufferT>();
  auto lstm_op_code = absl::make_unique<OperatorCodeT>();
  auto lstm_op = absl::make_unique<OperatorT>();

  tensor->name = "lstm_tensor0";
  tensor->shape = {2, 3, 4};
  tensor->type = TensorType_FLOAT32;
  lstm_op_code->builtin_code = BuiltinOperator_LSTM;
  lstm_op_code->version = 2;
  lstm_op->opcode_index = 0;
  lstm_op->inputs = {0};
  lstm_op->outputs = {0};

  model->subgraphs.push_back(std::move(subgraph));
  model->subgraphs[0]->operators.push_back(std::move(lstm_op));
  model->subgraphs[0]->tensors.push_back(std::move(tensor));
  model->operator_codes.push_back(std::move(lstm_op_code));
  model->buffers.push_back(std::move(buffer));

  // Add 2 tensors.
  flatbuffers::FlatBufferBuilder builder;
  tflite::optimize::AddIntemediateTensorsToFusedOp(&builder, model.get());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 1);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 6);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(model->operator_codes[0]->builtin_code, BuiltinOperator_LSTM);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "lstm_tensor0");
  EXPECT_EQ(model->subgraphs[0]->tensors[1]->name, "intermediate_0_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->name, "intermediate_0_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->name, "intermediate_0_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[4]->name, "intermediate_0_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[5]->name, "intermediate_0_4");
  EXPECT_THAT(model->subgraphs[0]->operators[0]->inputs, ElementsAreArray({0}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({0}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->intermediates,
              ElementsAreArray({1, 2, 3, 4, 5}));

  // Call AddIntemediateTensorsToFusedOp again and expect no change in model.
  tflite::optimize::AddIntemediateTensorsToFusedOp(&builder, model.get());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 1);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 6);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(model->operator_codes[0]->builtin_code, BuiltinOperator_LSTM);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "lstm_tensor0");
  EXPECT_EQ(model->subgraphs[0]->tensors[1]->name, "intermediate_0_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[2]->name, "intermediate_0_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[3]->name, "intermediate_0_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[4]->name, "intermediate_0_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[5]->name, "intermediate_0_4");
  EXPECT_THAT(model->subgraphs[0]->operators[0]->inputs, ElementsAreArray({0}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({0}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->intermediates,
              ElementsAreArray({1, 2, 3, 4, 5}));
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) { return RUN_ALL_TESTS(); }
