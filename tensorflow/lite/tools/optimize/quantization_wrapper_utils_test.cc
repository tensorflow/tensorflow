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

#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/schema/schema_utils.h"

namespace tflite {
namespace optimize {
namespace {

using ::testing::ElementsAreArray;

TEST(LstmPreprocess, Add2Tensors) {
  // Create a model with 1 lstm layer.
  auto model = std::make_unique<ModelT>();
  auto subgraph = std::make_unique<tflite::SubGraphT>();
  auto buffer = std::make_unique<tflite::BufferT>();
  auto lstm_op_code = std::make_unique<OperatorCodeT>();
  auto lstm_op = std::make_unique<OperatorT>();

  lstm_op_code->builtin_code = BuiltinOperator_LSTM;
  lstm_op_code->deprecated_builtin_code =
      static_cast<int8_t>(BuiltinOperator_LSTM);
  lstm_op_code->version = 2;
  lstm_op->opcode_index = 0;
  lstm_op->inputs = {0, 1,  2,  3,  4,  5,  6,  7,  8,  -1, -1, -1,
                     9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
  lstm_op->outputs = {24};

  model->subgraphs.push_back(std::move(subgraph));
  for (int i = 0; i < lstm_op->inputs.size(); ++i) {
    const int index = lstm_op->inputs[i];
    if (index == -1) {
      continue;
    }
    auto tensor = std::make_unique<TensorT>();
    tensor->name = "lstm_tensor" + std::to_string(index);
    tensor->shape = {2, 3, 4};
    tensor->type = TensorType_FLOAT32;
    model->subgraphs[0]->tensors.push_back(std::move(tensor));
  }
  model->subgraphs[0]->operators.push_back(std::move(lstm_op));
  model->operator_codes.push_back(std::move(lstm_op_code));
  model->buffers.push_back(std::move(buffer));

  // Add 2 tensors.
  flatbuffers::FlatBufferBuilder builder;
  tflite::optimize::AddIntermediateTensorsToFusedOp(&builder, model.get());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 1);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 26);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(GetBuiltinCode(model->operator_codes[0].get()),
            BuiltinOperator_LSTM);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "lstm_tensor0");
  EXPECT_EQ(model->subgraphs[0]->tensors[21]->name, "intermediate_0_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[22]->name, "intermediate_0_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[23]->name, "intermediate_0_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[24]->name, "intermediate_0_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[25]->name, "intermediate_0_4");
  EXPECT_THAT(
      model->subgraphs[0]->operators[0]->inputs,
      ElementsAreArray({0, 1,  2,  3,  4,  5,  6,  7,  8,  -1, -1, -1,
                        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({24}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->intermediates,
              ElementsAreArray({21, 22, 23, 24, 25}));

  // Call AddIntermediateTensorsToFusedOp again and expect no change in model.
  tflite::optimize::AddIntermediateTensorsToFusedOp(&builder, model.get());

  // Verify results.
  EXPECT_EQ(model->operator_codes.size(), 1);
  EXPECT_EQ(model->subgraphs.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->operators.size(), 1);
  EXPECT_EQ(model->subgraphs[0]->tensors.size(), 26);
  EXPECT_EQ(model->buffers.size(), 1);

  EXPECT_EQ(GetBuiltinCode(model->operator_codes[0].get()),
            BuiltinOperator_LSTM);
  EXPECT_EQ(model->subgraphs[0]->tensors[0]->name, "lstm_tensor0");
  EXPECT_EQ(model->subgraphs[0]->tensors[21]->name, "intermediate_0_0");
  EXPECT_EQ(model->subgraphs[0]->tensors[22]->name, "intermediate_0_1");
  EXPECT_EQ(model->subgraphs[0]->tensors[23]->name, "intermediate_0_2");
  EXPECT_EQ(model->subgraphs[0]->tensors[24]->name, "intermediate_0_3");
  EXPECT_EQ(model->subgraphs[0]->tensors[25]->name, "intermediate_0_4");
  EXPECT_THAT(
      model->subgraphs[0]->operators[0]->inputs,
      ElementsAreArray({0, 1,  2,  3,  4,  5,  6,  7,  8,  -1, -1, -1,
                        9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->outputs,
              ElementsAreArray({24}));
  EXPECT_THAT(model->subgraphs[0]->operators[0]->intermediates,
              ElementsAreArray({21, 22, 23, 24, 25}));
}

}  // namespace
}  // namespace optimize
}  // namespace tflite

int main(int argc, char** argv) { return RUN_ALL_TESTS(); }
