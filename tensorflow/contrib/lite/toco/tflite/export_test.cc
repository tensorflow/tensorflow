/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/contrib/lite/toco/tflite/export.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/contrib/lite/schema/schema_generated.h"

namespace toco {
namespace tflite {
namespace {

using ::testing::ElementsAre;

class ExportTest : public ::testing::Test {
 protected:
  // This is a very simplistic model. We are not interested in testing all the
  // details here, since tf.mini's testing framework will be exercising all the
  // conversions multiple times, and the conversion of operators is tested by
  // separate unittests.
  void BuildTestModel() {
    input_model_.GetOrCreateArray("tensor_one");
    input_model_.GetOrCreateArray("tensor_two");
    {
      auto* op = new ConvOperator;
      op->padding.type = PaddingType::kSame;
      input_model_.operators.emplace_back(op);
    }
    input_model_.operators.emplace_back(new AddOperator);
    {
      auto* op = new TensorFlowUnsupportedOperator;
      op->tensorflow_op = "MyCrazyOp";
      input_model_.operators.emplace_back(op);
    }
    // Note that Sub is not know to TF Lite, so it gets exported as a custom
    // op (and no options).
    input_model_.operators.emplace_back(new SubOperator);
  }

  Model input_model_;
};

TEST_F(ExportTest, LoadTensorsMap) {
  BuildTestModel();

  details::TensorsMap tensors;
  details::LoadTensorsMap(input_model_, &tensors);
  EXPECT_EQ(0, tensors["tensor_one"]);
  EXPECT_EQ(1, tensors["tensor_two"]);
}

TEST_F(ExportTest, LoadOperatorsMap) {
  BuildTestModel();

  details::OperatorsMap operators;
  details::LoadOperatorsMap(input_model_, &operators);
  EXPECT_EQ(0, operators[details::OperatorKey(OperatorType::kAdd, "")]);
  EXPECT_EQ(1, operators[details::OperatorKey(OperatorType::kConv, "")]);
  EXPECT_EQ(2, operators[details::OperatorKey(OperatorType::kSub, "")]);
  EXPECT_EQ(3, operators[details::OperatorKey(
                   OperatorType::kTensorFlowUnsupported, "MyCrazyOp")]);
}

TEST_F(ExportTest, Export) {
  BuildTestModel();

  string result;
  Export(input_model_, true, &result);

  auto* model = ::tflite::GetModel(result.data());

  std::vector<string> names;
  for (const ::tflite::OperatorCode* opcode : *model->operator_codes()) {
    if (opcode->builtin_code() != ::tflite::BuiltinOperator_CUSTOM) {
      names.push_back(string("builtin:") + ::tflite::EnumNameBuiltinOperator(
                                               opcode->builtin_code()));
    } else {
      names.push_back(string("custom:") + opcode->custom_code()->c_str());
    }
  }

  EXPECT_THAT(names, ElementsAre("builtin:ADD", "builtin:CONV_2D", "custom:Sub",
                                 "custom:MyCrazyOp"));

  std::vector<uint32_t> indices;
  auto operators = (*model->subgraphs())[0]->operators();
  EXPECT_EQ(operators->Length(), 4);
  for (const auto* op : *operators) {
    indices.push_back(op->opcode_index());
  }

  EXPECT_THAT(indices, ElementsAre(1, 0, 3, 2));
}

// TODO(ahentz): tests for tensors, inputs, outpus, opcodes and operators.

}  // namespace
}  // namespace tflite
}  // namespace toco
