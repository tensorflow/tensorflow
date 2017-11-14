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

namespace toco {

namespace tflite {
namespace {

class ExportTest : public ::testing::Test {
 protected:
  // This is a very simplistic model. We are not interested in testing all the
  // details here, since tf.mini's testing framework will be exercising all the
  // conversions multiple times, and the conversion of operators is tested by
  // separate unittests.
  void BuildTestModel() {
    input_model_.GetOrCreateArray("tensor_one");
    input_model_.GetOrCreateArray("tensor_two");
    input_model_.operators.emplace_back(new ConvOperator);
    input_model_.operators.emplace_back(new AddOperator);
    auto unsupported_operator = new TensorFlowUnsupportedOperator;
    unsupported_operator->tensorflow_op = "MyCrazyOp";
    input_model_.operators.emplace_back(unsupported_operator);
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
  EXPECT_EQ(2, operators[details::OperatorKey(
                   OperatorType::kTensorFlowUnsupported, "MyCrazyOp")]);
}

// TODO(ahentz): tests for tensors, inputs, outpus, opcodes and operators.

}  // namespace
}  // namespace tflite

}  // namespace toco
