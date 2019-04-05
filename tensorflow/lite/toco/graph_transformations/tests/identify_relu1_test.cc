/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void RunidentifyRelu1(const std::vector<float>& input,
                      const std::vector<int>& input_shape,
                      const std::vector<int>& output_shape,
                      const bool reversemaxmin = false) {
  Model model;
  Array& input0 = model.GetOrCreateArray("input0");

  // Filling the input dims, shape and data
  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  // Filling the Max value dims, shape and data
  Array& Maxinput = model.GetOrCreateArray("MaxValue");
  *Maxinput.mutable_shape()->mutable_dims() = {1};
  Maxinput.data_type = ArrayDataType::kFloat;
  Maxinput.GetMutableBuffer<ArrayDataType::kFloat>().data = {-1.0f};

  // Filling the Max_op output shape
  Array& output = model.GetOrCreateArray("output");
  *output.mutable_shape()->mutable_dims() = output_shape;

  auto max_op = new TensorFlowMaximumOperator;
  max_op->outputs = {"output"};

  Array& minoutput = model.GetOrCreateArray("minoutput");
  *minoutput.mutable_shape()->mutable_dims() = output_shape;

  // Filling the Min value dims, shape and data
  Array& Mininput = model.GetOrCreateArray("MinValue");
  *Mininput.mutable_shape()->mutable_dims() = {1};
  Mininput.data_type = ArrayDataType::kFloat;
  Mininput.GetMutableBuffer<ArrayDataType::kFloat>().data = {1.0f};

  auto min_op = new TensorFlowMinimumOperator;
  min_op->outputs = {"minoutput"};

  if (reversemaxmin) {
    max_op->inputs = {min_op->outputs[0], "MaxValue"};
    min_op->inputs = {"input0", "MinValue"};
    /*Stack everything with the model*/
    model.operators.push_back(std::unique_ptr<Operator>(max_op));
    model.operators.push_back(std::unique_ptr<Operator>(min_op));
  } else {
    min_op->inputs = {max_op->outputs[0], "MinValue"};
    max_op->inputs = {"input0", "MaxValue"};
    /*Stack everything with the model*/
    model.operators.push_back(std::unique_ptr<Operator>(min_op));
    model.operators.push_back(std::unique_ptr<Operator>(max_op));
  }

  bool modified;
  ASSERT_TRUE(IdentifyRelu1().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kicked in we should not find any
    // Min, Max, Neg operators
    EXPECT_FALSE(op->type == OperatorType::kMinimum);
    EXPECT_FALSE(op->type == OperatorType::kMaximum);
  }
}

// Test simple min, max
TEST(IdentifyRelu1, Relu1MaxMinTest) {
  RunidentifyRelu1(
      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4}, true);
}

// Test simple min, max
TEST(IdentifyRelu1, Relu1MinMaxTest) {
  RunidentifyRelu1(
      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4}, false);
}

}  // namespace
}  // namespace toco
