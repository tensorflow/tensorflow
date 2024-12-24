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
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"

namespace toco {

namespace {

void RunIdentifyL2Normalization(const std::vector<float>& input,
                                const std::vector<int>& input_shape,
                                const std::vector<int>& output_shape,
                                const bool div_square = false) {
  Model model;
  Array& input0 = model.GetOrCreateArray("input0");
  Array& output = model.GetOrCreateArray("output");

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto sq_op = new TensorFlowSquareOperator;
  sq_op->inputs = {"input0"};
  sq_op->outputs = {"output"};

  Array& sumoutput = model.GetOrCreateArray("Sumoutput");
  *sumoutput.mutable_shape()->mutable_dims() = output_shape;

  auto sum_op = new TensorFlowSumOperator;
  sum_op->inputs = {sq_op->outputs[0]};
  sum_op->outputs = {"Sumoutput"};

  if (div_square) {
    Array& sqrtoutput = model.GetOrCreateArray("squarertoutput");
    *sqrtoutput.mutable_shape()->mutable_dims() = output_shape;

    auto sqrt_op = new TensorFlowSqrtOperator;
    sqrt_op->inputs = {sum_op->outputs[0]};
    sqrt_op->outputs = {"squarertoutput"};

    Array& divoutput = model.GetOrCreateArray("Divoutput");
    *divoutput.mutable_shape()->mutable_dims() = output_shape;

    auto div_op = new DivOperator;
    div_op->inputs = {"input0", sqrt_op->outputs[0]};
    div_op->outputs = {"Divoutput"};

    /*Stack everything with the model*/
    model.operators.push_back(std::unique_ptr<Operator>(div_op));
    model.operators.push_back(std::unique_ptr<Operator>(sqrt_op));
    model.operators.push_back(std::unique_ptr<Operator>(sum_op));
    model.operators.push_back(std::unique_ptr<Operator>(sq_op));
  } else {
    Array& rsqoutput = model.GetOrCreateArray("Rsquareoutput");
    *rsqoutput.mutable_shape()->mutable_dims() = output_shape;

    auto rsqrt_op = new TensorFlowRsqrtOperator;
    rsqrt_op->inputs = {sum_op->outputs[0]};
    rsqrt_op->outputs = {"Rsquareoutput"};

    Array& muloutput = model.GetOrCreateArray("Muloutput");
    *muloutput.mutable_shape()->mutable_dims() = output_shape;

    auto mul_op = new MulOperator;
    mul_op->inputs = {"input0", rsqrt_op->outputs[0]};
    mul_op->outputs = {"Muloutput"};

    /*Stack everything with the model*/
    model.operators.push_back(std::unique_ptr<Operator>(mul_op));
    model.operators.push_back(std::unique_ptr<Operator>(rsqrt_op));
    model.operators.push_back(std::unique_ptr<Operator>(sum_op));
    model.operators.push_back(std::unique_ptr<Operator>(sq_op));
  }

  bool modified;
  ASSERT_TRUE(IdentifyL2Normalization().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kicked in we should not find any
    // Mul, Rsqrt, Add, Sqr  operators
    if (div_square) {
      EXPECT_FALSE(op->type == OperatorType::kDiv);
      EXPECT_FALSE(op->type == OperatorType::kSqrt);
    } else {
      EXPECT_FALSE(op->type == OperatorType::kMul);
      EXPECT_FALSE(op->type == OperatorType::kRsqrt);
    }
    EXPECT_FALSE(op->type == OperatorType::kAdd);
    EXPECT_FALSE(op->type == OperatorType::kSquare);
  }
}

// Test for reverse input in Min
TEST(IdentifyL2Normalization, MulRsqrtTest) {
  RunIdentifyL2Normalization(
      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4},

      false);
}

TEST(IdentifyL2Normalization, DivSqrtNormTest) {
  RunIdentifyL2Normalization(
      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4},

      true);
}

}  // namespace
}  // namespace toco
