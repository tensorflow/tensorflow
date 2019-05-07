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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void Runidentifyprelu(const std::vector<float>& input,
                      const std::vector<int>& input_shape,
                      const std::vector<int>& output_shape,
                      const bool condition,
                      const bool commutative_add = false) {
  Model model;
  Array& input0 = model.GetOrCreateArray("input0");
  Array& output = model.GetOrCreateArray("output");

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *output.mutable_shape()->mutable_dims() = output_shape;

  // We need to contruct a model which support PReLU implmentation as:
  // f(x) = Relu(x) + (negative_alpha * Neg(x, activation=Relu))

  auto relu_op = new ReluOperator;
  relu_op->inputs = {"input0"};
  relu_op->outputs = {"output"};
  relu_op->fused_activation_function = FusedActivationFunctionType::kNone;

  Array& Negoutput = model.GetOrCreateArray("Negoutput");
  *Negoutput.mutable_shape()->mutable_dims() = output_shape;

  auto neg_op = new NegOperator;
  neg_op->outputs = {"Negoutput"};
  if (condition) {
    neg_op->inputs = {"input0"};
    neg_op->fused_activation_function = FusedActivationFunctionType::kRelu;
  } else {
    neg_op->inputs = {relu_op->outputs[0]};
    neg_op->fused_activation_function = FusedActivationFunctionType::kNone;
  }
  Array& Neginput = model.GetOrCreateArray("Negalpha");

  *Neginput.mutable_shape()->mutable_dims() = {1};
  Neginput.data_type = ArrayDataType::kFloat;
  Neginput.GetMutableBuffer<ArrayDataType::kFloat>().data = {0.5f};

  Array& Muloutput = model.GetOrCreateArray("Muloutput");
  *Muloutput.mutable_shape()->mutable_dims() = output_shape;

  auto mul_op = new MulOperator;
  mul_op->inputs = {"Negalpha", neg_op->outputs[0]};
  mul_op->outputs = {"Muloutput"};

  Array& addoutput = model.GetOrCreateArray("addoutput");
  *addoutput.mutable_shape()->mutable_dims() = output_shape;

  auto add_op = new AddOperator;
  add_op->outputs = {"addoutput"};
  if (commutative_add) {
    add_op->inputs = {mul_op->outputs[0], relu_op->outputs[0]};
  } else {
    add_op->inputs = {relu_op->outputs[0], mul_op->outputs[0]};
  }

  add_op->fused_activation_function = FusedActivationFunctionType::kNone;
  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(add_op));
  model.operators.push_back(std::unique_ptr<Operator>(relu_op));

  model.operators.push_back(std::unique_ptr<Operator>(mul_op));
  model.operators.push_back(std::unique_ptr<Operator>(neg_op));

  bool modified;
  ASSERT_TRUE(IdentifyPRelu().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kickd in we should not find any
    // Add, Mul, Neg and Relu operators
    EXPECT_FALSE(op->type == OperatorType::kAdd);
    EXPECT_FALSE(op->type == OperatorType::kRelu);
    EXPECT_FALSE(op->type == OperatorType::kMul);
    EXPECT_FALSE(op->type == OperatorType::kNeg);
  }
}

TEST(IdentifyPRelu, PreluNegReluTest) {
  Runidentifyprelu(
      // Input data
      {3, 1, 4, 1,   //
       5, 9, 2, 6,   //
       5, 3, 5, 8},  //
      // Input shape
      {3, 4},
      // Expected output shape,
      {3, 4},
      // condition
      true);
}

TEST(IdentifyPRelu, PreluReluNegTest) {
  Runidentifyprelu(
      // Input data
      {3, 1, 4, 1,   //
       5, 9, 2, 6,   //
       5, 3, 5, 8},  //
      // Input shape
      {3, 4},
      // Expected output shape,
      {3, 4},
      // condition
      false);
}

}  // namespace
}  // namespace toco
