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

void RunIdentifyL2Pool(const std::vector<float>& input,
                       const std::vector<int>& input_shape,
                       const std::vector<int>& output_shape) {
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

  Array& avgpooloutput = model.GetOrCreateArray("Avgpooloutput");
  *avgpooloutput.mutable_shape()->mutable_dims() = output_shape;

  auto avgpool_op = new AveragePoolOperator;
  avgpool_op->inputs = {sq_op->outputs[0]};
  avgpool_op->outputs = {"Avgpooloutput"};

  Array& sqrtoutput = model.GetOrCreateArray("Sqrtoutput");
  *sqrtoutput.mutable_shape()->mutable_dims() = output_shape;

  auto sqrt_op = new TensorFlowSqrtOperator;
  sqrt_op->inputs = {avgpool_op->outputs[0]};
  sqrt_op->outputs = {"Sqrtoutput"};

  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(sqrt_op));
  model.operators.push_back(std::unique_ptr<Operator>(avgpool_op));
  model.operators.push_back(std::unique_ptr<Operator>(sq_op));

  bool modified;
  ASSERT_TRUE(IdentifyL2Pool().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kicked in we should not find any
    // Square, avgpool & Sqrt  operators
    EXPECT_FALSE(op->type == OperatorType::kSqrt);
    EXPECT_FALSE(op->type == OperatorType::kAveragePool);
    EXPECT_FALSE(op->type == OperatorType::kSquare);
  }
}
}  // namespace

TEST(IdentifyL2Pool, SimpleTest) {
  RunIdentifyL2Pool(
      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4});
}

}  // namespace toco
