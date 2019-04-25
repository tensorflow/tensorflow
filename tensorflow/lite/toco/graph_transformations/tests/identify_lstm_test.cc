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
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

void RunidentifyLstmCell(const std::vector<float>& input,
                         const std::vector<int>& input_shape,
                         const std::vector<float>& prev_input,
                         const std::vector<int>& prev_input_shape) {
  Model model;
  Array& input0 = model.GetOrCreateArray("input0");
  Array& input1 = model.GetOrCreateArray("input1");
  Array& output = model.GetOrCreateArray("output");

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *input1.mutable_shape()->mutable_dims() = prev_input_shape;
  input1.data_type = ArrayDataType::kFloat;
  input1.GetMutableBuffer<ArrayDataType::kFloat>().data = prev_input;

  output.data_type = ArrayDataType::kFloat;

  auto con_op = new ConcatenationOperator;
  con_op->inputs = {"input0", "input1"};
  con_op->outputs = {"output"};

  Array& fcoutput = model.GetOrCreateArray("Fcoutput");
  fcoutput.data_type = ArrayDataType::kFloat;

  auto fc_op = new FullyConnectedOperator;
  fc_op->inputs = {con_op->outputs[0], "input1", "input0"};
  fc_op->outputs = {"Fcoutput"};

  Array& splitoutput = model.GetOrCreateArray("Splitoutput");
  splitoutput.data_type = ArrayDataType::kFloat;

  auto split_op = new TensorFlowSplitOperator;
  split_op->inputs = {"input1", fc_op->outputs[0]};
  // Dummy out the outputs
  split_op->outputs = {"Splitoutput", "Splitoutput", "Splitoutput",
                       "Splitoutput"};

  Array& tanhoutput = model.GetOrCreateArray("tanhoutput");
  tanhoutput.data_type = ArrayDataType::kFloat;

  auto tanh_op = new TanhOperator;
  tanh_op->inputs = {split_op->outputs[0]};
  tanh_op->outputs = {"tanhoutput"};

  Array& sigoutput = model.GetOrCreateArray("Sigoutput");
  sigoutput.data_type = ArrayDataType::kFloat;

  // This forms the input to input gate
  auto log_op = new LogisticOperator;
  log_op->inputs = {split_op->outputs[0]};
  log_op->outputs = {"Sigoutput"};

  Array& muloutput = model.GetOrCreateArray("Muloutput");
  muloutput.data_type = ArrayDataType::kFloat;

  // This forms the one of the inputs to forget gate
  auto mul_op = new MulOperator;
  mul_op->inputs = {log_op->outputs[0], tanh_op->outputs[0]};
  mul_op->outputs = {"Muloutput"};

  Array& fgmuloutput = model.GetOrCreateArray("Fgmuloutput");
  fgmuloutput.data_type = ArrayDataType::kFloat;

  // This forms the one of the inputs to forget gate
  auto fgmul_op = new MulOperator;
  fgmul_op->inputs = {"input1", log_op->outputs[0]};
  fgmul_op->outputs = {"Fgmuloutput"};

  Array& addoutput = model.GetOrCreateArray("addoutput");
  addoutput.data_type = ArrayDataType::kFloat;

  auto add_op = new AddOperator;
  add_op->inputs = {fgmul_op->outputs[0], mul_op->outputs[0]};
  add_op->outputs = {"addoutput"};

  Array& ogtanhoutput = model.GetOrCreateArray("Ogtanhoutput");
  ogtanhoutput.data_type = ArrayDataType::kFloat;

  auto ogtanh_op = new TanhOperator;
  ogtanh_op->inputs = {add_op->outputs[0]};
  ogtanh_op->outputs = {"Ogtanhoutput"};

  Array& ogsigoutput = model.GetOrCreateArray("Ogsigoutput");
  ogsigoutput.data_type = ArrayDataType::kFloat;

  auto ogsig_op = new LogisticOperator;
  ogsig_op->inputs = {split_op->outputs[0]};
  ogsig_op->outputs = {"Ogsigoutput"};

  Array& finaloutput = model.GetOrCreateArray("Finaloutput");
  finaloutput.data_type = ArrayDataType::kFloat;

  auto final_op = new MulOperator;
  final_op->inputs = {ogtanh_op->outputs[0], ogsig_op->outputs[0]};
  final_op->outputs = {"Finaloutput"};

  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(final_op));
  model.operators.push_back(std::unique_ptr<Operator>(ogsig_op));
  model.operators.push_back(std::unique_ptr<Operator>(ogtanh_op));
  model.operators.push_back(std::unique_ptr<Operator>(add_op));
  model.operators.push_back(std::unique_ptr<Operator>(fgmul_op));
  model.operators.push_back(std::unique_ptr<Operator>(mul_op));
  model.operators.push_back(std::unique_ptr<Operator>(log_op));
  model.operators.push_back(std::unique_ptr<Operator>(tanh_op));
  model.operators.push_back(std::unique_ptr<Operator>(split_op));
  model.operators.push_back(std::unique_ptr<Operator>(fc_op));
  model.operators.push_back(std::unique_ptr<Operator>(con_op));

  bool modified;
  ASSERT_TRUE(IdentifyLstmCell().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kicked in we should not find any
    // Concat, FC, Split, Tanh, Logistic, Mul & Add  operators
    EXPECT_FALSE(op->type == OperatorType::kConcatenation);
    EXPECT_FALSE(op->type == OperatorType::kFullyConnected);
    EXPECT_FALSE(op->type == OperatorType::kSplit);
    EXPECT_FALSE(op->type == OperatorType::kTanh);
    EXPECT_FALSE(op->type == OperatorType::kLogistic);
    EXPECT_FALSE(op->type == OperatorType::kMul);
    EXPECT_FALSE(op->type == OperatorType::kAdd);
  }
}
}  // namespace

TEST(IdentifyLstmCell, SimpleTest) {
  RunidentifyLstmCell(
      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      // Prev state data
      {2, 1, 5, 1, -5, 9, -4, 6, 5, 7, 5, 8},

      // Prev state output shape
      {3, 4});
}

}  // namespace toco
