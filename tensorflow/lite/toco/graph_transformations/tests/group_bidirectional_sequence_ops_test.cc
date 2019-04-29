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

void RunGroupBidirectionalSequencelstm(const std::vector<float>& input,
                                       const std::vector<int>& input_shape,
                                       const std::vector<int>& output_shape) {
  Model model;
  Array& input0 = model.GetOrCreateArray("input0");
  Array& output = model.GetOrCreateArray("output");

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto pack_op = new PackOperator;
  pack_op->inputs = {"input0"};
  pack_op->outputs = {"output"};

  Array& fwlstmoutput = model.GetOrCreateArray("fwlstmoutput");
  *fwlstmoutput.mutable_shape()->mutable_dims() = output_shape;

  const int kInputsSize = 20;
  auto fwlstm_op = new UnidirectionalSequenceLstmOperator;
  fwlstm_op->inputs.resize(kInputsSize);
  fwlstm_op->inputs = {pack_op->outputs[0]};

  // Filling the  dummy input based on the time series input
  for (int idx = 1; idx < kInputsSize; idx++) {
    fwlstm_op->inputs[idx] = pack_op->inputs[0];
  }

  fwlstm_op->outputs = {"fwlstmoutput"};

  Array& bwlstmoutput = model.GetOrCreateArray("bwlstmoutput");
  *bwlstmoutput.mutable_shape()->mutable_dims() = output_shape;

  auto bwlstm_op = new UnidirectionalSequenceLstmOperator;
  bwlstm_op->inputs.resize(kInputsSize);
  bwlstm_op->inputs = {pack_op->outputs[0]};

  // Filling the  dummy input based on the time series input
  for (int idx = 1; idx < kInputsSize; idx++) {
    bwlstm_op->inputs[idx] = pack_op->inputs[0];
  }

  bwlstm_op->outputs = {"bwlstmoutput"};

  Array& unpackbwoutput = model.GetOrCreateArray("unpackbwoutput");
  *unpackbwoutput.mutable_shape()->mutable_dims() = output_shape;

  auto unpackbw_op = new UnpackOperator;
  unpackbw_op->inputs = {bwlstm_op->outputs[0]};
  unpackbw_op->outputs = {"unpackbwoutput"};

  Array& unpackfwoutput = model.GetOrCreateArray("unpackfwoutput");
  *unpackfwoutput.mutable_shape()->mutable_dims() = output_shape;

  auto unpackfw_op = new UnpackOperator;
  unpackfw_op->inputs = {fwlstm_op->outputs[0]};
  unpackfw_op->outputs = {"unpackfwoutput"};

  Array& concatoutput = model.GetOrCreateArray("concatoutput");
  *concatoutput.mutable_shape()->mutable_dims() = output_shape;

  auto concat_op = new ConcatenationOperator;
  concat_op->inputs = {unpackfw_op->outputs[0], unpackbw_op->outputs[0]};
  concat_op->outputs = {"concatoutput"};

  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(concat_op));
  model.operators.push_back(std::unique_ptr<Operator>(unpackfw_op));
  model.operators.push_back(std::unique_ptr<Operator>(unpackbw_op));
  model.operators.push_back(std::unique_ptr<Operator>(bwlstm_op));
  model.operators.push_back(std::unique_ptr<Operator>(fwlstm_op));
  model.operators.push_back(std::unique_ptr<Operator>(pack_op));

  bool modified;
  ASSERT_TRUE(GroupBidirectionalSequenceLstm().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kicked in we should not find any
    // UnidirectionalSequenceLstmOperator
    EXPECT_FALSE(op->type == OperatorType::kUnidirectionalSequenceLstm);
  }
}

void RunGroupBidirectionalSequenceRnn(const std::vector<float>& input,
                                      const std::vector<int>& input_shape,
                                      const std::vector<int>& output_shape) {
  Model model;
  Array& input0 = model.GetOrCreateArray("input0");
  Array& output = model.GetOrCreateArray("output");

  *input0.mutable_shape()->mutable_dims() = input_shape;
  input0.data_type = ArrayDataType::kFloat;
  input0.GetMutableBuffer<ArrayDataType::kFloat>().data = input;

  *output.mutable_shape()->mutable_dims() = output_shape;

  auto pack_op = new PackOperator;
  pack_op->inputs = {"input0"};
  pack_op->outputs = {"output"};

  Array& fwrnnoutput = model.GetOrCreateArray("fwrnnoutput");
  *fwrnnoutput.mutable_shape()->mutable_dims() = output_shape;

  const int kInputsSize = 20;
  auto fwrnn_op = new UnidirectionalSequenceRnnOperator;
  fwrnn_op->inputs.resize(kInputsSize);
  fwrnn_op->inputs = {pack_op->outputs[0]};

  // Filling the  dummy input based on the time series input
  for (int idx = 1; idx < kInputsSize; idx++) {
    fwrnn_op->inputs[idx] = pack_op->inputs[0];
  }

  fwrnn_op->outputs = {"fwrnnoutput"};

  Array& bwrnnoutput = model.GetOrCreateArray("bwrnnoutput");
  *bwrnnoutput.mutable_shape()->mutable_dims() = output_shape;

  auto bwrnn_op = new UnidirectionalSequenceRnnOperator;
  bwrnn_op->inputs.resize(kInputsSize);
  bwrnn_op->inputs = {pack_op->outputs[0]};

  // Filling the  dummy input based on the time series input
  for (int idx = 1; idx < kInputsSize; idx++) {
    bwrnn_op->inputs[idx] = pack_op->inputs[0];
  }

  bwrnn_op->outputs = {"bwrnnoutput"};

  Array& unpackbwoutput = model.GetOrCreateArray("unpackbwoutput");
  *unpackbwoutput.mutable_shape()->mutable_dims() = output_shape;

  auto unpackbw_op = new UnpackOperator;
  unpackbw_op->inputs = {bwrnn_op->outputs[0]};
  unpackbw_op->outputs = {"unpackbwoutput"};

  Array& unpackfwoutput = model.GetOrCreateArray("unpackfwoutput");
  *unpackfwoutput.mutable_shape()->mutable_dims() = output_shape;

  auto unpackfw_op = new UnpackOperator;
  unpackfw_op->inputs = {fwrnn_op->outputs[0]};
  unpackfw_op->outputs = {"unpackfwoutput"};

  Array& concatoutput = model.GetOrCreateArray("concatoutput");
  *concatoutput.mutable_shape()->mutable_dims() = output_shape;

  auto concatv2_op = new TensorFlowConcatV2Operator;
  concatv2_op->inputs = {unpackfw_op->outputs[0], unpackbw_op->outputs[0]};
  concatv2_op->outputs = {"concatoutput"};

  /*Stack everything with the model*/
  model.operators.push_back(std::unique_ptr<Operator>(concatv2_op));
  model.operators.push_back(std::unique_ptr<Operator>(unpackfw_op));
  model.operators.push_back(std::unique_ptr<Operator>(unpackbw_op));
  model.operators.push_back(std::unique_ptr<Operator>(bwrnn_op));
  model.operators.push_back(std::unique_ptr<Operator>(fwrnn_op));
  model.operators.push_back(std::unique_ptr<Operator>(pack_op));

  bool modified;
  ASSERT_TRUE(GroupBidirectionalSequenceRnn().Run(&model, 0, &modified).ok());
  for (auto& op_it : model.operators) {
    Operator* op = op_it.get();
    // Since the optimization has kicked in we should not find any
    // UnidirectionalSequenceLstmOperator
    EXPECT_FALSE(op->type == OperatorType::kUnidirectionalSequenceRnn);
  }
}

}  // namespace

TEST(GroupBidirectionalSequenceLstm, SimpleConcatinationTest) {
  RunGroupBidirectionalSequencelstm(

      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4});
}

TEST(GroupBidirectionalSequenceRnn, SimpleConcatinationTest) {
  RunGroupBidirectionalSequenceRnn(

      // Input data
      {3, 1, 4, 1, -5, 9, -2, 6, 5, 3, 5, 8},

      // Input shape
      {3, 4},

      {3, 4});
}

}  // namespace toco
