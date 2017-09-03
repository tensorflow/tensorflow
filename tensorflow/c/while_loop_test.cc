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

#include "tensorflow/c/c_api.h"

#include "tensorflow/c/c_test_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

using tensorflow::GraphDef;

namespace {

class CApiWhileLoopTest : public ::testing::Test {
 protected:
  CApiWhileLoopTest() : s_(TF_NewStatus()), graph_(TF_NewGraph()) {}

  ~CApiWhileLoopTest() override {
    TF_DeleteGraph(graph_);
    TF_DeleteStatus(s_);
  }

  void Init(int ninputs) {
    DCHECK(inputs_.empty());
    DCHECK_GT(ninputs, 0);

    for (int i = 0; i < ninputs; ++i) {
      TF_Operation* placeholder = Placeholder(
          graph_, s_, ::tensorflow::strings::StrCat("p", i).c_str());
      DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
      inputs_.push_back({placeholder, 0});
    }

    original_graph_description_ = GraphDebugString();

    params_.reset(new TF_WhileParams(
        TF_NewWhile(graph_, &inputs_[0], inputs_.size(), s_)));
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    ASSERT_EQ(original_graph_description_, GraphDebugString())
        << "TF_NewWhile() altered graph";

    params_->name = "test_loop";

    // Initialize outputs_ so we can easily detect errors/bugs
    outputs_.resize(ninputs, {nullptr, -1});
  }

  void ExpectOK() {
    TF_FinishWhile(params_.get(), s_, &outputs_[0]);
    EXPECT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void ExpectError(TF_Code expected_code, const string& expected_msg) {
    TF_FinishWhile(params_.get(), s_, &outputs_[0]);
    EXPECT_EQ(expected_code, TF_GetCode(s_));
    EXPECT_EQ(expected_msg, TF_Message(s_));
    // TODO(skyewm): this assert is currently broken. Fix or remove guarantee.
    // ASSERT_EQ(original_graph_description_, GraphDebugString()) <<
    //     "TF_FinishWhile() altered graph on error";
  }

  void Run(std::initializer_list<int> input_values) {
    DCHECK_EQ(inputs_.size(), input_values.size());
    std::vector<std::pair<TF_Operation*, TF_Tensor*>> inputs(inputs_.size());
    int i = 0;
    for (int v : input_values) {
      inputs[i] = {inputs_[i].oper, Int32Tensor(v)};
      ++i;
    }
    csession_.reset(new CSession(graph_, s_));
    csession_->SetInputs(inputs);
    csession_->SetOutputs(outputs_);
    csession_->Run(s_);
    ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  }

  void ExpectOutputValue(int idx, int expected_value) {
    TF_Tensor* out = csession_->output_tensor(idx);
    ASSERT_TRUE(out != nullptr);
    EXPECT_EQ(TF_INT32, TF_TensorType(out));
    EXPECT_EQ(0, TF_NumDims(out));
    ASSERT_EQ(sizeof(int32_t), TF_TensorByteSize(out));
    int32_t* data = static_cast<int32_t*>(TF_TensorData(out));
    EXPECT_EQ(expected_value, *data);
  }

  // Create a valid conditional graph. Useful for testing unrelated errors.
  void CreateCondGraph() {
    TF_Operation* one = ScalarConst(1, params_->cond_graph, s_);
    TF_Operation* less_than =
        LessThan(params_->cond_inputs[0], {one, 0}, params_->cond_graph, s_);
    DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    params_->cond_output = {less_than, 0};
  }

  string GraphDebugString() const {
    TF_Buffer* buf = TF_NewBuffer();
    TF_GraphToGraphDef(graph_, buf, s_);
    DCHECK_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
    GraphDef def;
    bool success = def.ParseFromArray(buf->data, buf->length);
    DCHECK(success);
    TF_DeleteBuffer(buf);
    return def.DebugString();
  }

  TF_Status* s_;
  TF_Graph* graph_;
  std::vector<TF_Output> inputs_;   // The inputs to the while loop
  std::vector<TF_Output> outputs_;  // The final outputs of the while loop
  std::unique_ptr<TF_WhileParams> params_;
  std::unique_ptr<CSession> csession_;

 private:
  // Used to verify that errors don't change graph_
  string original_graph_description_;
};

TEST_F(CApiWhileLoopTest, BasicLoop) {
  Init(2);

  // Validate TF_WhileParams returned by TF_NewWhile()
  EXPECT_TRUE(params_->body_graph != nullptr);
  EXPECT_TRUE(params_->cond_graph != nullptr);

  EXPECT_EQ(params_->ninputs, 2);

  ASSERT_TRUE(params_->cond_inputs != nullptr);
  ASSERT_TRUE(params_->cond_inputs[0].oper != nullptr);
  EXPECT_TRUE(params_->cond_inputs[1].oper != nullptr);

  ASSERT_TRUE(params_->body_inputs != nullptr);
  EXPECT_TRUE(params_->body_inputs[0].oper != nullptr);
  EXPECT_TRUE(params_->body_inputs[1].oper != nullptr);

  ASSERT_TRUE(params_->body_outputs != nullptr);

  // Create loop: while (input1 < input2) input1 += input2 + 1
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], params_->cond_inputs[1],
               params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  TF_Operation* add1 = Add(params_->body_inputs[0], params_->body_inputs[1],
                           params_->body_graph, s_, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* one = ScalarConst(1, params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* add2 = Add(add1, one, params_->body_graph, s_, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {add2, 0};
  params_->body_outputs[1] = params_->body_inputs[1];

  // Finalize while loop
  ExpectOK();

  // Validate while loop outputs returned by TF_FinishWhile()
  EXPECT_TRUE(outputs_[0].oper != nullptr);
  EXPECT_GE(outputs_[0].index, 0);
  EXPECT_TRUE(outputs_[1].oper != nullptr);
  EXPECT_GE(outputs_[1].index, 0);

  // Check that cond and body inputs are not present
  for (int i = 0; i < params_->ninputs; ++i) {
    string cond_name =
        ::tensorflow::strings::StrCat(params_->name, "/cond/cond_input", i);
    string body_name =
        ::tensorflow::strings::StrCat(params_->name, "/body/body_input", i);
    EXPECT_TRUE(TF_GraphOperationByName(graph_, cond_name.c_str()) == nullptr);
    EXPECT_TRUE(TF_GraphOperationByName(graph_, body_name.c_str()) == nullptr);
  }

  // Run the graph
  Run({-9, 2});
  ExpectOutputValue(0, 3);
  ExpectOutputValue(1, 2);
}

TEST_F(CApiWhileLoopTest, NestedLoop) {
  Init(2);
  // Create nested loop:
  //  while (input1 < 6) {
  //    inner_input1 = input1
  //    while (inner_input1 < 3) {
  //      input2 += 1
  //      inner_input1 += 2
  //    }
  //    input1 += input2
  //  }
  //
  // Expected execution with initial values input1 = input2 = 0:
  //
  // outer inner               inner_
  // step# step# input1 input2 input1
  // ------------------------------------
  //   0     0     0      0      0
  //   0     1     0      1      2
  //   0     2     0      2      4
  //   0     -     2      2      -
  //   1     0     2      2      2
  //   1     1     2      3      4
  //   1     -     5      3      -
  //   2     0     5      3      5
  //   2     -     8      3      -

  // Create outer cond graph
  TF_Operation* six = ScalarConst(6, params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* less_than =
      LessThan(params_->cond_inputs[0], {six, 0}, params_->cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->cond_output = {less_than, 0};

  // Create outer body graph
  // Init inner graph
  TF_Output inner_inputs[] = {params_->body_inputs[0], params_->body_inputs[1]};
  TF_WhileParams inner_params =
      TF_NewWhile(params_->body_graph, inner_inputs, 2, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.name = "inner_loop";

  // Create inner cond graph
  TF_Operation* three = ScalarConst(3, inner_params.cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* inner_less_than = LessThan(
      inner_params.cond_inputs[0], {three, 0}, inner_params.cond_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.cond_output = {inner_less_than, 0};

  // Create inner body graph
  TF_Operation* one = ScalarConst(1, inner_params.body_graph, s_, "one");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  TF_Operation* two = ScalarConst(2, inner_params.body_graph, s_, "two");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_Operation* input2_add =
      Add(inner_params.body_inputs[1].oper, one, inner_params.body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.body_outputs[1] = {input2_add, 0};

  TF_Operation* inner_input1_add = Add(inner_params.body_inputs[0].oper, two,
                                       inner_params.body_graph, s_, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  inner_params.body_outputs[0] = {inner_input1_add, 0};

  // Finalize inner graph
  TF_Output inner_outputs[2] = {{nullptr, -1}};
  TF_FinishWhile(&inner_params, s_, inner_outputs);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);

  TF_Operation* input1_add =
      Add(params_->body_inputs[0], inner_outputs[1], params_->body_graph, s_);
  ASSERT_EQ(TF_OK, TF_GetCode(s_)) << TF_Message(s_);
  params_->body_outputs[0] = {input1_add, 0};

  params_->body_outputs[1] = inner_outputs[1];

  // Finalize outer graph
  ExpectOK();

  // Check for a few expected nodes
  const char* node_name = "test_loop/cond/scalar";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/add";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/inner_loop/body/one";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);
  node_name = "test_loop/body/inner_loop/cond/less_than";
  EXPECT_TRUE(TF_GraphOperationByName(graph_, node_name) != nullptr);

  // Run the graph
  Run({0, 0});
  ExpectOutputValue(0, 8);
  ExpectOutputValue(1, 3);
}

TEST_F(CApiWhileLoopTest, BadCondOutput) {
  Init(1);
  params_->body_outputs[0] = params_->body_inputs[0];
  ExpectError(TF_INVALID_ARGUMENT,
              "TF_WhileParams `cond_output` field isn't set");
}

TEST_F(CApiWhileLoopTest, BadBodyOutput) {
  Init(1);
  CreateCondGraph();
  ExpectError(TF_INVALID_ARGUMENT,
              "TF_WhileParams `body_outputs[0]` field isn't set");
}

TEST_F(CApiWhileLoopTest, NullName) {
  Init(1);
  CreateCondGraph();
  params_->body_outputs[0] = params_->body_inputs[0];
  params_->name = nullptr;
  ExpectError(TF_INVALID_ARGUMENT, "TF_WhileParams `name` field is null");
}

TEST_F(CApiWhileLoopTest, WrongGraph) {
  Init(1);
  CreateCondGraph();
  // Set body output to output from outer graph
  params_->body_outputs[0] = inputs_[0];
  // TODO(skyewm): improve error message
  ExpectError(TF_INVALID_ARGUMENT,
              "Requested return node 'p0' not found in graph def");
}

TEST_F(CApiWhileLoopTest, BadTypes) {
  Init(1);
  CreateCondGraph();
  // Op that has a float input + output
  TF_OperationDescription* desc = TF_NewOperation(
      params_->body_graph, "FakeQuantWithMinMaxArgs", "float_op");
  TF_AddInput(desc, params_->body_inputs[0]);
  TF_FinishOperation(desc, s_);
  ASSERT_EQ(TF_INVALID_ARGUMENT, TF_GetCode(s_));
  string msg(TF_Message(s_));
  EXPECT_NE(msg.find("Input 'inputs' passed int32 expected float while "
                     "building NodeDef 'float_op'"),
            msg.npos);
  TF_AbortWhile(params_.get());
}

}  // namespace
