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

#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/while_context.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class WhileLoopTest : public ::testing::Test {
 protected:
  WhileLoopTest() : scope_(Scope::NewRootScope()) {}

  void Init(int num_inputs, DataType dtype = DT_INT32) {
    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(ops::Placeholder(scope_, dtype));
    }
  }

  void CreateLoop(const ops::CondGraphBuilderFn& cond,
                  const ops::BodyGraphBuilderFn& body,
                  error::Code error_code = error::OK,
                  const string& error_msg = "") {
    Status s =
        ops::BuildWhileLoop(scope_, inputs_, cond, body, kFrameName, &outputs_);
    EXPECT_EQ(s.code(), error_code);
    EXPECT_EQ(s.error_message(), error_msg);
  }

  template <typename T>
  void Run(const std::vector<Input::Initializer>& input_values,
           const std::vector<T>& expected_output_values) {
    ClientSession session(scope_);

    DCHECK_EQ(input_values.size(), inputs_.size());
    ClientSession::FeedType feeds;
    for (int i = 0; i < inputs_.size(); ++i) {
      feeds.emplace(inputs_[i], input_values[i]);
    }

    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(feeds, outputs_, &out_tensors));
    ASSERT_EQ(out_tensors.size(), outputs_.size());

    DCHECK_EQ(expected_output_values.size(), out_tensors.size());
    for (int i = 0; i < out_tensors.size(); ++i) {
      test::ExpectTensorEqual<T>(
          out_tensors[i], test::AsTensor<T>({expected_output_values[i]}, {}));
    }
  }

  Scope scope_;
  std::vector<Output> inputs_;
  std::vector<Output> outputs_;

  static const char* const kFrameName;
};

const char* const WhileLoopTest::kFrameName = "test_loop";

Status LessThanTenCond(const Scope& s, const std::vector<Output>& inputs,
                       Output* output) {
  *output = ops::Less(s, inputs[0], 10);
  return s.status();
}

Status AddOneBody(const Scope& s, const std::vector<Output>& inputs,
                  std::vector<Output>* outputs) {
  outputs->push_back(ops::Add(s, inputs[0], 1));
  return s.status();
}

TEST_F(WhileLoopTest, Basic) {
  // Create loop: while (i < 10) i += 1
  Init(1);
  CreateLoop(LessThanTenCond, AddOneBody);

  // Verify some output invariants
  WhileContext* while_ctx;
  for (int i = 0; i < outputs_.size(); ++i) {
    Node* node = outputs_[i].node();
    ASSERT_TRUE(node->IsExit()) << "Output node " << i << ":\n"
                                << node->DebugString();
    ASSERT_TRUE(node->while_ctx() != nullptr) << i;
    if (i == 0) {
      while_ctx = node->while_ctx();
      EXPECT_EQ(while_ctx->frame_name(), kFrameName);
    } else {
      EXPECT_EQ(node->while_ctx(), while_ctx) << i;
    }
  }

  // Run the loop and test we get the expected results
  Run<int>({1}, {10});
  Run<int>({11}, {11});
}

TEST_F(WhileLoopTest, WrongCondOutputType) {
  Init(1);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Placeholder(s, DT_FLOAT);
        return s.status();
      },
      AddOneBody, error::INVALID_ARGUMENT,
      "BuildWhileLoop: 'cond' argument must return a boolean output, got "
      "float");
}

// TODO(skyewm): test bad cond output shape

TEST_F(WhileLoopTest, NullCondOutputNode) {
  Init(1);
  // TODO(skyewm): improve error message
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = {nullptr, 0};
        return s.status();
      },
      AddOneBody, error::INVALID_ARGUMENT, "Node is null");
}

TEST_F(WhileLoopTest, InvalidCondOutputIndex) {
  Init(1);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        auto less = ops::Less(s, inputs[0], 10);
        *output = {less.node(), 100};
        return s.status();
      },
      AddOneBody, error::OUT_OF_RANGE,
      "Node 'cond/Less' (type: 'Less', num of outputs: 1) does not have output "
      "100");
}

TEST_F(WhileLoopTest, UnsetCondOutput) {
  Init(1);
  CreateLoop([](const Scope& s, const std::vector<Output>& inputs,
                Output* output) { return s.status(); },
             AddOneBody, error::INVALID_ARGUMENT, "Node is null");
}

// TODO(skyewm): test bad body output type
// TODO(skyewm): test bad body output shape

TEST_F(WhileLoopTest, NullBodyOutputNode) {
  Init(1);
  // TODO(skyewm): improve error message
  CreateLoop(LessThanTenCond,
             [](const Scope& s, const std::vector<Output>& inputs,
                std::vector<Output>* outputs) {
               outputs->push_back({nullptr, 0});
               return s.status();
             },
             error::INVALID_ARGUMENT, "Node is null");
}

TEST_F(WhileLoopTest, InvalidBodyOutputIndex) {
  Init(1);
  CreateLoop(LessThanTenCond,
             [](const Scope& s, const std::vector<Output>& inputs,
                std::vector<Output>* outputs) {
               auto add = ops::Add(s, inputs[0], 1);
               outputs->emplace_back(add.node(), 100);
               return s.status();
             },
             error::OUT_OF_RANGE,
             "Node 'body/Add' (type: 'Add', num of outputs: 1) does not have "
             "output 100");
}

TEST_F(WhileLoopTest, UnsetBodyOutputs) {
  Init(1);
  CreateLoop(
      LessThanTenCond,
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) { return s.status(); },
      error::INVALID_ARGUMENT,
      "BuildWhileLoop: 'body' argument expected to return 1 output(s), got 0");
}

}  // namespace
}  // namespace tensorflow
