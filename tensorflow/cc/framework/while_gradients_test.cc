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

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/framework/testutil.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class WhileGradientsTest : public ::testing::Test {
 protected:
  WhileGradientsTest() : scope_(Scope::NewRootScope()) {}

  void Init(int num_inputs, DataType dtype = DT_INT32) {
    for (int i = 0; i < num_inputs; ++i) {
      inputs_.push_back(ops::Placeholder(scope_, dtype));
    }
  }

  void CreateLoop(const ops::CondGraphBuilderFn& cond,
                  const ops::BodyGraphBuilderFn& body,
                  const std::vector<Output>* inputs = nullptr) {
    if (inputs == nullptr) inputs = &inputs_;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope_, *inputs, cond, body, "test_loop",
                                     &outputs_));
  }

  void CreateBackprop() {
    TF_ASSERT_OK(
        AddSymbolicGradients(scope_, outputs_, inputs_, &grad_outputs_));
    ASSERT_EQ(grad_outputs_.size(), inputs_.size());
  }

  template <typename T>
  void Run(const std::vector<Input::Initializer>& input_values,
           const std::vector<T>& expected_grad_values) {
    Run<T>(ClientSession(scope_), input_values, expected_grad_values);
  }

  template <typename T>
  void Run(const ClientSession& session,
           const std::vector<Input::Initializer>& input_values,
           const std::vector<T>& expected_grad_values,
           const RunOptions& run_options = RunOptions(),
           RunMetadata* run_metadata = nullptr) {
    DCHECK_EQ(input_values.size(), inputs_.size());
    ClientSession::FeedType feeds;
    for (int i = 0; i < inputs_.size(); ++i) {
      feeds.emplace(inputs_[i], input_values[i]);
    }

    std::vector<Operation> run_outputs;
    std::vector<Tensor> out_tensors;
    TF_ASSERT_OK(session.Run(run_options, feeds, grad_outputs_, run_outputs,
                             &out_tensors, run_metadata));
    ASSERT_EQ(out_tensors.size(), grad_outputs_.size());

    DCHECK_EQ(expected_grad_values.size(), out_tensors.size());
    for (int i = 0; i < out_tensors.size(); ++i) {
      test::ExpectTensorEqual<T>(
          out_tensors[i], test::AsTensor<T>({expected_grad_values[i]}, {}));
    }
  }

  Scope scope_;
  std::vector<Output> inputs_;
  std::vector<Output> outputs_;
  std::vector<Output> grad_outputs_;
};

TEST_F(WhileGradientsTest, Basic) {
  // Create loop: while (i < 10) i += 1
  Init(1);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
        // Use AddN, rather than Add, because the gradient function doesn't
        // depend on the input shapes, and thus we do not need to store
        // intermediate values in a stack.
        outputs->push_back(ops::AddN(s, {inputs[0], 1}));
        return s.status();
      });
  CreateBackprop();

  Run<int>({1}, {1});
  Run<int>({11}, {1});
}

TEST_F(WhileGradientsTest, MultipleLoopVars) {
  // Create loop: while (i < 10) i += j; j += 1; k = k
  Init(3);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
        outputs->push_back(ops::AddN(s, {inputs[0], inputs[1]}));
        outputs->push_back(ops::AddN(s, {inputs[1], 1}));
        outputs->push_back(inputs[2]);
        return s.status();
      });
  CreateBackprop();

  // The following execution traces illustrate why we expect dF/dj to be 5:
  //
  //  i  j  k
  // ---------
  //  0  1  2 <-- initial values
  //  1  2  2
  //  3  3  2
  //  6  4  2
  // 10  5  2 <-- while output values
  // outputs sum = 17
  //
  //  i  j  k
  // ---------
  //  0  2  2 <-- initial values (add 1 to j)
  //  2  3  2
  //  5  4  2
  //  9  5  2
  // 14  6  2 <-- while output values
  // outputs sum = 22
  //
  // Calculate the "slope" between j=1 and j=2:
  // 22 - 17 = 5 => dF/dj = 5
  Run<int>({0, 1, 2}, {1, 5, 1});

  Run<int>({1, 1, 0}, {1, 5, 1});
  Run<int>({0, 0, 0}, {1, 6, 1});
}

TEST_F(WhileGradientsTest, Chaining) {
  Init(2, DT_DOUBLE);

  // Multiply each input by 2 before passing to while loop to make sure chaining
  // works properly
  std::vector<Output> loop_inputs = {ops::Multiply(scope_, inputs_[0], 2.0),
                                     ops::Multiply(scope_, inputs_[1], 2.0)};

  // Create loop: while (i > 0 && j > 0) i -= 1
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::LogicalAnd(s, ops::Greater(s, inputs[0], 0.0),
                                  ops::Greater(s, inputs[1], 0.0));
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
        outputs->push_back(ops::AddN(s, {inputs[0], -1.0}));
        outputs->push_back(inputs[1]);
        return s.status();
      },
      &loop_inputs);

  // Take negative of first output to make sure chaining works properly
  outputs_[0] = ops::Neg(scope_, outputs_[0]);

  CreateBackprop();

  Run<double>({1.0, 1.0}, {-2.0, 2.0});
  Run<double>({0.0, 0.0}, {-2.0, 2.0});
}

TEST_F(WhileGradientsTest, MultipleDevices) {
  // Make sure loop is created on cpu0
  scope_ = scope_.WithDevice("/cpu:0");

  // Create loop: while (i < 10) i += j
  Init(2);
  CreateLoop(
      [](const Scope& s, const std::vector<Output>& inputs, Output* output) {
        *output = ops::Less(s, inputs[0], 10);
        return s.status();
      },
      [](const Scope& s, const std::vector<Output>& inputs,
         std::vector<Output>* outputs) {
        // Place body on cpu1
        Scope cpu1_scope = s.WithDevice("/cpu:1");
        outputs->push_back(ops::AddN(cpu1_scope, {inputs[0], inputs[1]}));
        outputs->push_back(inputs[1]);
        return cpu1_scope.status();
      });

  // Build gradient graph on cpu1
  Scope cpu1_scope = scope_.WithDevice("/cpu:1");
  TF_ASSERT_OK(
      AddSymbolicGradients(cpu1_scope, outputs_, inputs_, &grad_outputs_));
  ASSERT_EQ(grad_outputs_.size(), inputs_.size());

  // Run with two CPU devices and output partition graphs
  SessionOptions session_options;
  (*session_options.config.mutable_device_count())["CPU"] = 2;
  RunOptions run_options;
  run_options.set_output_partition_graphs(true);
  RunMetadata run_metadata;
  Run<int>(ClientSession(scope_, session_options), {0, 1}, {1, 11}, run_options,
           &run_metadata);

  // Check that at least one node ran on each device
  ASSERT_EQ(run_metadata.partition_graphs().size(), 2);
  for (const GraphDef& partition_graph : run_metadata.partition_graphs()) {
    EXPECT_GE(partition_graph.node().size(), 1);
  }
}

}  // namespace
}  // namespace tensorflow
