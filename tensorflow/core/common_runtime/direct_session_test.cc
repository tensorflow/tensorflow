/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

Session* CreateSession() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  return NewSession(options);
}

class DirectSessionMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    test::graph::ToGraphDef(&graph, &def_);
  }

  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork) {
  Initialize({3, 2, -1, 0});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(DirectSessionMinusAXTest, TestFeed) {
  Initialize({1, 2, 3, 4});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  //
  // Note that the input being fed is on the second device.
  Tensor t(DT_FLOAT, TensorShape({2, 1}));
  t.matrix<float>()(0, 0) = 5;
  t.matrix<float>()(1, 0) = 6;
  std::vector<std::pair<string, Tensor>> inputs = {{x_, t}};
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<Tensor> outputs;

  // Run the graph
  Status s = session->Run(inputs, output_names, {}, &outputs);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  auto mat = outputs[0].matrix<float>();

  // Expect outputs to be; 1*5 + 2*6, 3*5 + 4*6
  EXPECT_FLOAT_EQ(17.0, mat(0, 0));
  EXPECT_FLOAT_EQ(39.0, mat(1, 0));
}

TEST_F(DirectSessionMinusAXTest, TestConcurrency) {
  Initialize({1, 2, 3, 4});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  // Run the graph 1000 times in 4 different threads concurrently.
  std::vector<string> output_names = {y_ + ":0"};
  auto fn = [&session, output_names]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<Tensor> outputs;
      // Run the graph
      Status s = session->Run(inputs, output_names, {}, &outputs);
      TF_ASSERT_OK(s);
      ASSERT_EQ(1, outputs.size());
      auto mat = outputs[0].matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
    }
  };

  for (int i = 0; i < 4; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the functions to finish.
  delete tp;
}

TEST_F(DirectSessionMinusAXTest, TestPerSessionThreads) {
  Initialize({1, 2, 3, 4});

  SessionOptions options;
  options.config.set_use_per_session_threads(true);
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));

  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  // Run the graph 1000 times in 4 different threads concurrently.
  std::vector<string> output_names = {y_ + ":0"};
  auto fn = [&session, output_names]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<Tensor> outputs;
      // Run the graph
      Status s = session->Run(inputs, output_names, {}, &outputs);
      TF_ASSERT_OK(s);
      ASSERT_EQ(1, outputs.size());
      auto mat = outputs[0].matrix<float>();
      EXPECT_FLOAT_EQ(3.0, mat(0, 0));
    }
  };

  for (int i = 0; i < 4; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the functions to finish.
  delete tp;
}

TEST_F(DirectSessionMinusAXTest, TwoCreateCallsFails) {
  Initialize({1, 2, 3, 4});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Second is not.
  ASSERT_FALSE(session->Create(def_).ok());
}

TEST_F(DirectSessionMinusAXTest, ForgetToCreate) {
  Initialize({1, 2, 3, 4});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;
  ASSERT_FALSE(session->Run(inputs, {y_ + ":0"}, {y_neg_}, &outputs).ok());
}

TEST_F(DirectSessionMinusAXTest, InvalidDevice) {
  GraphDef def;
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  a_tensor.flat<float>().setRandom();
  Node* a = test::graph::Constant(&graph, a_tensor);
  a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  x_tensor.flat<float>().setRandom();
  Node* x = test::graph::Constant(&graph, x_tensor);
  x->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
  // Skip placing y.
  Node* y = test::graph::Matmul(&graph, a, x, false, false);
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:2");

  test::graph::ToGraphDef(&graph, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_names = {y->name() + ":0"};
  std::vector<Tensor> outputs;

  // Should return an error.
  ASSERT_FALSE(session->Run(inputs, output_names, {}, &outputs).ok());

  // Fix placement and run again
  def.Clear();
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
  test::graph::ToGraphDef(&graph, &def);
  session.reset(CreateSession());
  TF_ASSERT_OK(session->Create(def));
  TF_ASSERT_OK(session->Run(inputs, output_names, {}, &outputs));
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetworkWithOpts) {
  Initialize({3, 2, -1, 0});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunMetadata
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 0);

  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Checks RunMetadata is well-formed
  ASSERT_TRUE(run_metadata.has_step_stats());
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 2);
}

TEST(DirectSessionTest, KeepsStateAcrossRunsOfSession) {
  GraphDef def;
  Graph g(OpRegistry::Global());
  Node* var = test::graph::Var(&g, DT_FLOAT, TensorShape({10}));
  var->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  Tensor twenty(DT_FLOAT, TensorShape({10}));
  for (int i = 0; i < 10; ++i) {
    twenty.flat<float>()(i) = 20.0;
  }

  Node* twenty_node = test::graph::Constant(&g, twenty);
  twenty_node->set_assigned_device_name(
      "/job:localhost/replica:0/task:0/cpu:0");

  Node* init = test::graph::Assign(&g, var, twenty_node);
  init->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

  test::graph::ToGraphDef(&g, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;

  // Initialize the variable
  Status s = session->Run(inputs, {init->name()}, {}, &outputs);
  TF_ASSERT_OK(s);

  // Get the variable's data
  s = session->Run(inputs, {var->name() + ":0"}, {}, &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_EQ(20.0, outputs[0].flat<float>()(0));
}

TEST(DirectSessionTest, MultipleFeedTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  test::graph::ToGraphDef(&g, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  // Fetch without feeding.
  Status s = session->Run(
      {}, {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(1.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(2.0, outputs[1].flat<float>()(0));

  s = session->Run(
      {}, {second_identity->name() + ":0", first_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(1.0, outputs[1].flat<float>()(0));

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Feed [first_const, second_const]
  s = session->Run(
      {{first_const->name(), value_11}, {second_const->name(), value_22}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [second_const, first_const]
  s = session->Run(
      {{second_const->name(), value_22}, {first_const->name(), value_11}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));
}

REGISTER_OP("Darth")
    .Input("x: float")
    .Output("y: float")
    .Doc(R"doc(
Darth promises one return value.

x: float
y: float
)doc");

// The DarthOp kernel violates its promise to return one-value.
class DarthOp : public OpKernel {
 public:
  explicit DarthOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {}
};
REGISTER_KERNEL_BUILDER(Name("Darth").Device(DEVICE_CPU), DarthOp);

TEST(DirectSessionTest, DarthKernel) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_FLOAT, TensorShape({}));
  vx.scalar<float>()() = 1.0;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "Darth", x);
  GraphDef def;
  test::graph::ToGraphDef(&g, &def);
  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  auto s = sess->Run({}, {y->name() + ":0"}, {}, &outputs);
  EXPECT_TRUE(errors::IsInternal(s));
  delete sess;
}

TEST(DirectSessionTest, PartialRunTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  Node* third = test::graph::Add(&g, first_identity, second_identity);
  Node* third_identity = test::graph::Identity(&g, third);

  test::graph::ToGraphDef(&g, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup(
      {first_const->name(), second_const->name()},
      {first_identity->name() + ":0", second_identity->name() + ":0",
       third_identity->name() + ":0"},
      {}, &handle);
  TF_ASSERT_OK(s);

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Feed first_const, fetch first_identity
  s = session->PRun(handle, {{first_const->name(), value_11}},
                    {first_identity->name() + ":0"}, &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));

  // Feed second_const, fetch second_identity and third_identity
  s = session->PRun(
      handle, {{second_const->name(), value_22}},
      {second_identity->name() + ":0", third_identity->name() + ":0"},
      &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(22.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(11.0 + 22.0, outputs[1].flat<float>()(0));
}

TEST(DirectSessionTest, PartialRunMissingFeed) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  Node* third = test::graph::Add(&g, first_identity, second_identity);
  Node* third_identity = test::graph::Identity(&g, third);

  test::graph::ToGraphDef(&g, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup({first_const->name(), second_const->name()},
                                {third_identity->name() + ":0"}, {}, &handle);
  TF_ASSERT_OK(s);

  // Feed first_const, fetch third_identity
  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  s = session->PRun(handle, {{first_const->name(), value_11}},
                    {third_identity->name() + ":0"}, &outputs);
  ASSERT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("can't be computed from the feeds"));
}

TEST(DirectSessionTest, PartialRunMultiOutputFeed) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor bool_value(DT_BOOL, TensorShape({}));
  bool_value.scalar<bool>()() = true;
  Node* bool_const = test::graph::Constant(&g, bool_value);
  Node* switch_node = test::graph::Switch(&g, bool_const, bool_const);
  Node* fourth_identity = test::graph::Identity(&g, switch_node, 1);

  test::graph::ToGraphDef(&g, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  string handle;
  Status s = session->PRunSetup({switch_node->name() + ":1"},
                                {fourth_identity->name() + ":0"}, {}, &handle);
  TF_ASSERT_OK(s);

  // Fetch fourth_identity without feeds.
  s = session->PRun(handle, {}, {fourth_identity->name() + ":0"}, &outputs);
  ASSERT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(StringPiece(s.error_message())
                  .contains("can't be computed from the feeds"));

  // Feed switch_node:1 and fetch fourth_identity.
  s = session->PRun(handle, {{switch_node->name() + ":1", bool_value}},
                    {fourth_identity->name() + ":0"}, &outputs);
  TF_ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_EQ(true, outputs[0].flat<bool>()(0));
}

TEST(DirectSessionTest, RunHandleTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor value0(DT_FLOAT, TensorShape({}));
  value0.scalar<float>()() = 1.0;
  Node* const0 = test::graph::Constant(&g, value0);
  Node* identity0 = test::graph::Identity(&g, const0);

  Tensor value1(DT_FLOAT, TensorShape({}));
  value1.scalar<float>()() = 2.0;
  Node* const1 = test::graph::Constant(&g, value1);
  Node* node3 = test::graph::Add(&g, identity0, const1);
  Node* node4 = test::graph::Unary(&g, "GetSessionHandle", node3);

  Tensor value2(DT_STRING, TensorShape({}));
  Node* const2 = test::graph::Constant(&g, value2);
  Node* node5 = test::graph::GetSessionTensor(&g, const2);
  Node* node6 = test::graph::Add(&g, node5, const1);

  Node* node7 = test::graph::Unary(&g, "DeleteSessionTensor", const2);

  test::graph::ToGraphDef(&g, &def);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // First run call: Create a handle.
  std::vector<Tensor> outputs;
  Status s = session->Run({}, {node4->name() + ":0"}, {}, &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs.size());

  // Second run call: Use a handle.
  std::vector<Tensor> outputs1;
  s = session->Run({{const2->name(), outputs[0]}}, {node6->name() + ":0"}, {},
                   &outputs1);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs1.size());
  ASSERT_EQ(5.0, outputs1[0].flat<float>()(0));

  // Third run call: Delete a handle.
  std::vector<Tensor> outputs2;
  s = session->Run({{const2->name(), outputs[0]}}, {}, {node7->name()},
                   &outputs2);
  ASSERT_TRUE(s.ok());
}

TEST(DirectSessionTest, CreateGraphFailsWhenAssigningAFedVar) {
  Graph graph(OpRegistry::Global());

  Node* a = test::graph::Var(&graph, DT_FLOAT, {});
  Node* b = test::graph::Constant(&graph, {});

  Tensor zero(DT_FLOAT, {});
  test::FillValues<float>(&zero, {0});

  // a = b
  Node* assign = test::graph::Assign(&graph, a, b);

  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  // The graph is invalid since a constant cannot be assigned to a constant.
  // The return Status of session->Run should flag this as an invalid argument.
  std::vector<Tensor> outputs;
  Status s = session->Run({{a->name(), zero}}, {assign->name()}, {}, &outputs);
  ASSERT_TRUE(errors::IsInvalidArgument(s));
}

TEST(DirectSessionTest, TimeoutSession) {
  GraphDef graph;
  // Creates a graph with one FIFOQueue and one dequeue op.
  protobuf::TextFormat::ParseFromString(R"proto(
    node {
      name: 'fifo_queue'
      op: 'FIFOQueue'
      device: '/device:CPU:0'
      attr {
        key: 'capacity'
        value {
          i: 10
        }
      }
      attr {
        key: 'component_types'
        value {
          list {
            type: DT_FLOAT
          }
        }
      }
      attr {
        key: 'container'
        value {
          s: ''
        }
      }
      attr {
        key: 'shapes'
        value {
          list {
          }
        }
      }
      attr {
        key: 'shared_name'
        value {
          s: ''
        }
      }
    }
    node {
      name: 'fifo_queue_Dequeue'
      op: 'QueueDequeue'
      input: 'fifo_queue'
      device: '/device:CPU:0'
      attr {
        key: 'component_types'
        value {
          list {
            type: DT_FLOAT
          }
        }
      }
      attr {
        key: 'timeout_ms'
        value {
          i: -1
        }
      }
    }
    versions {
      producer: 9
    }
  )proto",
                                        &graph);

  // Creates a session with operation_timeout_in_ms set to 100 milliseconds.
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  options.config.set_operation_timeout_in_ms(100);
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(graph));

  // Verifies that the error code is DEADLINE_EXCEEDED.
  Status s = session->Run({}, {}, {"fifo_queue_Dequeue"}, nullptr);
  ASSERT_EQ(error::DEADLINE_EXCEEDED, s.code());
  session->Close();

  // Creates a session with no operation_timeout_in_ms.
  session.reset(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(graph));
  RunOptions run_options;
  run_options.set_timeout_in_ms(20);
  // Verifies that the error code is DEADLINE_EXCEEDED.
  Status s2 = session->Run(run_options, {}, {}, {"fifo_queue_Dequeue"}, nullptr,
                           nullptr);
  ASSERT_EQ(error::DEADLINE_EXCEEDED, s2.code());
  session->Close();
}

}  // namespace
}  // namespace tensorflow
