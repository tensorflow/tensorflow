/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <random>
#include <string>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/function_testlib.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/stacktrace.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/device_name_utils.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {
namespace {

CallableOptions MakeCallableOptions(gtl::ArraySlice<string> feeds,
                                    gtl::ArraySlice<string> fetches,
                                    gtl::ArraySlice<string> targets) {
  CallableOptions ret;
  for (const string& feed : feeds) {
    ret.add_feed(feed);
  }
  for (const string& fetch : fetches) {
    ret.add_fetch(fetch);
  }
  for (const string& target : targets) {
    ret.add_target(target);
  }
  return ret;
}

SessionOptions DefaultSessionOptions() {
  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  return options;
}

std::unique_ptr<Session> CreateSession() {
  return std::unique_ptr<Session>(NewSession(DefaultSessionOptions()));
}

class DirectSessionMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");
    a_ = a->name();

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

    Node* z = test::graph::Unary(&graph, "Identity", y_neg);
    z_ = z->name();
    z->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");

    graph.ToGraphDef(&def_);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  string z_;
  GraphDef def_;
};

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
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

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork_Callable) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Run the test twice to ensure that the Make/Run/Release cycle is hermetic.
  for (int i = 0; i < 2; ++i) {
    // Request two targets: one fetch output and one non-fetched output.
    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(
        MakeCallableOptions({}, {y_ + ":0"}, {y_neg_}), &handle));

    for (int i = 0; i < 2; ++i) {
      std::vector<Tensor> outputs;
      TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));

      ASSERT_EQ(1, outputs.size());
      // The first output should be initialized and have the correct
      // output.
      auto mat = outputs[0].matrix<float>();
      ASSERT_TRUE(outputs[0].IsInitialized());
      EXPECT_FLOAT_EQ(5.0, mat(0, 0));
    }

    Status s = session->RunCallable(handle, {}, nullptr, nullptr);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(s.error_message(),
                                  "`fetch_tensors` must be provided"));

    TF_ASSERT_OK(session->ReleaseCallable(handle));

    std::vector<Tensor> outputs;
    s = session->RunCallable(handle, {}, &outputs, nullptr);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(
        s.error_message(),
        "Attempted to run callable after handle was released"));

    s = session->RunCallable(handle + 1, {}, &outputs, nullptr);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(
        absl::StrContains(s.error_message(), "No such callable handle"));
  }
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork_OptimizeForStaticGraph) {
  Initialize({3, 2, -1, 0});
  SessionOptions options(DefaultSessionOptions());
  options.config.mutable_experimental()->set_optimize_for_static_graph(true);
  auto session = absl::WrapUnique(NewSession(options));

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

  s = session->Extend({});
  EXPECT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "optimize_for_static_graph"));
}

TEST_F(DirectSessionMinusAXTest,
       RunSimpleNetwork_DisableOutputPartitionGraphs) {
  Initialize({3, 2, -1, 0});
  SessionOptions options(DefaultSessionOptions());
  options.config.mutable_experimental()->set_disable_output_partition_graphs(
      true);
  auto session = absl::WrapUnique(NewSession(options));

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

  // The Run() call should fail when `output_partition_graphs` is set to true.
  RunOptions run_options;
  run_options.set_output_partition_graphs(true);
  RunMetadata run_metadata;
  s = session->Run(run_options, inputs, output_names, target_nodes, &outputs,
                   &run_metadata);

  EXPECT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "disable_output_partition_graphs"));
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork_FinalizeWithCallables) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Request two targets: one fetch output and one non-fetched output.
  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions({}, {y_ + ":0"}, {y_neg_}), &handle));

  // Finalize the session.
  TF_ASSERT_OK(session->Finalize());

  // The callable is usable after finalization.
  for (int i = 0; i < 2; ++i) {
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));

    ASSERT_EQ(1, outputs.size());
    // The first output should be initialized and have the correct
    // output.
    auto mat = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(5.0, mat(0, 0));
  }

  TF_ASSERT_OK(session->ReleaseCallable(handle));

  // Making a new callable fails because the session has been finalized.
  Status s =
      session->MakeCallable(MakeCallableOptions({}, {y_ + ":0"}, {}), &handle);
  EXPECT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "Session has been finalized."));
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetwork_FinalizeWithRun) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {y_ + ":0"}, {y_neg_}, &outputs));

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Finalize the session.
  TF_ASSERT_OK(session->Finalize());

  // Running the exact same subgraph succeeds after finalization.
  TF_ASSERT_OK(session->Run({}, {y_ + ":0"}, {y_neg_}, &outputs));
  ASSERT_EQ(1, outputs.size());
  mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Running a different subgraph fails because the session has been finalized.
  Status s = session->Run({}, {y_ + ":0"}, {}, &outputs);
  EXPECT_TRUE(errors::IsFailedPrecondition(s));
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "Session has been finalized."));
}

TEST_F(DirectSessionMinusAXTest, TestTensorConnection) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  {
    // Directly wire the output of node a to the output of node y, making the
    // callable graph into "Neg(a);".
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(y_neg_ + ":0");

    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
    ASSERT_EQ(1, outputs.size());
    auto mat = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(-3.0, mat(0, 0));
    EXPECT_FLOAT_EQ(-2.0, mat(0, 1));
    EXPECT_FLOAT_EQ(1.0, mat(1, 0));
    EXPECT_FLOAT_EQ(0.0, mat(1, 1));
    TF_ASSERT_OK(session->ReleaseCallable(handle));
  }

  {
    // Directly wire the output of node a to the output of node y, making the
    // callable graph into "Neg(a);"; also fetch the result of a.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(a_ + ":0");
    callable_options.add_fetch(y_neg_ + ":0");

    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
    ASSERT_EQ(2, outputs.size());
    auto mat_a = outputs[0].matrix<float>();
    ASSERT_TRUE(outputs[0].IsInitialized());
    EXPECT_FLOAT_EQ(3.0, mat_a(0, 0));
    EXPECT_FLOAT_EQ(2.0, mat_a(0, 1));
    EXPECT_FLOAT_EQ(-1.0, mat_a(1, 0));
    EXPECT_FLOAT_EQ(0.0, mat_a(1, 1));

    auto mat_y_neg = outputs[1].matrix<float>();
    ASSERT_TRUE(outputs[1].IsInitialized());
    EXPECT_FLOAT_EQ(-3.0, mat_y_neg(0, 0));
    EXPECT_FLOAT_EQ(-2.0, mat_y_neg(0, 1));
    EXPECT_FLOAT_EQ(1.0, mat_y_neg(1, 0));
    EXPECT_FLOAT_EQ(0.0, mat_y_neg(1, 1));
    TF_ASSERT_OK(session->ReleaseCallable(handle));
  }

  {
    // Wire the output of "Neg(Matmul(a, x))" to the output of "a",
    // creating an invalid cycle.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(y_ + ":0");
    c->set_to_tensor(a_ + ":0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(s.error_message(), "would create a cycle"));
  }

  {
    // Attempt to wire a non-existent node to a node that does exist.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor("unknown_node:0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(s.error_message(), "unknown node"));
  }

  {
    // Attempt to wire a non-existent output from a node that does
    // exist to another node.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":17");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(s.error_message(), "unknown edge"));
  }

  {
    // Attempt to wire a tensor to a node that doesn't exist.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor("unknown_node:0");
    callable_options.add_fetch(y_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsNotFound(s));
    EXPECT_TRUE(
        absl::StrContains(s.error_message(), "unable to find feed output"));
  }

  {
    // Attempt to wire two tensors to the same tensor.
    CallableOptions callable_options;
    TensorConnection* c1 = callable_options.add_tensor_connection();
    c1->set_from_tensor(a_ + ":0");
    c1->set_to_tensor(y_neg_ + ":0");
    TensorConnection* c2 = callable_options.add_tensor_connection();
    c2->set_from_tensor(x_ + ":0");
    c2->set_to_tensor(y_neg_ + ":0");
    callable_options.add_fetch(z_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(s.error_message(), "fed more than once"));
  }

  {
    // Attempt to wire a tensor to a tensor that is also being fed.
    CallableOptions callable_options;
    TensorConnection* c = callable_options.add_tensor_connection();
    c->set_from_tensor(a_ + ":0");
    c->set_to_tensor(y_ + ":0");
    callable_options.add_feed(y_ + ":0");
    callable_options.add_fetch(y_neg_ + ":0");

    Session::CallableHandle handle;
    Status s = session->MakeCallable(callable_options, &handle);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
    EXPECT_TRUE(absl::StrContains(s.error_message(), "fed more than once"));
  }
}

TEST_F(DirectSessionMinusAXTest, TestFeed) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
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

TEST_F(DirectSessionMinusAXTest, TestFeed_Callable) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);

  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  //
  // Note that the input being fed is on the second device.
  CallableOptions callable_options;
  callable_options.add_feed(x_);
  callable_options.add_fetch(y_ + ":0");
  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(MakeCallableOptions({x_}, {y_ + ":0"}, {}),
                                     &handle));
  Tensor t(DT_FLOAT, TensorShape({2, 1}));
  t.matrix<float>()(0, 0) = 5;
  t.matrix<float>()(1, 0) = 6;
  std::vector<Tensor> inputs = {t};
  std::vector<Tensor> outputs;

  // Run the callable
  TF_ASSERT_OK(session->RunCallable(handle, inputs, &outputs, nullptr));

  ASSERT_EQ(1, outputs.size());
  auto mat = outputs[0].matrix<float>();

  // Expect outputs to be; 1*5 + 2*6, 3*5 + 4*6
  EXPECT_FLOAT_EQ(17.0, mat(0, 0));
  EXPECT_FLOAT_EQ(39.0, mat(1, 0));
}

TEST_F(DirectSessionMinusAXTest, TestConcurrency) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
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

TEST_F(DirectSessionMinusAXTest, TestConcurrency_Callable) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Fill in the input and ask for the output
  thread::ThreadPool* tp = new thread::ThreadPool(Env::Default(), "test", 4);

  Session::CallableHandle handle;
  TF_ASSERT_OK(
      session->MakeCallable(MakeCallableOptions({}, {y_ + ":0"}, {}), &handle));

  // Run the callable 1000 times in 4 different threads concurrently.
  auto fn = [&session, handle]() {
    for (int i = 0; i < 1000; ++i) {
      std::vector<Tensor> outputs;
      // Run the graph
      TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
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
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Second is not.
  ASSERT_FALSE(session->Create(def_).ok());
}

TEST_F(DirectSessionMinusAXTest, ForgetToCreate) {
  Initialize({1, 2, 3, 4});
  auto session = CreateSession();
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

  graph.ToGraphDef(&def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  // Should return an error.
  ASSERT_FALSE(session->Create(def).ok());

  // Fix placement and run again
  def.Clear();
  y->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:1");
  graph.ToGraphDef(&def);
  session.reset(NewSession(options));
  TF_ASSERT_OK(session->Create(def));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run({}, {y->name() + ":0"}, {}, &outputs));
}

TEST_F(DirectSessionMinusAXTest, RunSimpleNetworkWithOpts) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunMetadata
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::SOFTWARE_TRACE);
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

TEST_F(DirectSessionMinusAXTest, RunSimpleNetworkWithOpts_Callable) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Request two targets: one fetch output and one non-fetched output.
  Session::CallableHandle handle;
  CallableOptions callable_options =
      MakeCallableOptions({}, {y_ + ":0"}, {y_neg_});
  callable_options.mutable_run_options()->set_trace_level(
      RunOptions::SOFTWARE_TRACE);
  TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));

  RunMetadata run_metadata;
  EXPECT_EQ(run_metadata.step_stats().dev_stats_size(), 0);

  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, &run_metadata));

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

TEST_F(DirectSessionMinusAXTest, UseRunHandlerPool) {
  Initialize({3, 2, -1, 0});
  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  // Prepares RunOptions and RunMetadata
  RunOptions run_options;
  run_options.mutable_experimental()->set_use_run_handler_pool(true);

  Status s = session->Run(run_options, inputs, output_names, target_nodes,
                          &outputs, nullptr);
  TF_ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
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

  g.ToGraphDef(&def);

  auto session = CreateSession();
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

  g.ToGraphDef(&def);

  auto session = CreateSession();
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

  // Feed [first_const, first_const]
  s = session->Run(
      {{first_const->name(), value_11}, {first_const->name(), value_22}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  EXPECT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(absl::StrContains(s.error_message(), "fed more than once"));
}

TEST(DirectSessionTest, MultipleFeedTest_Callable) {
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

  g.ToGraphDef(&def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  Session::CallableHandle handle;
  std::vector<Tensor> outputs;

  // Fetch without feeding.
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {}, {first_identity->name() + ":0", second_identity->name() + ":0"},
          {}),
      &handle));
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(1.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(2.0, outputs[1].flat<float>()(0));

  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {}, {second_identity->name() + ":0", first_identity->name() + ":0"},
          {}),
      &handle));
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(2.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(1.0, outputs[1].flat<float>()(0));

  Tensor value_11(DT_FLOAT, TensorShape({}));
  value_11.scalar<float>()() = 11.0;
  Tensor value_22(DT_FLOAT, TensorShape({}));
  value_22.scalar<float>()() = 22.0;

  // Feed [first_const, second_const]
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {first_const->name(), second_const->name()},
          {first_identity->name() + ":0", second_identity->name() + ":0"}, {}),
      &handle));
  TF_ASSERT_OK(
      session->RunCallable(handle, {value_11, value_22}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [second_const, first_const]
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions(
          {second_const->name(), first_const->name()},
          {first_identity->name() + ":0", second_identity->name() + ":0"}, {}),
      &handle));
  TF_ASSERT_OK(
      session->RunCallable(handle, {value_22, value_11}, &outputs, nullptr));
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [first_const, first_const]
  Status s = session->MakeCallable(
      MakeCallableOptions(
          {first_const->name(), first_const->name()},
          {first_identity->name() + ":0", second_identity->name() + ":0"}, {}),
      &handle);
  EXPECT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(absl::StrContains(s.error_message(), "fed more than once"));
}

TEST(DirectSessionTest, TestTensorConnectionUseTwice) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  test::FillValues<float>(&a_tensor, {1.0, 2.0, 3.0, 4.0});
  Node* a = test::graph::Constant(&graph, a_tensor);

  Tensor dummy_tensor(DT_FLOAT, TensorShape({1}));
  test::FillValues<float>(&dummy_tensor, {-1.0});

  Node* left = test::graph::Constant(&graph, dummy_tensor);
  Node* right = test::graph::Constant(&graph, dummy_tensor);

  // y = A * x
  Node* y = test::graph::Add(&graph, left, right);

  GraphDef def;
  graph.ToGraphDef(&def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  CallableOptions callable_options;
  // Directly wire the output of node a to the outputs of nodes left
  // and right, making the callable graph into "a + a;".
  TensorConnection* c_left = callable_options.add_tensor_connection();
  c_left->set_from_tensor(a->name() + ":0");
  c_left->set_to_tensor(left->name() + ":0");
  TensorConnection* c_right = callable_options.add_tensor_connection();
  c_right->set_from_tensor(a->name() + ":0");
  c_right->set_to_tensor(right->name() + ":0");

  callable_options.add_fetch(y->name() + ":0");

  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(callable_options, &handle));
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->RunCallable(handle, {}, &outputs, nullptr));
  ASSERT_EQ(1, outputs.size());
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(2.0, mat(0, 0));
  EXPECT_FLOAT_EQ(4.0, mat(0, 1));
  EXPECT_FLOAT_EQ(6.0, mat(1, 0));
  EXPECT_FLOAT_EQ(8.0, mat(1, 1));
  TF_ASSERT_OK(session->ReleaseCallable(handle));
}

TEST(DirectSessionTest, FetchMultipleTimes) {
  Graph g(OpRegistry::Global());
  Tensor seven_tensor(DT_INT32, TensorShape());
  seven_tensor.flat<int32>()(0) = 7;
  Node* seven_node = test::graph::Constant(&g, seven_tensor);

  GraphDef def;
  g.ToGraphDef(&def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  const std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;

  auto seven = seven_node->name();
  Status s = session->Run(inputs, {seven, seven}, {}, &outputs);
  TF_ASSERT_OK(s);

  EXPECT_EQ(2, outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    const Tensor& t = outputs[i];
    ASSERT_TRUE(t.IsInitialized()) << i;
    EXPECT_EQ(7, t.flat<int32>()(0)) << i;
  }
}

TEST(DirectSessionTest, MultipleFeedTestSomeSyncRun) {
  GraphDef def;
  Graph g(OpRegistry::Global());
  RunOptions run_options;
  run_options.set_inter_op_thread_pool(-1);

  Tensor first_value(DT_FLOAT, TensorShape({}));
  first_value.scalar<float>()() = 1.0;
  Node* first_const = test::graph::Constant(&g, first_value);
  Node* first_identity = test::graph::Identity(&g, first_const);

  Tensor second_value(DT_FLOAT, TensorShape({}));
  second_value.scalar<float>()() = 2.0;
  Node* second_const = test::graph::Constant(&g, second_value);
  Node* second_identity = test::graph::Identity(&g, second_const);

  g.ToGraphDef(&def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  // Fetch without feeding.
  Status s = session->Run(
      run_options, {},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs, nullptr);
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

  // Feed [first_const, first_const]
  s = session->Run(
      run_options,
      {{first_const->name(), value_11}, {first_const->name(), value_22}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs, nullptr);
  EXPECT_TRUE(errors::IsInvalidArgument(s));
  EXPECT_TRUE(absl::StrContains(s.error_message(), "fed more than once"));
}

REGISTER_OP("SessionMetadataReader")
    .Input("x: int64")
    .Output("y: string")
    .SetIsStateful()
    .Doc(R"doc(SessionMetadataReader returns the session metadata.

x: int64
y: string
)doc");

class SessionMetadataReaderOp : public OpKernel {
 public:
  explicit SessionMetadataReaderOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("y", TensorShape({}), &out_tensor));
    if (ctx->session_metadata() != nullptr) {
      out_tensor->scalar<tstring>()() = ctx->session_metadata()->DebugString();
    } else {
      out_tensor->scalar<tstring>()() = "";
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("SessionMetadataReader").Device(DEVICE_CPU),
                        SessionMetadataReaderOp);
REGISTER_KERNEL_BUILDER(Name("SessionMetadataReader").Device(DEVICE_GPU),
                        SessionMetadataReaderOp);

FunctionDef SessionMetadataReaderOpFn() {
  return FunctionDefHelper::Define(
      // Name
      "SessionMetadataReaderFn",
      // Args
      {"x: int64"},
      // Return values
      {"y: string"},
      // Attr def
      {},
      // Nodes
      {{{"y"}, "SessionMetadataReader", {"x"}, {}}});
}

TEST(DirectSessionTest, SessionMetadataAbsent) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_INT64, TensorShape({}));
  vx.scalar<int64>()() = 17;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "SessionMetadataReader", x);
  GraphDef def;
  g.ToGraphDef(&def);
  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  run_opts.set_inter_op_thread_pool(-1);
  auto s = sess->Run(run_opts, {}, {y->name() + ":0"}, {}, &outputs, nullptr);

  EXPECT_EQ("", outputs[0].scalar<tstring>()());
}

TEST(DirectSessionTest, SessionMetadataAbsentViaFunction) {
  FunctionDefLibrary library_graph_def;
  *library_graph_def.add_function() = SessionMetadataReaderOpFn();
  FunctionLibraryDefinition flib(OpRegistry::Global(), library_graph_def);
  Graph g(&flib);
  Tensor vx(DT_INT64, TensorShape({}));
  vx.scalar<int64>()() = 17;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "SessionMetadataReaderFn", x);
  GraphDef def;
  g.ToGraphDef(&def);
  *def.mutable_library() = library_graph_def;
  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  run_opts.set_inter_op_thread_pool(-1);
  auto s = sess->Run(run_opts, {}, {y->name() + ":0"}, {}, &outputs, nullptr);

  EXPECT_EQ("", outputs[0].scalar<tstring>()());
}

TEST(DirectSessionTest, SessionMetadataPresent) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_INT64, TensorShape({}));
  vx.scalar<int64>()() = 17;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "SessionMetadataReader", x);
  GraphDef def;
  g.ToGraphDef(&def);
  auto session_options = DefaultSessionOptions();
  auto* session_metadata =
      session_options.config.mutable_experimental()->mutable_session_metadata();
  session_metadata->set_name("name");
  session_metadata->set_version(1);
  auto sess = std::unique_ptr<Session>(NewSession(session_options));
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  run_opts.set_inter_op_thread_pool(-1);
  auto s = sess->Run(run_opts, {}, {y->name() + ":0"}, {}, &outputs, nullptr);

  SessionMetadata read_metadata;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      outputs[0].scalar<tstring>()(), &read_metadata));
  EXPECT_EQ("name", read_metadata.name());
  EXPECT_EQ(1, read_metadata.version());
}

TEST(DirectSessionTest, SessionMetadataPresentViaFunction) {
  FunctionDefLibrary library_graph_def;
  *library_graph_def.add_function() = SessionMetadataReaderOpFn();
  FunctionLibraryDefinition flib(OpRegistry::Global(), library_graph_def);
  Graph g(&flib);
  Tensor vx(DT_INT64, TensorShape({}));
  vx.scalar<int64>()() = 17;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "SessionMetadataReaderFn", x);
  GraphDef def;
  g.ToGraphDef(&def);
  *def.mutable_library() = library_graph_def;
  auto session_options = DefaultSessionOptions();
  auto* session_metadata =
      session_options.config.mutable_experimental()->mutable_session_metadata();
  session_metadata->set_name("name");
  session_metadata->set_version(1);
  auto sess = std::unique_ptr<Session>(NewSession(session_options));
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  run_opts.set_inter_op_thread_pool(-1);
  auto s = sess->Run(run_opts, {}, {y->name() + ":0"}, {}, &outputs, nullptr);

  SessionMetadata read_metadata;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      outputs[0].scalar<tstring>()(), &read_metadata));
  EXPECT_EQ("name", read_metadata.name());
  EXPECT_EQ(1, read_metadata.version());
}

TEST(DirectSessionTest, SessionMetadataKey) {
  auto session_options0 = DefaultSessionOptions();
  auto* session_metadata0 = session_options0.config.mutable_experimental()
                                ->mutable_session_metadata();
  session_metadata0->set_name("name");
  Session* sess0_ptr;
  ASSERT_TRUE(NewSession(session_options0, &sess0_ptr).ok());
  auto sess0 = absl::WrapUnique(sess0_ptr);

  // Trying to use the same metadata (name, version) will cause an error.
  Session* dup_ptr;
  EXPECT_TRUE(
      errors::IsInvalidArgument(NewSession(session_options0, &dup_ptr)));

  // A new (name, version) is fine.
  auto session_options1 = DefaultSessionOptions();
  auto* session_metadata1 = session_options1.config.mutable_experimental()
                                ->mutable_session_metadata();
  session_metadata1->set_name("name");
  session_metadata1->set_version(1);
  Session* sess1_ptr;
  EXPECT_TRUE(NewSession(session_options1, &sess1_ptr).ok());
  auto sess1 = absl::WrapUnique(sess1_ptr);

  // If the previous session, using the same (name, version) is gone, then it's
  // fine.
  sess0 = nullptr;
  EXPECT_TRUE(NewSession(session_options0, &dup_ptr).ok());
  auto dup = absl::WrapUnique(dup_ptr);

  // Sessions without metadata options are always fine.
  auto sess_without_metadata0 = CreateSession();
  EXPECT_NE(sess_without_metadata0, nullptr);
  auto sess_without_metadata1 = CreateSession();
  EXPECT_NE(sess_without_metadata1, nullptr);
}

TEST(DirectSessionTest, SessionMetadataInvalid) {
  const auto valid_session_options = DefaultSessionOptions();
  Session* sess_ptr;
  ASSERT_TRUE(NewSession(valid_session_options, &sess_ptr).ok());
  auto sess = absl::WrapUnique(sess_ptr);

  auto invalid_session_options = valid_session_options;
  auto* invalid_metadata =
      invalid_session_options.config.mutable_experimental()
          ->mutable_session_metadata();
  invalid_metadata->set_name("name");
  // Version should be >= 0.
  invalid_metadata->set_version(-1);
  Session* error_sess_ptr;
  EXPECT_TRUE(errors::IsInvalidArgument(
      NewSession(invalid_session_options, &error_sess_ptr)));
}

REGISTER_OP("ThreadID").Input("x: int64").Output("y: int64").Doc(R"doc(
ThreadID returns the thread ID that called compute.

x: int64
y: int64
)doc");

// The ThreadID kernel returns the thread ID that executed Compute.
class ThreadIDOp : public OpKernel {
 public:
  explicit ThreadIDOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output("y", TensorShape({}), &out_tensor));
    std::hash<std::thread::id> hasher;
    out_tensor->scalar<int64>()() =
        static_cast<int64>(hasher(std::this_thread::get_id()));
  }
};
REGISTER_KERNEL_BUILDER(Name("ThreadID").Device(DEVICE_CPU), ThreadIDOp);

TEST(DirectSessionTest, SessionSyncRun) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_INT64, TensorShape({}));
  vx.scalar<int64>()() = 17;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "ThreadID", x);
  GraphDef def;
  g.ToGraphDef(&def);
  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  run_opts.set_inter_op_thread_pool(-1);
  auto s = sess->Run(run_opts, {}, {y->name() + ":0"}, {}, &outputs, nullptr);

  std::hash<std::thread::id> hasher;
  EXPECT_EQ(static_cast<int64>(hasher(std::this_thread::get_id())),
            static_cast<int64>(outputs[0].scalar<int64>()()));
}

REGISTER_OP("ExpensiveNoop").SetIsStateful();

class ExpensiveNoopOp : public OpKernel {
 public:
  using OpKernel::OpKernel;
  bool IsExpensive() override { return true; }
  void Compute(OpKernelContext* ctx) override {
    const string& stack_trace = tensorflow::CurrentStackTrace();
    const string process_method = "ExecutorState::Process()";
    size_t pos = 0;
    int frame_count = 0;
    while ((pos = stack_trace.find("ExecutorState::Process()", pos)) !=
           string::npos) {
      ++frame_count;
      ++pos;
    }
    OP_REQUIRES(ctx, frame_count <= 1,
                errors::Internal(
                    "Recursive call to ExecutorState::Process() detected."));
  }
};

REGISTER_KERNEL_BUILDER(Name("ExpensiveNoop").Device(DEVICE_CPU),
                        ExpensiveNoopOp);

TEST(DirectSessionTest, SessionSyncRun_DeepGraph) {
  Graph g(OpRegistry::Global());

  std::vector<Node*> nodes;
  nodes.reserve(1024);

  auto make_expensive_noop = [&g](gtl::ArraySlice<Node*> control_deps) {
    Node* ret;
    auto builder = NodeBuilder(g.NewName("N"), "ExpensiveNoop");
    for (Node* control_dep : control_deps) {
      builder = builder.ControlInput(control_dep);
    }
    TF_CHECK_OK(builder.Finalize(&g, &ret));
    return ret;
  };

  Node* base = make_expensive_noop({});

  Node* child_1 = make_expensive_noop({base});
  Node* child_2 = make_expensive_noop({base});

  GraphDef def;
  g.ToGraphDef(&def);

  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  run_opts.set_inter_op_thread_pool(-1);

  EXPECT_TRUE(sess->Run(run_opts, {}, {}, {child_1->name(), child_2->name()},
                        &outputs, nullptr)
                  .ok());
}

TEST(DirectSessionTest, SyncSession) {
  Graph g(OpRegistry::Global());
  Tensor vx(DT_INT64, TensorShape({}));
  vx.scalar<int64>()() = 17;
  Node* x = test::graph::Constant(&g, vx);
  Node* y = test::graph::Unary(&g, "ThreadID", x);
  GraphDef def;
  g.ToGraphDef(&def);
  SessionOptions options;
  options.config.set_inter_op_parallelism_threads(-1);
  std::unique_ptr<Session> sess(NewSession(options));
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  RunOptions run_opts;
  auto s = sess->Run(run_opts, {}, {y->name() + ":0"}, {}, &outputs, nullptr);

  std::hash<std::thread::id> hasher;
  EXPECT_EQ(static_cast<int64>(hasher(std::this_thread::get_id())),
            static_cast<int64>(outputs[0].scalar<int64>()()));
}

REGISTER_OP("Darth").Input("x: float").Output("y: float").Doc(R"doc(
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
  g.ToGraphDef(&def);
  auto sess = CreateSession();
  TF_ASSERT_OK(sess->Create(def));
  std::vector<Tensor> outputs;
  auto s = sess->Run({}, {y->name() + ":0"}, {}, &outputs);
  EXPECT_TRUE(errors::IsInternal(s));
}

// Have the Darth op in the graph placed on GPU, but don't run it.
TEST(DirectSessionTest, PlacePrunedGraph) {
  {
    Graph g(OpRegistry::Global());
    Tensor vx(DT_FLOAT, TensorShape({}));
    vx.scalar<float>()() = 1.0;
    Node* x = test::graph::Constant(&g, vx);
    Node* y = test::graph::Unary(&g, "Darth", x);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");
    GraphDef def;
    g.ToGraphDef(&def);

    // By default, we place the entire graph, so we should fail the
    // call to Create.
    SessionOptions options;
    std::unique_ptr<Session> sess(NewSession(options));
    auto s = sess->Create(def);
    EXPECT_TRUE(errors::IsInvalidArgument(s));
  }

  {
    Graph g(OpRegistry::Global());
    Tensor vx(DT_FLOAT, TensorShape({}));
    vx.scalar<float>()() = 1.0;
    Node* x = test::graph::Constant(&g, vx);
    Node* y = test::graph::Unary(&g, "Darth", x);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/device:GPU:0");
    GraphDef def;
    g.ToGraphDef(&def);

    SessionOptions options;
    // Set the option to place pruned graphs, we should expect this
    // to run.
    options.config.mutable_graph_options()->set_place_pruned_graph(true);
    std::unique_ptr<Session> sess(NewSession(options));
    TF_ASSERT_OK(sess->Create(def));
    std::vector<Tensor> outputs;
    auto s = sess->Run({}, {x->name() + ":0"}, {}, &outputs);
    TF_EXPECT_OK(s);
  }
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

  g.ToGraphDef(&def);

  auto session = CreateSession();
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

  g.ToGraphDef(&def);

  auto session = CreateSession();
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
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "can't be computed from the feeds"));
}

TEST(DirectSessionTest, PartialRunMultiOutputFeed) {
  GraphDef def;
  Graph g(OpRegistry::Global());

  Tensor bool_value(DT_BOOL, TensorShape({}));
  bool_value.scalar<bool>()() = true;
  Node* bool_const = test::graph::Constant(&g, bool_value);
  Node* switch_node = test::graph::Switch(&g, bool_const, bool_const);
  Node* fourth_identity = test::graph::Identity(&g, switch_node, 1);

  g.ToGraphDef(&def);

  auto session = CreateSession();
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
  EXPECT_TRUE(
      absl::StrContains(s.error_message(), "can't be computed from the feeds"));

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
  Node* node4 = test::graph::Unary(&g, "GetSessionHandleV2", node3);

  Tensor value2(DT_STRING, TensorShape({}));
  Node* const2 = test::graph::Constant(&g, value2);
  Node* node5 = test::graph::GetSessionTensor(&g, const2);
  Node* node6 = test::graph::Add(&g, node5, const1);

  Node* node7 = test::graph::Unary(&g, "DeleteSessionTensor", const2);

  g.ToGraphDef(&def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // First run call: Create a handle.
  std::vector<Tensor> outputs;
  Status s = session->Run({}, {node4->name() + ":0"}, {}, &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs.size());

  const ResourceHandle& resource_handle = outputs[0].scalar<ResourceHandle>()();
  Tensor string_handle(DT_STRING, {});
  string_handle.flat<tstring>().setConstant(resource_handle.name());

  // Second run call: Use a handle.
  std::vector<Tensor> outputs1;
  s = session->Run({{const2->name(), string_handle}}, {node6->name() + ":0"},
                   {}, &outputs1);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs1.size());
  ASSERT_EQ(5.0, outputs1[0].flat<float>()(0));

  // Third run call: Delete a handle.
  std::vector<Tensor> outputs2;
  s = session->Run({{const2->name(), string_handle}}, {}, {node7->name()},
                   &outputs2);
  ASSERT_TRUE(s.ok());
}

TEST(DirectSessionTest, RunHandleTest_Callable) {
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
  Node* node4 = test::graph::Unary(&g, "GetSessionHandleV2", node3);

  Tensor value2(DT_STRING, TensorShape({}));
  Node* const2 = test::graph::Constant(&g, value2);
  Node* node5 = test::graph::GetSessionTensor(&g, const2);
  Node* node6 = test::graph::Add(&g, node5, const1);

  Node* node7 = test::graph::Unary(&g, "DeleteSessionTensor", const2);

  g.ToGraphDef(&def);

  auto session = CreateSession();
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // First run call: Create a handle.
  std::vector<Tensor> outputs;
  Status s = session->Run({}, {node4->name() + ":0"}, {}, &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs.size());

  const ResourceHandle& resource_handle = outputs[0].scalar<ResourceHandle>()();
  Tensor string_handle(DT_STRING, {});
  string_handle.flat<tstring>().setConstant(resource_handle.name());

  // Second run call: Use a handle.
  std::vector<Tensor> outputs1;
  s = session->Run({{const2->name(), string_handle}}, {node6->name() + ":0"},
                   {}, &outputs1);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(1, outputs1.size());
  ASSERT_EQ(5.0, outputs1[0].flat<float>()(0));

  // Third run call: Delete a handle.
  std::vector<Tensor> outputs2;
  s = session->Run({{const2->name(), string_handle}}, {}, {node7->name()},
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

  auto session = CreateSession();
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
        value { i: 10 }
      }
      attr {
        key: 'component_types'
        value { list { type: DT_FLOAT } }
      }
      attr {
        key: 'container'
        value { s: '' }
      }
      attr {
        key: 'shapes'
        value { list {} }
      }
      attr {
        key: 'shared_name'
        value { s: '' }
      }
    }
    node {
      name: 'fifo_queue_Dequeue'
      op: 'QueueDequeue'
      input: 'fifo_queue'
      device: '/device:CPU:0'
      attr {
        key: 'component_types'
        value { list { type: DT_FLOAT } }
      }
      attr {
        key: 'timeout_ms'
        value { i: -1 }
      }
    }
    versions { producer: 9 }
  )proto", &graph);

  {
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
    TF_ASSERT_OK(session->Close());
  }

  {
    // Creates a session with no operation_timeout_in_ms.
    auto session = CreateSession();
    ASSERT_TRUE(session != nullptr);
    TF_ASSERT_OK(session->Create(graph));
    RunOptions run_options;
    run_options.set_timeout_in_ms(20);
    // Verifies that the error code is DEADLINE_EXCEEDED.
    Status s2 = session->Run(run_options, {}, {}, {"fifo_queue_Dequeue"},
                             nullptr, nullptr);
    ASSERT_EQ(error::DEADLINE_EXCEEDED, s2.code());
    TF_ASSERT_OK(session->Close());
  }
}

// Accesses the cancellation manager for the step after the step has been
// cancelled.
class CancellationMgrPollingOp : public OpKernel {
 public:
  explicit CancellationMgrPollingOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    CancellationManager* cm = ctx->cancellation_manager();
    while (!cm->IsCancelled()) {
      ctx->env()->SleepForMicroseconds(1000);
    }
    notification.Notify();
  }
  static Notification notification;
};
Notification CancellationMgrPollingOp::notification;

REGISTER_KERNEL_BUILDER(Name("CancellationMgrPollingOp").Device(DEVICE_CPU),
                        CancellationMgrPollingOp);
REGISTER_OP("CancellationMgrPollingOp").Doc("");

TEST(DirectSessionTest, TestTimeoutCleanShutdown) {
  GraphDef graph;
  // Creates a graph with one FIFOQueue and one dequeue op.
  protobuf::TextFormat::ParseFromString(R"proto(
    node {
      name: 'cm_polling'
      op: 'CancellationMgrPollingOp'
      device: '/device:CPU:0'
    }
    versions { producer: 9 }
  )proto", &graph);

  // Creates a session with operation_timeout_in_ms set to 100 milliseconds.
  SessionOptions options;
  options.config.set_operation_timeout_in_ms(100);
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(graph));

  // Verifies that the error code is DEADLINE_EXCEEDED.
  Status s = session->Run({}, {}, {"cm_polling"}, nullptr);
  ASSERT_EQ(error::DEADLINE_EXCEEDED, s.code());

  // Verify that the op ran to completion.
  ASSERT_TRUE(CancellationMgrPollingOp::notification.HasBeenNotified());

  TF_ASSERT_OK(session->Close());
}

static void TestSessionInterOpThreadsImpl(bool use_function_lib,
                                          bool use_global_pools) {
  using test::function::blocking_op_state;
  using test::function::BlockingOpState;

  FunctionDefLibrary library_graph_def;
  if (use_function_lib) {
    *library_graph_def.add_function() = test::function::BlockingOpFn();
  }

  FunctionLibraryDefinition flib(OpRegistry::Global(), library_graph_def);
  Graph g(&flib);
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* x = test::graph::Constant(&g, t);
  Node* y;
  if (use_function_lib) {
    y = test::graph::Unary(&g, "BlockingOpFn", x);
  } else {
    y = test::graph::Unary(&g, "BlockingOp", x);
  }
  GraphDef def;
  g.ToGraphDef(&def);
  *def.mutable_library() = library_graph_def;

  // Create session with two inter-op thread pools.
  SessionOptions options;
  // Turn off optimizations so that the blocking op doesn't get invoked during
  // graph setup.
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  options.config.mutable_graph_options()
      ->mutable_rewrite_options()
      ->set_constant_folding(RewriterConfig::OFF);
  (*options.config.mutable_device_count())["CPU"] = 2;
  (*options.config.mutable_device_count())["GPU"] = 0;

  auto* p = options.config.add_session_inter_op_thread_pool();
  if (use_global_pools) p->set_global_name("large pool");
  p = options.config.add_session_inter_op_thread_pool();
  if (use_global_pools) p->set_global_name("small pool");
  p->set_num_threads(1);
  const int kSyncPool = -1;
  const int kLargePool = 0;
  const int kSmallPool = 1;

  std::vector<std::unique_ptr<Session>> sessions;
  if (!use_global_pools) {
    sessions.emplace_back(NewSession(options));
    TF_ASSERT_OK(sessions.back()->Create(def));
  }
  mutex sessions_mu;

  std::atomic<int32> num_done(0);
  // Runs session to compute <node>:0 using inter_op thread pool <pool>.
  auto add_session_run_call =
      [use_global_pools, &def, &options, &sessions, &sessions_mu, &num_done](
          thread::ThreadPool* tp, Node* node, int inter_op_pool) {
        auto fn = [use_global_pools, &def, &options, &sessions, &sessions_mu,
                   inter_op_pool, node, &num_done]() {
          RunOptions run_options;
          run_options.set_inter_op_thread_pool(inter_op_pool);
          std::vector<Tensor> outputs;

          Session* session;
          if (use_global_pools) {
            std::unique_ptr<Session> s(NewSession(options));
            TF_ASSERT_OK(s->Create(def));
            session = s.get();

            mutex_lock l(sessions_mu);
            sessions.emplace_back(std::move(s));
          } else {
            session = sessions[0].get();
          }

          Status s = session->Run(run_options, {} /* inputs */,
                                  {node->name() + ":0"} /* output_names */, {},
                                  &outputs, nullptr /* run_metadata */);
          TF_CHECK_OK(s);
          ASSERT_EQ(1, outputs.size());
          auto flat = outputs[0].flat<float>();
          EXPECT_FLOAT_EQ(1.2, flat(0));
          num_done.fetch_add(1);
        };
        if (tp != nullptr) {
          tp->Schedule(fn);
        } else {
          fn();
        }
      };

  // For blocking states:
  // - Starts at 0, BlockingOp::Compute will move to 1.
  // - This main thread will wait for 1, then move to 2 when other ops are done.
  //   Moving to 2 unblocks the blocking op, which then moves to state 3.

  // Run the graph once on the non-limited pool.
  thread::ThreadPool* tp1 = new thread::ThreadPool(Env::Default(), "tp1", 1);
  blocking_op_state = new BlockingOpState();
  add_session_run_call(tp1, y, kLargePool);
  blocking_op_state->AwaitState(1);
  blocking_op_state->MoveToState(1, 2);
  blocking_op_state->AwaitState(3);
  blocking_op_state->MoveToState(3, 0);
  delete tp1;
  num_done = 0;

  tp1 = new thread::ThreadPool(Env::Default(), "tp1", 5);

  // Launch a session run call. It will not finish until the blocking op is
  // unblocked, because it is using all threads in the small pool.
  add_session_run_call(tp1, y, kSmallPool);

  blocking_op_state->AwaitState(1);  // Wait for the blocking op to Compute.

  // These will block on <BlockingOpState>.
  const int kBlockedThreads = 3;
  for (int i = 0; i < kBlockedThreads; ++i) {
    add_session_run_call(tp1, x, kSmallPool);
  }

  // Launch session calls using the other inter-op pool. These will finish
  // as they are in inter_op pool #2.
  thread::ThreadPool* tp2 = new thread::ThreadPool(Env::Default(), "tp2", 3);
  const int kUnblockedThreads = 4;
  for (int i = 0; i < kUnblockedThreads; ++i) {
    add_session_run_call(tp2, x, kLargePool);
  }
  delete tp2;
  EXPECT_EQ(kUnblockedThreads, num_done.load());

  // Launch a session call using this thread. This will finish as it runs
  // synchronously in this thread.
  add_session_run_call(nullptr, x, kSyncPool);

  // Unblock the blocked op and wait for the blocked functions to finish.
  blocking_op_state->MoveToState(1, 2);
  delete tp1;

  EXPECT_EQ(kUnblockedThreads + kBlockedThreads + 1 + 1, num_done.load());
  delete blocking_op_state;
  blocking_op_state = nullptr;
}

TEST(DirectSessionTest, TestSessionInterOpThreads) {
  TestSessionInterOpThreadsImpl(false /* use_function_lib */,
                                false /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpThreadsWithFunctions) {
  TestSessionInterOpThreadsImpl(true /* use_function_lib */,
                                false /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpGlobalPools) {
  TestSessionInterOpThreadsImpl(false /* use_function_lib */,
                                true /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpGlobalPoolsWithFunctions) {
  TestSessionInterOpThreadsImpl(true /* use_function_lib */,
                                true /*use_global_pools */);
}

TEST(DirectSessionTest, TestSessionInterOpThreadsInvalidOptions) {
  Graph g(OpRegistry::Global());
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* x = test::graph::Constant(&g, t);
  GraphDef def;
  g.ToGraphDef(&def);

  SessionOptions options;
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  (*options.config.mutable_device_count())["CPU"] = 2;

  options.config.add_session_inter_op_thread_pool();

  // Wrong pool number on Run call.
  {
    std::unique_ptr<Session> session(NewSession(options));
    TF_ASSERT_OK(session->Create(def));
    for (int pool_num = -2; pool_num <= 1; pool_num += 3) {
      RunOptions run_options;
      run_options.set_inter_op_thread_pool(pool_num);
      std::vector<Tensor> outputs;
      Status s = session->Run(run_options, {} /* inputs */,
                              {x->name() + ":0"} /* output_names */, {},
                              &outputs, nullptr /* run_metadata */);
      EXPECT_EQ(
          strings::StrCat("Invalid argument: Invalid inter_op_thread_pool: ",
                          pool_num),
          s.ToString());
    }
  }

  // Global name changes thread count.
  std::vector<std::unique_ptr<Session>> sessions;
  auto* pool_config = options.config.mutable_session_inter_op_thread_pool(0);
  pool_config->set_num_threads(0);
  pool_config->set_global_name("foo");
  sessions.emplace_back(NewSession(options));
  TF_ASSERT_OK(sessions.back()->Create(def));
  sessions.emplace_back(NewSession(options));  // repeat creation, okay.
  TF_ASSERT_OK(sessions.back()->Create(def));
  for (int pass = 0; pass < 2; ++pass) {
    for (int i = 1; i < 128; ++i) {
      pool_config->set_num_threads(i);
      sessions.emplace_back(NewSession(options));
      auto status = sessions.back()->Create(def);
      ASSERT_FALSE(status.ok()) << status;
    }

    // Clear existing sessions before second pass; error still happens.
    sessions.clear();
  }
}

TEST(DirectSessionTest, TestDirectSessionRunClose) {
  // Construct a graph with a variable and a single assign.
  Graph g(OpRegistry::Global());
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* var_val = test::graph::Constant(&g, t);
  Node* var = test::graph::Var(&g, DT_FLOAT, {});
  Node* var_assign = test::graph::Assign(&g, var, var_val);
  GraphDef def;
  g.ToGraphDef(&def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // Assign a value to the var.
  TF_ASSERT_OK(session->Run({} /* inputs */, {},
                            {var_assign->name()} /* target_nodes */, nullptr));

  // Run a read on the variable to ensure that it works.
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(
      {} /* inputs */, {var->name() + ":0"} /* output_names */, {}, &outputs));
  EXPECT_EQ(t.scalar<float>()(), outputs[0].scalar<float>()());
  outputs.clear();

  // Make a callable handle before closing the session.
  Session::CallableHandle handle;
  TF_ASSERT_OK(session->MakeCallable(
      MakeCallableOptions({}, {}, {var_assign->name()}), &handle));

  // Close the session.
  TF_ASSERT_OK(session->Close());

  // Run the read on the variable to get an error.
  Status s = session->Run({} /* inputs */, {},
                          {var_assign->name()} /* target_nodes */, nullptr);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());

  // Run the read as a callable to verify that we get the same error.
  s = session->RunCallable(handle, {}, {}, nullptr);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());
}

TEST(DirectSessionTest, TestDirectSessionPRunClose) {
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

  g.ToGraphDef(&def);

  auto session = CreateSession();
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

  // Close the session.
  TF_ASSERT_OK(session->Close());

  // Feed first_const, fetch first_identity
  s = session->PRun(handle, {{first_const->name(), value_11}},
                    {first_identity->name() + ":0"}, &outputs);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());
}

TEST(DirectSessionTest, TestDirectSessionReset) {
  // Construct a graph with a variable and a single assign.
  Graph g(OpRegistry::Global());
  Tensor t(DT_FLOAT, TensorShape({}));
  t.scalar<float>()() = {1.2f};
  Node* var_val = test::graph::Constant(&g, t);
  Node* var = test::graph::Var(&g, DT_FLOAT, {});
  Node* var_assign = test::graph::Assign(&g, var, var_val);
  GraphDef def;
  g.ToGraphDef(&def);

  SessionOptions options;
  (*options.config.mutable_device_count())["CPU"] = 2;
  std::unique_ptr<Session> session(NewSession(options));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def));

  // Assign a value to the var.
  TF_ASSERT_OK(session->Run({} /* inputs */, {},
                            {var_assign->name()} /* target_nodes */, nullptr));

  // Run a read on the variable to ensure that it works.
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(session->Run(
      {} /* inputs */, {var->name() + ":0"} /* output_names */, {}, &outputs));
  EXPECT_EQ(t.scalar<float>()(), outputs[0].scalar<float>()());
  outputs.clear();

  // Reset the containers.
  TF_EXPECT_OK(Reset(options, {}));

  // Run the read on the variable to get an error.
  // TODO(suharshs): This test only works because we close the Session in Reset.
  // If we change the behavior of Reset to not close the Session, this test will
  // fail, since the Variable buffer is cached by var.
  Status s = session->Run({} /* inputs */, {},
                          {var_assign->name()} /* target_nodes */, nullptr);
  EXPECT_EQ("Cancelled: Session has been closed.", s.ToString());
}

TEST(DirectSessionTest, LocalDeviceManager) {
  SessionOptions options;
  std::unique_ptr<Session> session(NewSession(options));

  const DeviceMgr* mgr = nullptr;
  TF_ASSERT_OK(session->LocalDeviceManager(&mgr));
  ASSERT_TRUE(mgr != nullptr);
  EXPECT_GT(mgr->ListDevices().size(), 0);
}

// y = tf.square(x)
GraphDef CreateGraphForYEqualsXSquared() {
  GraphDef graph_def;
  const char* text_proto = R"EOF(
node {
  name: "x"
  op: "Placeholder"
  attr { key: "dtype" value { type: DT_FLOAT } }
  attr { key: "shape" value { shape { unknown_rank: true } } }
}
node {
  name: "y"
  op: "Square"
  input: "x"
  attr { key: "T" value { type: DT_FLOAT } }
}
versions {
  producer: 26
}
  )EOF";

  QCHECK(protobuf::TextFormat::ParseFromString(text_proto, &graph_def));
  return graph_def;
}

// A graph that consumes and produces string tensors
// (which are not GPU-compatible, i.e., there are no
// GPU kernels for these operations).
bool IsCUDATensor(const Tensor& t) {
#ifdef GOOGLE_CUDA
  cudaPointerAttributes attributes;
  cudaError_t err =
      cudaPointerGetAttributes(&attributes, t.tensor_data().data());
  if (err == cudaErrorInvalidValue) return false;
  CHECK_EQ(cudaSuccess, err) << cudaGetErrorString(err);
  return (attributes.type == cudaMemoryTypeDevice);
#elif TENSORFLOW_USE_ROCM
  hipPointerAttribute_t attributes;
  hipError_t err = hipPointerGetAttributes(&attributes, t.tensor_data().data());
  if (err == hipErrorInvalidValue) return false;
  CHECK_EQ(hipSuccess, err) << hipGetErrorString(err);
  return (attributes.memoryType == hipMemoryTypeDevice);
#else
  return false;
#endif
}

string GPUDeviceName(Session* session) {
  std::vector<DeviceAttributes> devices;
  TF_CHECK_OK(session->ListDevices(&devices));
  for (const DeviceAttributes& d : devices) {
    if (d.device_type() == "GPU" || d.device_type() == "gpu") {
      return d.name();
    }
  }
  return "";
}

TEST(DirectSessionTest, FeedAndFetchTensorsInDeviceMemory) {
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  const string gpu_device_name = GPUDeviceName(session.get());
  if (gpu_device_name.empty()) {
    LOG(INFO) << "Skipping test since no GPU is available";
    return;
  }

  TF_ASSERT_OK(session->Create(CreateGraphForYEqualsXSquared()));

  CallableOptions opts;
  opts.add_feed("x:0");
  opts.add_fetch("y:0");

  Tensor gpu_tensor;

  {
    Session::CallableHandle feed_cpu_fetch_gpu;
    opts.mutable_fetch_devices()->insert({"y:0", gpu_device_name});
    opts.set_fetch_skip_sync(true);
    TF_ASSERT_OK(session->MakeCallable(opts, &feed_cpu_fetch_gpu));
    Tensor input(DT_FLOAT, {});
    input.scalar<float>()() = 2.0f;
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(
        session->RunCallable(feed_cpu_fetch_gpu, {input}, &outputs, nullptr));
    TF_ASSERT_OK(session->ReleaseCallable(feed_cpu_fetch_gpu));
    ASSERT_EQ(1, outputs.size());
    gpu_tensor = outputs[0];
    ASSERT_TRUE(IsCUDATensor(gpu_tensor));
  }

  {
    Session::CallableHandle feed_gpu_fetch_cpu;
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({"x:0", gpu_device_name});
    TF_ASSERT_OK(session->MakeCallable(opts, &feed_gpu_fetch_cpu));
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(feed_gpu_fetch_cpu, {gpu_tensor},
                                      &outputs, nullptr));
    TF_ASSERT_OK(session->ReleaseCallable(feed_gpu_fetch_cpu));
    ASSERT_EQ(1, outputs.size());
    // The output is in CPU/host memory, so it can be dereferenced.
    ASSERT_EQ(16.0, outputs[0].scalar<float>()());
  }
}

GraphDef CreateIdentityGraphDef(DataType dtype) {
  GraphDef def;

  AttrValue dtype_attr;
  dtype_attr.set_type(dtype);

  AttrValue shape_attr;
  shape_attr.mutable_shape()->set_unknown_rank(true);

  auto* placeholder = def.add_node();
  placeholder->set_name("x");
  placeholder->set_op("Placeholder");
  placeholder->mutable_attr()->insert({"dtype", dtype_attr});
  placeholder->mutable_attr()->insert({"shape", shape_attr});

  auto* identity = def.add_node();
  identity->set_name("y");
  identity->set_op("Identity");
  identity->add_input("x");
  identity->mutable_attr()->insert({"T", dtype_attr});

  return def;
}

void TestFeedAndFetchTensorsInDeviceMemory(
    const SessionOptions& session_options, DataType dtype) {
  std::unique_ptr<Session> session(NewSession(session_options));
  const string gpu_device_name = GPUDeviceName(session.get());
  if (gpu_device_name.empty()) {
    LOG(INFO) << "Skipping test since no GPU is available";
    return;
  }

  TF_ASSERT_OK(session->Create(CreateIdentityGraphDef(dtype)))
      << DataType_Name(dtype);

  CallableOptions opts;
  opts.add_feed("x:0");
  opts.add_fetch("y:0");

  Tensor gpu_tensor;
  Tensor host_tensor(dtype, {3});
  {
    // Ask for the fetched tensor to be backed by device memory.
    // Even though the kernel that created the tensor produced it in host
    // memory.
    opts.mutable_fetch_devices()->insert({"y:0", gpu_device_name});
    opts.set_fetch_skip_sync(true);
    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(opts, &handle)) << DataType_Name(dtype);
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {host_tensor}, &outputs, nullptr))
        << DataType_Name(dtype);
    TF_ASSERT_OK(session->ReleaseCallable(handle)) << DataType_Name(dtype);
    ASSERT_EQ(1, outputs.size()) << DataType_Name(dtype);
    gpu_tensor = outputs[0];
    ASSERT_TRUE(IsCUDATensor(gpu_tensor)) << DataType_Name(dtype);
  }

  {
    // Feed a tensor backed by device memory, even though the operations in the
    // graph expect it in host memory.
    opts.clear_fetch_devices();
    opts.mutable_feed_devices()->insert({"x:0", gpu_device_name});
    Session::CallableHandle handle;
    TF_ASSERT_OK(session->MakeCallable(opts, &handle)) << DataType_Name(dtype);
    std::vector<Tensor> outputs;
    TF_ASSERT_OK(session->RunCallable(handle, {gpu_tensor}, &outputs, nullptr))
        << DataType_Name(dtype);
    TF_ASSERT_OK(session->ReleaseCallable(handle)) << DataType_Name(dtype);
    ASSERT_EQ(1, outputs.size());
    const StringPiece actual_data = outputs[0].tensor_data();
    const StringPiece expected_data = host_tensor.tensor_data();
    EXPECT_EQ(expected_data.size(), actual_data.size()) << DataType_Name(dtype);
    EXPECT_EQ(0, memcmp(expected_data.data(), actual_data.data(),
                        std::min(expected_data.size(), actual_data.size())))
        << DataType_Name(dtype);
  }
}

void TestFeedAndFetchTensorsInDeviceMemoryFailsToMakeCallable(
    const SessionOptions& session_options, DataType dtype) {
  std::unique_ptr<Session> session(NewSession(session_options));
  const string gpu_device_name = GPUDeviceName(session.get());
  if (gpu_device_name.empty()) {
    LOG(INFO) << "Skipping test since no GPU is available";
    return;
  }

  TF_ASSERT_OK(session->Create(CreateIdentityGraphDef(dtype)))
      << DataType_Name(dtype);

  CallableOptions opts;
  opts.add_feed("x:0");
  opts.add_fetch("y:0");

  // Fail when asking to fetch into GPU memory.
  {
    opts.mutable_fetch_devices()->insert({"y:0", gpu_device_name});
    opts.set_fetch_skip_sync(true);
    Session::CallableHandle handle;
    Status status = session->MakeCallable(opts, &handle);
    EXPECT_FALSE(status.ok()) << DataType_Name(dtype);
    EXPECT_TRUE(absl::StrContains(
        status.error_message(),
        strings::StrCat(
            "Cannot feed or fetch tensor 'y:0' from device ", gpu_device_name,
            " as feeding/fetching from GPU devices is not yet supported for ",
            DataTypeString(dtype), " tensors")))
        << DataType_Name(dtype) << ", Status: " << status;
  }

  // Fail when feeding from GPU memory.
  {
    opts.clear_feed_devices();
    opts.mutable_feed_devices()->insert({"x:0", gpu_device_name});
    Session::CallableHandle handle;
    Status status = session->MakeCallable(opts, &handle);
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(absl::StrContains(
        status.error_message(),
        strings::StrCat(
            "Cannot feed or fetch tensor 'x:0' from device ", gpu_device_name,
            " as feeding/fetching from GPU devices is not yet supported for ",
            DataTypeString(dtype), " tensors")))
        << DataType_Name(dtype) << ", Status: " << status;
  }
}

void TestFeedAndFetchTensorsInDeviceMemoryForAllDataTypes(
    const SessionOptions& opts) {
  // Feeding/fetching on device does not work for all DataTypes as it
  // relies on the implementation of the _Arg and _Retval kernels which
  // are not registered for some types or consume/produce inputs/outputs
  // in host memory for some types.
  //
  // Run through all datatypes to validate that either:
  // (a) MakeCallable fails (because the given type cannot be fed/fetched
  //     in device memory),
  //     OR
  // (b) Succeeds: RunCallable should gladly accept inputs in device memory
  //     and produce output tensors in device memory.
  for (int i = DataType_MIN; i <= DataType_MAX; ++i) {
    if (!DataType_IsValid(i)) continue;
    const DataType dtype = static_cast<DataType>(i);
    switch (dtype) {
      case DT_INVALID:
        break;
      case DT_BFLOAT16:
      case DT_BOOL:
      case DT_COMPLEX128:
      case DT_COMPLEX64:
      case DT_DOUBLE:
      case DT_FLOAT:
      case DT_HALF:
      case DT_INT16:
      case DT_INT64:
      case DT_INT8:
      case DT_UINT16:
      case DT_UINT8:
        TestFeedAndFetchTensorsInDeviceMemory(opts, dtype);
        break;
      default:
        // Ignore all REF types since Tensors of this type aren't intended to
        // be fed (and attempting to create one via the Tensor constructor
        // will result in a LOG(FATAL)).
        if (!IsRefType(dtype)) {
          TestFeedAndFetchTensorsInDeviceMemoryFailsToMakeCallable(opts, dtype);
        }
        break;
    }
  }
}

TEST(DirectSessionTest, FeedAndFetchTensorsInDeviceMemory_AllDataTypes) {
  SessionOptions opts;
  opts.config.set_allow_soft_placement(false);
  TestFeedAndFetchTensorsInDeviceMemoryForAllDataTypes(opts);
}

TEST(DirectSessionTest,
     FeedAndFetchTensorsInDeviceMemory_AllDataTypes_SoftPlacement) {
  SessionOptions opts;
  opts.config.set_allow_soft_placement(true);
  TestFeedAndFetchTensorsInDeviceMemoryForAllDataTypes(opts);
}

// A simple benchmark for the overhead of `DirectSession::Run()` calls
// with varying numbers of feeds/fetches.
void FeedFetchBenchmarkHelper(::testing::benchmark::State& state, int num_feeds,
                              bool use_make_callable, int inter_op_threads,
                              bool use_single_threaded_executor) {
  Tensor value(DT_FLOAT, TensorShape());
  value.flat<float>()(0) = 37.0;

  std::vector<std::pair<string, Tensor>> inputs;
  inputs.reserve(num_feeds);
  std::vector<string> outputs;

  Graph g(OpRegistry::Global());
  for (int i = 0; i < num_feeds; ++i) {
    // NOTE(mrry): We pin nodes to the "/cpu:0" device, so as not to
    // measure CPU<->GPU copying overhead. We should also optimize and
    // monitor this overhead where possible, but that is not the
    // object of study in this benchmark.
    Node* placeholder;
    TF_CHECK_OK(NodeBuilder(g.NewName("Placeholder"), "Placeholder")
                    .Attr("shape", TensorShape())
                    .Attr("dtype", DT_FLOAT)
                    .Device("/cpu:0")
                    .Finalize(&g, &placeholder));
    Node* identity;
    TF_CHECK_OK(NodeBuilder(g.NewName("Identity"), "Identity")
                    .Input(placeholder)
                    .Attr("T", DT_FLOAT)
                    .Device("/cpu:0")
                    .Finalize(&g, &identity));
    inputs.push_back({placeholder->name() + ":0", value});
    outputs.push_back(identity->name() + ":0");
  }
  GraphDef gd;
  g.ToGraphDef(&gd);
  SessionOptions opts;
  opts.config.set_inter_op_parallelism_threads(inter_op_threads);
  if (use_single_threaded_executor) {
    opts.config.mutable_experimental()->set_executor_type(
        "SINGLE_THREADED_EXECUTOR");
  }
  std::unique_ptr<Session> session(NewSession(opts));
  TF_CHECK_OK(session->Create(gd));
  if (use_make_callable) {
    Session::CallableHandle handle;
    CallableOptions callable_options;
    std::vector<Tensor> input_tensors;
    for (const auto& input : inputs) {
      callable_options.add_feed(input.first);
      input_tensors.push_back(input.second);
    }
    for (const string& output : outputs) {
      callable_options.add_fetch(output);
    }
    TF_CHECK_OK(session->MakeCallable(callable_options, &handle));

    for (auto s : state) {
      std::vector<Tensor> output_values;
      TF_CHECK_OK(
          session->RunCallable(handle, input_tensors, &output_values, nullptr));
    }
  } else {
    {
      // NOTE(mrry): Ignore the first run, which will incur the graph
      // partitioning/pruning overhead and skew the results.
      //
      // Note that we should also optimize and monitor the overhead on
      // the first run, which will impact application startup times, but
      // that is not the object of study in this benchmark.
      std::vector<Tensor> output_values;
      TF_CHECK_OK(session->Run(inputs, outputs, {}, &output_values));
    }

    for (auto s : state) {
      std::vector<Tensor> output_values;
      TF_CHECK_OK(session->Run(inputs, outputs, {}, &output_values));
    }
  }
}

void BM_FeedFetch(::testing::benchmark::State& state) {
  const int num_feeds = state.range(0);

  FeedFetchBenchmarkHelper(state, num_feeds, /* use_make_callable */ false,
                           /* inter_op_threads */ 0,
                           /* use_single_threaded_executor */ false);
}
void BM_FeedFetchCallable(::testing::benchmark::State& state) {
  const int num_feeds = state.range(0);

  FeedFetchBenchmarkHelper(state, num_feeds, /* use_make_callable */ true,
                           /* inter_op_threads */ 0,
                           /* use_single_threaded_executor */ false);
}
void BM_FeedFetchCallableSingleThread(::testing::benchmark::State& state) {
  const int num_feeds = state.range(0);

  FeedFetchBenchmarkHelper(state, num_feeds, /* use_make_callable */ true,
                           /* inter_op_threads */ -1,
                           /* use_single_threaded_executor */ false);
}
void BM_FeedFetchCallableSingleThreadExecutor(
    ::testing::benchmark::State& state) {
  const int num_feeds = state.range(0);

  FeedFetchBenchmarkHelper(state, num_feeds, /* use_make_callable */ true,
                           /* inter_op_threads */ -1,
                           /* use_single_threaded_executor */ true);
}

BENCHMARK(BM_FeedFetch)->Arg(1)->Arg(2)->Arg(5)->Arg(10);
BENCHMARK(BM_FeedFetchCallable)->Arg(1)->Arg(2)->Arg(5)->Arg(10);
BENCHMARK(BM_FeedFetchCallableSingleThread)->Arg(1)->Arg(2)->Arg(5)->Arg(10);
BENCHMARK(BM_FeedFetchCallableSingleThreadExecutor)
    ->Arg(1)
    ->Arg(2)
    ->Arg(5)
    ->Arg(10);

}  // namespace

class DirectSessionCollectiveTest : public ::testing::Test {
 public:
  // Creates a graph with CollectiveOps inside functions and runs it.  Returns
  // the generated collective_graph_key.
  Status RunGraphWithCollectiveFunctions(bool add_unused_function,
                                         int64* collective_graph_key) {
    GraphDef g = CreateGraph(add_unused_function);
    const Tensor t1 =
        test::AsTensor<float>({0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1});
    const Tensor t2 =
        test::AsTensor<float>({0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3});
    auto session = CreateSession();
    TF_RETURN_IF_ERROR(session->Create(g));
    std::vector<Tensor> outputs;
    TF_RETURN_IF_ERROR(
        session->Run({{"input0:0", t1}, {"input1:0", t2}}, {},
                     {"collective_call0:0", "collective_call1:0"}, &outputs));
    DirectSession* direct_session = static_cast<DirectSession*>(session.get());
    {
      mutex_lock l(direct_session->collective_graph_key_lock_);
      *collective_graph_key = direct_session->collective_graph_key_;
    }
    return Status::OK();
  }

 private:
  // Creates a function with name `function_name` and a single CollectiveReduce
  // node with instance key set as `instance_key`.
  FunctionDef CollectiveFunction(const string& function_name,
                                 int instance_key) {
    return FunctionDefHelper::Define(
        // Function name
        function_name,
        // In def
        {"arg:float"},
        // Out def
        {"reduce:float"},
        // Attr def
        {},
        // Node def
        {{
            {"reduce"},
            "CollectiveReduce",
            {"arg"},
            {{"group_size", 2},
             {"group_key", 1},
             {"instance_key", instance_key},
             {"subdiv_offsets", gtl::ArraySlice<int32>({0})},
             {"merge_op", "Add"},
             {"final_op", "Div"},
             {"T", DT_FLOAT}},
        }});
  }

  NodeDef Input(int id) {
    AttrValue dtype_attr;
    SetAttrValue(DT_FLOAT, &dtype_attr);
    NodeDef input;
    input.set_name(strings::StrCat("input", id));
    input.set_op("Placeholder");
    input.mutable_attr()->insert({"dtype", dtype_attr});
    return input;
  }

  NodeDef CollectiveCall(const string& op, const string& input, int cpu_id) {
    NodeDef collective_call;
    collective_call.set_name(strings::StrCat("collective_call", cpu_id));
    collective_call.set_op(op);
    collective_call.add_input(input);
    collective_call.set_device(
        strings::StrCat("/job:localhost/replica:0/task:0/device:CPU:", cpu_id));
    return collective_call;
  }

  // Creates a GraphDef that adds two CollectiveFunctions, one each on CPU0 and
  // CPU1, with instance_key 1, and appropriate placeholder inputs.  If
  // `add_unused_function` is true, adds another CollectiveFunction with
  // instance_key 2 that is not invoked in the graph.
  GraphDef CreateGraph(bool add_unused_function) {
    GraphDef g;
    FunctionDef collective_function =
        CollectiveFunction("CollectiveFunction1", 1);
    FunctionDefLibrary* lib = g.mutable_library();
    *lib->add_function() = collective_function;
    if (add_unused_function) {
      FunctionDef unused_function =
          CollectiveFunction("CollectiveFunction2", 2);
      *lib->add_function() = unused_function;
    }

    *g.add_node() = Input(0);
    *g.add_node() = Input(1);
    // CollectiveReduce on CPU0 with instance_key 1.
    *g.add_node() = CollectiveCall("CollectiveFunction1", "input0", 0);
    // CollectiveReduce on CPU1 with instance_key 1.
    *g.add_node() = CollectiveCall("CollectiveFunction1", "input1", 1);

    return g;
  }
};

TEST_F(DirectSessionCollectiveTest,
       TestCollectiveGraphKeyUsesOnlyCalledFunctions) {
  int64 key1;
  TF_ASSERT_OK(RunGraphWithCollectiveFunctions(false, &key1));
  int64 key2;
  TF_ASSERT_OK(RunGraphWithCollectiveFunctions(true, &key2));
  ASSERT_EQ(key1, key2);
}

// Accesses the cancellation manager for the step after the step has been
// cancelled.
class StatefulOutputRequiredOp : public OpKernel {
 public:
  explicit StatefulOutputRequiredOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {}
  void Compute(OpKernelContext* ctx) override {
    // The op counts the number of outputs required in the current subgraph,
    // and emits that number on each of its required outputs.
    Tensor count_outputs_required_t(int64{0});
    int64& count_outputs_required = count_outputs_required_t.scalar<int64>()();
    for (int i = 0; i < num_outputs(); ++i) {
      if (ctx->output_required(i)) ++count_outputs_required;
    }
    for (int i = 0; i < num_outputs(); ++i) {
      if (ctx->output_required(i)) ctx->set_output(i, count_outputs_required_t);
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("StatefulOutputRequired").Device(DEVICE_CPU),
                        StatefulOutputRequiredOp);
REGISTER_OP("StatefulOutputRequired")
    .Output("results : num_outs * int64")
    .Attr("num_outs : int = 5")
    .SetIsStateful();

TEST(DirectSessionTest, TestStatefulOutputRequiredOp) {
  GraphDef graph;
  // Creates a graph with a StatefulOutputRequired op with 5 outputs.
  protobuf::TextFormat::ParseFromString(
      R"proto(
        node { name: 'n' op: 'StatefulOutputRequired' device: '/device:CPU:0' }
        versions { producer: 9 }
      )proto",
      &graph);

  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(std::move(graph)));

  // As a stateful op, a single StatefulOutputRequired kernel will be created
  // and shared across multiple subgraphs. We create 5 different subgraphs,
  // fetching different prefixes of the output of the op.
  for (int num_outputs_required = 1; num_outputs_required <= 5;
       ++num_outputs_required) {
    std::vector<string> fetch_tensor_names;
    fetch_tensor_names.reserve(num_outputs_required);
    for (int output_idx = 0; output_idx < num_outputs_required; ++output_idx) {
      fetch_tensor_names.push_back(strings::StrCat("n:", output_idx));
    }
    std::vector<Tensor> fetch_tensors;
    TF_ASSERT_OK(session->Run({}, fetch_tensor_names, {}, &fetch_tensors));
    ASSERT_EQ(num_outputs_required, fetch_tensors.size());
    for (const Tensor& t : fetch_tensors) {
      ASSERT_EQ(num_outputs_required, t.scalar<int64>()());
    }
  }

  TF_ASSERT_OK(session->Close());
}

}  // namespace tensorflow
