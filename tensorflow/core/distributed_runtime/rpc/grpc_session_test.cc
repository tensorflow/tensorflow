/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/rpc/grpc_session.h"

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

static SessionOptions Devices(int num_cpus, int num_gpus) {
  SessionOptions result;
  (*result.config.mutable_device_count())["CPU"] = num_cpus;
  (*result.config.mutable_device_count())["GPU"] = num_gpus;
  return result;
}

void CreateGraphDef(GraphDef* graph_def, string node_names[3]) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({1, 2}));
  test::FillValues<float>(&a_tensor, {1, 2});
  Node* a = test::graph::Constant(&graph, a_tensor);
  node_names[0] = a->name();

  Tensor b_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&b_tensor, {2, 1});
  Node* b = test::graph::Constant(&graph, b_tensor);
  node_names[1] = b->name();

  Node* c = test::graph::Matmul(&graph, a, b, false, false);
  node_names[2] = c->name();

  test::graph::ToGraphDef(&graph, graph_def);
}

// Asserts that "val" is a single float tensor. The only float is
// "expected_val".
static void IsSingleFloatValue(const Tensor& val, float expected_val) {
  ASSERT_EQ(val.dtype(), DT_FLOAT);
  ASSERT_EQ(val.NumElements(), 1);
  ASSERT_EQ(val.flat<float>()(0), expected_val);
}

static SessionOptions Options(const string& target, int placement_period) {
  SessionOptions options;
  // NOTE(mrry): GrpcSession requires a grpc:// scheme prefix in the target
  // string.
  options.target = strings::StrCat("grpc://", target);
  options.config.set_placement_period(placement_period);
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions::L0);
  return options;
}

static Session* NewRemote(const SessionOptions& options) {
  return CHECK_NOTNULL(NewSession(options));
}

TEST(GrpcSessionTest, BasicNonProtoAPI) {
  GraphDef graph;
  string node_names[3];
  // c = a * b
  CreateGraphDef(&graph, node_names);

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1)));
  ASSERT_TRUE(session != nullptr);

  for (int iters = 0; iters < 25; ++iters) {
    TF_CHECK_OK(session->Create(graph));
    {
      // Just run to target node
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<string> targets = {node_names[2]};
      TF_CHECK_OK(session->Run(inputs, {}, targets, nullptr));
    }
    {
      // Run to a target node and a real tensor
      std::vector<std::pair<string, Tensor>> inputs;
      std::vector<string> names = {node_names[2] + ":0"};
      std::vector<string> targets = {node_names[1]};
      std::vector<Tensor> outputs;
      TF_CHECK_OK(session->Run(inputs, names, targets, &outputs));
      ASSERT_TRUE(outputs[0].IsInitialized());
      ASSERT_EQ(4.0, outputs[0].flat<float>()(0));
    }

    TF_CHECK_OK(session->Close());
  }
}

TEST(GrpcSessionTest, BasicNonProtoAPIConsistentOrder) {
  GraphDef graph;
  string node_names[3];
  // c = a * b
  CreateGraphDef(&graph, node_names);

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1)));
  ASSERT_TRUE(session != nullptr);
  ASSERT_TRUE(session->Create(graph).ok());

  // Test that the order of the output names matches the order of the
  // returned Tensors.
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> names = {node_names[2] + ":0", node_names[0] + ":0",
                               node_names[1] + ":0"};

  std::vector<string> target_ops = {node_names[1]};
  std::vector<Tensor> outputs;
  ASSERT_TRUE(session->Run(inputs, names, target_ops, &outputs).ok());
  ASSERT_TRUE(outputs[0].IsInitialized());
  ASSERT_EQ(4.0, outputs[0].flat<float>()(0));
  ASSERT_TRUE(outputs[1].IsInitialized());
  ASSERT_EQ(1.0, outputs[1].flat<float>()(0));
  ASSERT_TRUE(outputs[2].IsInitialized());
  ASSERT_EQ(2.0, outputs[2].flat<float>()(0));
  ASSERT_TRUE(session->Close().ok());
}

TEST(GrpcSessionTest, NonLocalWithFilters) {
  GraphDef graph;
  string node_names[3];
  // c = a * b
  CreateGraphDef(&graph, node_names);

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  SessionOptions options;
  options.target = strings::StrCat("grpc://", cluster->targets()[0]);
  options.config.add_device_filters(cluster->devices()[0].name());

  std::unique_ptr<Session> session(NewRemote(options));
  ASSERT_TRUE(session != nullptr);

  {
    GraphDef graph_copy(graph);
    graph::SetDefaultDevice(cluster->devices()[0].name(), &graph_copy);
    TF_CHECK_OK(session->Create(graph_copy));
    TF_CHECK_OK(session->Run({}, {}, {node_names[2]}, nullptr));
    TF_CHECK_OK(session->Close());
  }
  {
    GraphDef graph_copy(graph);
    graph::SetDefaultDevice(cluster->devices()[1].name(), &graph_copy);
    auto status = session->Create(graph_copy);
    EXPECT_EQ(tensorflow::error::INVALID_ARGUMENT, status.code());
  }
}

TEST(GrpcSessionTest, FetchMultipleTimes) {
  GraphDef graph;
  string node_names[3];
  CreateGraphDef(&graph, node_names);

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1)));
  ASSERT_TRUE(session != nullptr);

  TF_CHECK_OK(session->Create(graph));
  const std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;

  const string node = node_names[2] + ":0";
  TF_CHECK_OK(session->Run(inputs, {node, node}, {}, &outputs));
  EXPECT_EQ(2, outputs.size());
  for (int i = 0; i < outputs.size(); ++i) {
    const Tensor& t = outputs[i];
    ASSERT_TRUE(t.IsInitialized()) << i;
    ASSERT_EQ(4.0, t.flat<float>()(0)) << i;
  }
  TF_CHECK_OK(session->Close());
}

// A = [3 2; -1 0]; x = rand(2, 1); We want to compute the largest
// eigenvalue for A, which is 2.0. Iteratively, we do
//   repeat x = y / y.norm(); y = A * x; end
// At the end, we expect "lambda" converges to 2.0.
void FindMaxEigen(const string& target) {
  Graph graph(OpRegistry::Global());

  Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
  // Store rows [3, 2] and [-1, 0] in row major format.
  test::FillValues<float>(&a_tensor, {3, 2, -1, 0});
  Node* a = test::graph::Constant(&graph, a_tensor);

  // x is from the feed.
  Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
  test::FillValues<float>(&x_tensor, {0, 0});
  Node* x = test::graph::Constant(&graph, x_tensor);

  // y = A * x
  Node* y = test::graph::Matmul(&graph, a, x, false, false);

  // y2 = y.^2
  Node* y2 = test::graph::Unary(&graph, "Square", y);

  // const tensor for reduction
  Tensor rdim_tensor(DT_INT32, TensorShape({}));
  rdim_tensor.scalar<int32>()() = 0;
  Node* rdim = test::graph::Constant(&graph, rdim_tensor);

  // y2_sum = sum(y2)
  Node* y2_sum = test::graph::Reduce(&graph, "Sum", y2, rdim);

  // y_norm = sqrt(y2_sum)
  Node* y_norm = test::graph::Unary(&graph, "Sqrt", y2_sum);

  // y_normalized = y ./ y_norm
  Node* y_normalized = test::graph::Binary(&graph, "Div", y, y_norm);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  std::unique_ptr<Session> session(NewRemote(Options(target, 1)));
  ASSERT_TRUE(session != nullptr);
  TF_CHECK_OK(session->Create(def));

  // Setup feeds and fetches.
  float lambda;
  Tensor feed_value(DT_FLOAT, TensorShape({2, 1}));
  feed_value.matrix<float>()(0, 0) = -3.1415;
  feed_value.matrix<float>()(1, 0) = +2.7183;

  for (int i = 0; i < 25; ++i) {
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({{x->name(), feed_value}},
                             {y->name(), y_normalized->name()}, {}, &outputs));
    const Tensor& y = outputs[0];
    const Tensor& y_normalized = outputs[1];
    // Print out lambda, x, and y.
    CHECK_EQ(2, feed_value.NumElements());
    CHECK_EQ(2, y.NumElements());
    lambda = y.flat<float>()(0) / feed_value.flat<float>()(0);
    printf("%06d lambda = %8.6f x = [%8.6f %8.6f] y = [%8.6f %8.6f]\n", i,
           lambda, feed_value.flat<float>()(0), feed_value.flat<float>()(1),
           y.flat<float>()(0), y.flat<float>()(1));
    // Copies y_normalized to  *x.
    feed_value = y_normalized;
  }
  EXPECT_NEAR(2.0, lambda, 1e-6);
}

TEST(FindMaxEigenTest, RemoteDevice) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));
  FindMaxEigen(cluster->targets()[0]);
}

void SetDevice(GraphDef* graph, const string& name, const string& dev) {
  for (int i = 0; i < graph->node_size(); ++i) {
    if (graph->node(i).name() == name) {
      graph->mutable_node(i)->set_device(dev);
      return;
    }
  }
  LOG(FATAL) << "Name '" << name << "' not found.";
}

// TODO(b/32636929): This test fails 1/1000 times. Disable it while we
// figure out why.
TEST(GrpcSessionTest, DISABLED_MultiDevices) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  Graph graph(OpRegistry::Global());
  const int kSize = 1048576;

  // c = a * b = 2 * 3 * kSize
  Tensor a_tensor(DT_FLOAT, TensorShape({1, kSize}));
  Tensor b_tensor(DT_FLOAT, TensorShape({kSize, 1}));
  for (int i = 0; i < kSize; ++i) {
    a_tensor.flat<float>()(i) = 2;
    b_tensor.flat<float>()(i) = 3;
  }
  Node* a = test::graph::Constant(&graph, a_tensor);
  Node* b = test::graph::Constant(&graph, b_tensor);
  Node* c = test::graph::Matmul(&graph, a, b, false, false);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  // In this test, we force each node (a, b, c) on every possible device.
  // We test all possible cases.
  for (const auto& a_dev : cluster->devices()) {
    for (const auto& b_dev : cluster->devices()) {
      for (const auto& c_dev : cluster->devices()) {
        LOG(INFO) << "a: " << a_dev.name() << " b: " << b_dev.name()
                  << " c: " << c_dev.name();

        SetDevice(&def, a->name(), a_dev.name());
        SetDevice(&def, b->name(), b_dev.name());
        SetDevice(&def, c->name(), c_dev.name());

        std::unique_ptr<Session> session(
            NewRemote(Options(cluster->targets()[0], 1000)));
        ASSERT_TRUE(session != nullptr);
        TF_CHECK_OK(session->Create(def));
        {
          std::vector<Tensor> outputs;
          RunOptions options;
          options.set_trace_level(RunOptions::FULL_TRACE);
          RunMetadata metadata;
          TF_CHECK_OK(
              session->Run(options, {}, {c->name()}, {}, &outputs, &metadata));
          ASSERT_EQ(1, outputs.size());
          IsSingleFloatValue(outputs[0], 6.0 * kSize);

          const StepStats& ss = metadata.step_stats();
          // NOTE(mrry): We only assert that `c` is placed correctly,
          // because the current placement algorithm will move its
          // inputs to be colocated with it, when it is the sole
          // consumer.
          bool c_placed_correctly = false;
          for (const auto& dev : ss.dev_stats()) {
            for (const auto& node : dev.node_stats()) {
              if (node.node_name() == c->name() &&
                  dev.device() == c_dev.name()) {
                c_placed_correctly = true;
              }
            }
          }
          ASSERT_TRUE(c_placed_correctly);
        }
        TF_CHECK_OK(session->Close());
      }
    }
  }
}

TEST(GrpcSessionTest, LargeTensorSend) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  Graph graph(OpRegistry::Global());

  // Define a 3 GB fill result.
  Tensor fill_shape_tensor(DT_INT32, TensorShape({4}));
  fill_shape_tensor.vec<int32>()(0) = 1;
  fill_shape_tensor.vec<int32>()(1) = 256;
  fill_shape_tensor.vec<int32>()(2) = 1024;
  fill_shape_tensor.vec<int32>()(3) = 1024;
  Node* fill_shape_node = test::graph::Constant(&graph, fill_shape_tensor);

  Tensor fill_val_tensor(DT_FLOAT, TensorShape({}));
  fill_val_tensor.flat<float>()(0) = 1.0;
  Node* fill_val_node = test::graph::Constant(&graph, fill_val_tensor);

  Node* fill_node =
      test::graph::Binary(&graph, "Fill", fill_shape_node, fill_val_node);

  Tensor max_axes_tensor(DT_INT32, TensorShape({4}));
  max_axes_tensor.vec<int32>()(0) = 0;
  max_axes_tensor.vec<int32>()(1) = 1;
  max_axes_tensor.vec<int32>()(2) = 2;
  max_axes_tensor.vec<int32>()(3) = 3;
  Node* max_axes_node = test::graph::Constant(&graph, max_axes_tensor);
  Node* max_node = test::graph::Reduce(&graph, "Max", fill_node, max_axes_node);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  SetDevice(&def, fill_node->name(), cluster->devices()[0].name());
  SetDevice(&def, fill_node->name(), cluster->devices()[1].name());

  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1000)));
  ASSERT_TRUE(session != nullptr);
  TF_CHECK_OK(session->Create(def));
  {
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({}, {max_node->name()}, {}, &outputs));
    ASSERT_EQ(1, outputs.size());
    IsSingleFloatValue(outputs[0], 1.0);
  }
  TF_CHECK_OK(session->Close());
}

TEST(GrpcSessionTest, MultiDevices_String) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 1), 2, &cluster));
  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1000)));
  ASSERT_TRUE(session != nullptr);

  // b = a
  Graph graph(OpRegistry::Global());
  Tensor a_tensor(DT_STRING, TensorShape({2, 2}));
  for (int i = 0; i < 4; ++i) {
    a_tensor.flat<string>()(i) = "hello, world";
  }
  Node* a = test::graph::Constant(&graph, a_tensor);
  Node* b = test::graph::Identity(&graph, a);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  // In this test, we force each node (a, b) on every possible device.
  // We test all possible cases.
  for (const auto& a_dev : cluster->devices()) {
    for (const auto& b_dev : cluster->devices()) {
      LOG(INFO) << "a: " << a_dev.name() << " b: " << b_dev.name();
      SetDevice(&def, a->name(), a_dev.name());
      SetDevice(&def, b->name(), b_dev.name());

      Status s = session->Create(def);
      if (s.ok()) {
        std::vector<Tensor> outputs;
        TF_CHECK_OK(session->Run({}, {b->name()}, {}, &outputs));
        ASSERT_EQ(1, outputs.size());
        ASSERT_EQ(outputs[0].dtype(), DT_STRING);
        ASSERT_EQ(outputs[0].NumElements(), 4);
        for (int i = 0; i < outputs[0].NumElements(); ++i) {
          EXPECT_EQ(outputs[0].flat<string>()(i), "hello, world");
        }
        TF_CHECK_OK(session->Close());
      } else {
        LOG(ERROR) << "Error: " << s;
        ASSERT_TRUE((a_dev.device_type() == DEVICE_GPU) ||
                    (b_dev.device_type() == DEVICE_GPU));
        ASSERT_FALSE(s.ok());
      }
    }
  }
}

TEST(GrpcSessionTest, SendRecv_Node_Naming) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 3, &cluster));
  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1)));
  ASSERT_TRUE(session != nullptr);

  // This test case needs at least 3 devices.
  CHECK_GE(cluster->devices().size(), 3);
  const DeviceAttributes& src = cluster->devices()[0];
  const DeviceAttributes& dst0 = cluster->devices()[1];
  const DeviceAttributes& dst1 = cluster->devices()[2];
  LOG(INFO) << "src = " << src.name() << " dst0 = " << dst0.name()
            << " dst1 = " << dst1.name();

  // Within the same session, we compute two subgraphs:
  //   1) a on 'src' sends to b on 'dst0';
  //   2) a on 'src' sends to c on 'dst1'.
  Graph graph(OpRegistry::Global());
  Tensor a_tensor(DT_FLOAT, TensorShape({1, 1}));
  a_tensor.flat<float>()(0) = 100;
  Node* a = test::graph::Constant(&graph, a_tensor);
  Node* b = test::graph::Identity(&graph, a);
  Node* c = test::graph::Identity(&graph, a);

  GraphDef def;
  test::graph::ToGraphDef(&graph, &def);

  // The base graph have a, b, c, assigned to devices explicitly.
  SetDevice(&def, a->name(), src.name());
  SetDevice(&def, b->name(), dst0.name());
  SetDevice(&def, c->name(), dst1.name());
  TF_CHECK_OK(session->Create(def));

  // Run subgraph a -> b, and fetch b.
  {
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({}, {b->name()}, {}, &outputs));
    ASSERT_EQ(1, outputs.size());
    IsSingleFloatValue(outputs[0], 100);
  }

  // Run subgraph a -> c, and fetch c.
  {
    std::vector<Tensor> outputs;
    TF_CHECK_OK(session->Run({}, {c->name()}, {}, &outputs));
    ASSERT_EQ(1, outputs.size());
    IsSingleFloatValue(outputs[0], 100);
  }

  TF_CHECK_OK(session->Close());
}

TEST(GrpcSessionTest, Error) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));
  const string& master = cluster->targets()[0];
  const string& dev_a = cluster->devices()[0].name();
  const string& dev_b = cluster->devices()[1].name();
  LOG(INFO) << "master " << master << "dev_a " << dev_a << "dev_b " << dev_b;
  GraphDef gdef;
  std::vector<string> fetches;
  {
    Graph g(OpRegistry::Global());

    // a2 = a + error(a)
    //
    // Subgraph for "a" fails. The master will cancel the subgraph for
    // "b" and then returns the Session::Run.
    auto a = test::graph::Constant(&g, Tensor());
    a->set_assigned_device_name(dev_a);
    auto a_err = test::graph::Error(&g, a, "fantasia!");
    a_err->set_assigned_device_name(dev_a);
    auto a2 = test::graph::Add(&g, a, a_err);
    a2->set_assigned_device_name(dev_a);
    fetches.push_back(a2->name());

    // b2 = b + delay(b)
    //
    // Subgraph for "b" sleeps at the node "b_delay". When the sleep
    // finishes, the subgraph "b" will continue execution till it
    // notices that it is canceled. Meanwhile, subgraph's executor
    // and its related state (registered ops) should still be alive.
    auto b = test::graph::Constant(&g, Tensor());
    b->set_assigned_device_name(dev_b);
    auto b_delay = test::graph::Delay(&g, b, Microseconds(1000000));
    b_delay->set_assigned_device_name(dev_b);
    auto b2 = test::graph::Add(&g, b, b_delay);
    b2->set_assigned_device_name(dev_b);
    fetches.push_back(b2->name());
    test::graph::ToGraphDef(&g, &gdef);
  }
  std::unique_ptr<Session> session(NewRemote(Options(master, 1)));
  ASSERT_TRUE(session != nullptr);

  TF_CHECK_OK(session->Create(gdef));
  {
    Status status = session->Run({}, fetches, {}, nullptr);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("fantasia!"), string::npos);
  }
  // session->Close() shall clean up all states related to the session->
  // E.g., deregisters subgraph with workers, etc.
  TF_CHECK_OK(session->Close());

  // Sleep a bit so that most of asynchronous works finishes before
  // the test process finishes.
  Env::Default()->SleepForMicroseconds(2000000);
}

TEST(GrpcSessionTest, LongErrorMessage) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));
  const string& master = cluster->targets()[0];
  const string& dev_a = cluster->devices()[0].name();
  const string& dev_b = cluster->devices()[1].name();
  LOG(INFO) << "master " << master << "dev_a " << dev_a << "dev_b " << dev_b;
  GraphDef gdef;
  std::vector<string> fetches;
  {
    Graph g(OpRegistry::Global());

    // a2 = a + error(a)
    //
    // Subgraph for "a" fails. The master will cancel the subgraph for
    // "b" and then returns the Session::Run.
    auto a = test::graph::Constant(&g, Tensor());
    a->set_assigned_device_name(dev_a);
    std::vector<char> long_string_buffer(1024 * 1024, 'x');
    StringPiece long_string(long_string_buffer.data(), 1024 * 1024);
    string name = strings::StrCat(long_string, "fantasia!");
    auto a_err = test::graph::Error(&g, a, name);
    a_err->set_assigned_device_name(dev_a);
    auto a2 = test::graph::Add(&g, a, a_err);
    a2->set_assigned_device_name(dev_a);
    fetches.push_back(a2->name());

    // b2 = b + delay(b)
    //
    // Subgraph for "b" sleeps at the node "b_delay". When the sleep
    // finishes, the subgraph "b" will continue execution till it
    // notices that it is canceled. Meanwhile, subgraph's executor
    // and its related state (registered ops) should still be alive.
    auto b = test::graph::Constant(&g, Tensor());
    b->set_assigned_device_name(dev_b);
    auto b_delay = test::graph::Delay(&g, b, Microseconds(1000000));
    b_delay->set_assigned_device_name(dev_b);
    auto b2 = test::graph::Add(&g, b, b_delay);
    b2->set_assigned_device_name(dev_b);
    fetches.push_back(b2->name());
    test::graph::ToGraphDef(&g, &gdef);
  }
  std::unique_ptr<Session> session(NewRemote(Options(master, 1)));
  ASSERT_TRUE(session != nullptr);

  TF_CHECK_OK(session->Create(gdef));
  {
    Status status = session->Run({}, fetches, {}, nullptr);
    EXPECT_FALSE(status.ok());
    EXPECT_NE(status.ToString().find("fantasia!"), string::npos);
  }
  // session->Close() shall clean up all states related to the session->
  // E.g., deregisters subgraph with workers, etc.
  TF_CHECK_OK(session->Close());

  // Sleep a bit so that most of asynchronous works finishes before
  // the test process finishes.
  Env::Default()->SleepForMicroseconds(2000000);
}

TEST(SessionTest, SharedVar) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 1, &cluster));
  const string master = cluster->targets()[0];
  CHECK_EQ(cluster->devices().size(), 1);

  GraphDef gdef;
  string init_name;
  string inc_name;
  string get_name;
  {
    Graph g(OpRegistry::Global());
    Tensor one(DT_FLOAT, TensorShape({}));
    one.scalar<float>()() = 1.0;
    Node* var = test::graph::Var(&g, DT_FLOAT, one.shape());
    Node* init = test::graph::Assign(&g, var, test::graph::Constant(&g, one));
    init_name = init->name();
    Node* update = test::graph::Assign(
        &g, var, test::graph::Add(&g, var, test::graph::Constant(&g, one)));
    inc_name = update->name();
    get_name = var->name();
    test::graph::ToGraphDef(&g, &gdef);
  }

  // Init a variable
  {
    Session* sess = NewRemote(Options(master, 1));
    TF_CHECK_OK(sess->Create(gdef));
    std::vector<std::pair<string, Tensor>> inp;
    TF_CHECK_OK(sess->Run(inp, {}, {init_name}, nullptr));
    TF_CHECK_OK(sess->Close());
    delete sess;
  }

  for (int rep = 1; rep < 10; ++rep) {
    // Update a variable
    {
      Session* sess = NewRemote(Options(master, 1));
      TF_CHECK_OK(sess->Create(gdef));
      std::vector<std::pair<string, Tensor>> inp;
      TF_CHECK_OK(sess->Run(inp, {}, {inc_name}, nullptr));
      TF_CHECK_OK(sess->Close());
      delete sess;
    }

    // Gets the variable's value.
    {
      Session* sess = NewRemote(Options(master, 1));
      TF_CHECK_OK(sess->Create(gdef));
      std::vector<std::pair<string, Tensor>> inp;
      std::vector<Tensor> ret;
      TF_CHECK_OK(sess->Run(inp, {get_name}, {}, &ret));
      ASSERT_EQ(ret.size(), 1);
      EXPECT_EQ(ret[0].scalar<float>()(), 1.0 * (1 + rep));
      TF_CHECK_OK(sess->Close());
      delete sess;
    }
  }
}

void CreateInvalidGraph(const string& graph_def_ascii,
                        const string& error_substring) {
  GraphDef graph;
  CHECK(protobuf::TextFormat::ParseFromString(graph_def_ascii, &graph));

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1)));
  Status s = session->Create(graph);

  ASSERT_FALSE(s.ok());
  EXPECT_NE(s.error_message().find(error_substring), string::npos);
}

TEST(SessionTest, InvalidOpName) {
  CreateInvalidGraph(R"(
    node {
      name: 'a:b' op: 'Const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
  )",
                     "Illegal op name");

  CreateInvalidGraph(R"(
    node {
      name: 'a:0' op: 'Const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
  )",
                     "Illegal op name");

  CreateInvalidGraph(R"(
    node {
      name: '_a' op: 'Const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
  )",
                     "Illegal op name");
}

TEST(SessionTest, InvalidOpInputName) {
  CreateInvalidGraph(R"(
    node {
      name: 'a' op: 'const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
    node {
      name:'b' op:'MatMul' input:'a:first' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                     "Illegal op input name");

  CreateInvalidGraph(R"(
    node {
      name: 'a' op: 'const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
    node {
      name:'b' op:'MatMul' input:'_a' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                     "Illegal op input name");

  CreateInvalidGraph(R"(
    node {
      name: 'a' op: 'const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
    node {
      name:'b' op:'MatMul' input:'_a:0' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                     "Illegal op input name");

  CreateInvalidGraph(R"(
    node {
      name: 'a' op: 'const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
    node {
      name:'b' op:'MatMul' input:'a:01' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                     "Illegal op input name");
}

TEST(SessionTest, ExtendValidation) {
  GraphDef graph;
  bool success = protobuf::TextFormat::ParseFromString(R"(
    node {
      name: 'a' op: 'Const'
      attr { key: 'dtype' value { type: DT_FLOAT } }
      attr { key: 'value' value {
        tensor { dtype: DT_FLOAT tensor_shape { dim [{size:1}, {size:1}] }
                 float_val: [100] }
      } }
    }
  )",
                                                       &graph);
  // NOTE(mrry): CHECK not done inline to avoid a compilation error in
  // open-source (due to a multi-line string in a macro argument).
  ASSERT_TRUE(success);

  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 2, &cluster));

  std::unique_ptr<Session> session(
      NewRemote(Options(cluster->targets()[0], 1)));
  TF_CHECK_OK(session->Create(graph));

  // 1. Fail with an unknown input name.
  GraphDef extension;
  success = protobuf::TextFormat::ParseFromString(R"(
    node {
      name:'b' op:'MatMul' input:'a:first' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                                                  &extension);
  ASSERT_TRUE(success);

  Status s = session->Extend(extension);
  ASSERT_FALSE(s.ok());
  EXPECT_NE(s.error_message().find("Illegal op input name"), string::npos);

  // 2. Succeed with a valid node.
  success = protobuf::TextFormat::ParseFromString(R"(
    node {
      name:'b' op:'MatMul' input:'a' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                                                  &extension);
  ASSERT_TRUE(success);
  TF_CHECK_OK(session->Extend(extension));

  // 2. Fail with a duplicate node.
  success = protobuf::TextFormat::ParseFromString(R"(
    node {
      name:'b' op:'MatMul' input:'a' input:'a'
      attr { key: 'T' value { type: DT_FLOAT } }
      attr { key: 'transpose_a' value { b: false } }
      attr { key: 'transpose_b' value { b: false } }
      attr { key: '_kernel' value { s: 'eigen' } }
    }
  )",
                                                  &extension);
  ASSERT_TRUE(success);
  s = session->Extend(extension);
  ASSERT_FALSE(s.ok());
  EXPECT_NE(s.error_message().find("'b', which was created by a previous call"),
            string::npos);
}
// Tests that Create() with "operation_timeout_in_ms" set times out.
TEST(SessionTest, CreateTimeoutWithSessionOptions) {
  // Creates a RemoteSession with "operation_timeout_in_ms" set to 100.
  SessionOptions options = Options("example.org:2222", 1);
  options.config.set_operation_timeout_in_ms(100);
  std::unique_ptr<Session> session(NewRemote(options));

  // Creates a long running op.
  Graph graph(OpRegistry::Global());
  Node* b = test::graph::Constant(&graph, Tensor());
  test::graph::Delay(&graph, b, Microseconds(1000000));
  GraphDef gdef;
  test::graph::ToGraphDef(&graph, &gdef);
  Status status = session->Create(gdef);
  // Either error is possible, depending on the environment.
  EXPECT_TRUE(error::DEADLINE_EXCEEDED == status.code() ||
              error::UNAVAILABLE == status.code());
}

// Tests that Create() with "timeout_in_ms" in RunOptions set times out.
TEST(SessionTest, CreateTimeoutWithRunOptions) {
  SessionOptions options = Options("example.org:2222", 1);
  std::unique_ptr<Session> session(NewRemote(options));

  // Creates a long running op.
  Graph graph(OpRegistry::Global());
  Node* b = test::graph::Constant(&graph, Tensor());
  test::graph::Delay(&graph, b, Microseconds(1000000));
  GraphDef gdef;
  test::graph::ToGraphDef(&graph, &gdef);
  RunOptions run_options;
  // Sets RunOption timeout_in_ms to 20.
  run_options.set_timeout_in_ms(20);
  Status status = session->Create(run_options, gdef);
  // Either error is possible, depending on the environment.
  EXPECT_TRUE(error::DEADLINE_EXCEEDED == status.code() ||
              error::UNAVAILABLE == status.code());
}

// Tests that Run() with "operation_timeout_in_ms" set times out.
TEST(SessionTest, RunTimeoutWithSessionOptions) {
  // Creates a RemoteSession with "operation_timeout_in_ms" set to 100.
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 1, &cluster));
  SessionOptions options = Options(cluster->targets()[0], 100);
  options.config.set_operation_timeout_in_ms(1);
  std::unique_ptr<Session> session(NewRemote(options));

  // Creates a long running op.
  Graph graph(OpRegistry::Global());
  Node* b = test::graph::Constant(&graph, Tensor());
  Node* b_delay = test::graph::Delay(&graph, b, Microseconds(2000000));
  GraphDef gdef;
  test::graph::ToGraphDef(&graph, &gdef);
  RunOptions run_options;
  TF_CHECK_OK(session->Create(run_options, gdef));

  // Verifies that Run() times out, and the error code is DEADLINE_EXCEEDED.
  std::vector<std::pair<string, Tensor>> inputs;
  Status status = session->Run(inputs, {}, {b_delay->name()}, nullptr);
  // TODO(sherrym): Due to potentially a GRPC bug, we sometimes get
  // GRPC_CHTTP2_INTERNAL_ERROR which is mapped to error::INTERNAL.
  EXPECT_TRUE(error::DEADLINE_EXCEEDED == status.code() ||
              error::INTERNAL == status.code());
}

// Tests that Run() with "timeout_in_ms" set times out.
TEST(SessionTest, RunTimeoutWithRunOptions) {
  std::unique_ptr<test::TestCluster> cluster;
  TF_CHECK_OK(test::TestCluster::MakeTestCluster(Devices(1, 0), 1, &cluster));
  SessionOptions options = Options(cluster->targets()[0], 1);
  std::unique_ptr<Session> session(NewRemote(options));

  // Creates a long running op.
  Graph graph(OpRegistry::Global());
  Node* b = test::graph::Constant(&graph, Tensor());
  Node* b_delay = test::graph::Delay(&graph, b, Microseconds(1000000));
  GraphDef gdef;
  test::graph::ToGraphDef(&graph, &gdef);
  TF_CHECK_OK(session->Create(gdef));

  // Verifies that Run() times out, and the error code is DEADLINE_EXCEEDED.
  std::vector<std::pair<string, Tensor>> inputs;
  RunOptions run_options;
  run_options.set_timeout_in_ms(100);
  Status status = session->Run(run_options, inputs, {}, {b_delay->name()},
                               nullptr, nullptr);
  // TODO(sherrym): Due to potentially a GRPC bug, we sometimes get
  // GRPC_CHTTP2_INTERNAL_ERROR which is mapped to error::INTERNAL.
  EXPECT_TRUE(error::DEADLINE_EXCEEDED == status.code() ||
              error::INTERNAL == status.code());
}

}  // namespace tensorflow
