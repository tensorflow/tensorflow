#include "tensorflow/core/common_runtime/direct_session.h"

#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/public/status.h"
#include "tensorflow/core/public/tensor.h"
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
    RequireDefaultOps();
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
  ASSERT_OK(session->Create(def_));
  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  ASSERT_OK(s);

  ASSERT_EQ(1, outputs.size());
  // The first output should be initiailzed and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));
}

TEST_F(DirectSessionMinusAXTest, TestFeed) {
  Initialize({1, 2, 3, 4});
  std::unique_ptr<Session> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  ASSERT_OK(session->Create(def_));

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
  ASSERT_OK(s);

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
  ASSERT_OK(session->Create(def_));

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
      ASSERT_TRUE(s.ok());
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
  ASSERT_OK(session->Create(def_));

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
  ASSERT_OK(session->Create(def));
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
  ASSERT_OK(session->Create(def));
  ASSERT_OK(session->Run(inputs, output_names, {}, &outputs));
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
  ASSERT_OK(session->Create(def));

  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<Tensor> outputs;

  // Initialize the variable
  Status s = session->Run(inputs, {init->name()}, {}, &outputs);
  ASSERT_OK(s);

  // Get the variable's data
  s = session->Run(inputs, {var->name() + ":0"}, {}, &outputs);
  ASSERT_OK(s);
  ASSERT_EQ(1, outputs.size());
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_EQ(20.0, outputs[0].flat<float>()(0));
}

TEST(DirectSessionTest, MultipleFeedTest) {
  GraphDef def;
  Graph g(OpRegistry::Global());
  Node* var = test::graph::Var(&g, DT_FLOAT, TensorShape({10}));
  var->set_assigned_device_name("/job:localhost/replica:0/task:0/cpu:0");

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
  ASSERT_OK(session->Create(def));

  std::vector<Tensor> outputs;

  // Fetch without feeding.
  Status s = session->Run(
      {}, {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(1.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(2.0, outputs[1].flat<float>()(0));

  s = session->Run(
      {}, {second_identity->name() + ":0", first_identity->name() + ":0"}, {},
      &outputs);
  ASSERT_TRUE(s.ok());
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
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));

  // Feed [second_const, first_const]
  s = session->Run(
      {{second_const->name(), value_22}, {first_const->name(), value_11}},
      {first_identity->name() + ":0", second_identity->name() + ":0"}, {},
      &outputs);
  ASSERT_TRUE(s.ok());
  ASSERT_EQ(2, outputs.size());
  ASSERT_EQ(11.0, outputs[0].flat<float>()(0));
  ASSERT_EQ(22.0, outputs[1].flat<float>()(0));
}

}  // namespace

}  // namespace tensorflow
