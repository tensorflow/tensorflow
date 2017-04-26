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

#include "tensorflow/core/debug/debug_gateway.h"

#include <algorithm>
#include <cstdlib>
#include <unordered_map>

#include "tensorflow/core/debug/debug_graph_utils.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"

namespace tensorflow {
namespace {

DirectSession* CreateSession() {
  SessionOptions options;
  // Turn off graph optimizer so we can observe intermediate node states.
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);

  return dynamic_cast<DirectSession*>(NewSession(options));
}

class SessionDebugMinusAXTest : public ::testing::Test {
 public:
  void Initialize(std::initializer_list<float> a_values) {
    Graph graph(OpRegistry::Global());

#if GOOGLE_CUDA
    const string kDeviceName = "/job:localhost/replica:0/task:0/gpu:0";
#else
    const string kDeviceName = "/job:localhost/replica:0/task:0/cpu:0";
#endif

    Tensor a_tensor(DT_FLOAT, TensorShape({2, 2}));
    test::FillValues<float>(&a_tensor, a_values);
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name(kDeviceName);
    a_ = a->name();

    Tensor x_tensor(DT_FLOAT, TensorShape({2, 1}));
    test::FillValues<float>(&x_tensor, {1, 1});
    Node* x = test::graph::Constant(&graph, x_tensor);
    x->set_assigned_device_name(kDeviceName);
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name(kDeviceName);
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name(kDeviceName);

    test::graph::ToGraphDef(&graph, &def_);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

TEST_F(SessionDebugMinusAXTest, RunSimpleNetwork) {
  Initialize({3, 2, -1, 0});
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_nodes_w_outputs;
  std::vector<string> completed_nodes_wo_outputs;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [&mu, &completed_nodes_w_outputs, &completed_nodes_wo_outputs](
          const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output) {
          completed_nodes_w_outputs.push_back(node_name);
        } else {
          completed_nodes_wo_outputs.push_back(node_name);
        }
      });

  std::vector<bool> tensors_initialized;
  std::unordered_map<string, Tensor> tensor_vals;
  // output_slot values recorded in value callbacks
  std::vector<int> output_slots_val;
  // is_ref values recorded in value callbacks
  std::vector<bool> is_refs_val;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &tensors_initialized, &tensor_vals, &output_slots_val,
       &is_refs_val,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        tensors_initialized.push_back(tensor_value.IsInitialized());
        tensor_vals.insert(std::make_pair(node_name, tensor_value));
        output_slots_val.push_back(output_slot);
        is_refs_val.push_back(is_ref);

        // Set the notification once we have the value from the target node.
        if (node_name == y_neg_ && !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  TF_ASSERT_OK(session->Create(def_));

  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  ASSERT_EQ(1, outputs.size());
  // The first output should be initialized and have the correct
  // output.
  auto mat = outputs[0].matrix<float>();
  ASSERT_TRUE(outputs[0].IsInitialized());
  EXPECT_FLOAT_EQ(5.0, mat(0, 0));

  // Verify the calling history of the completion callback
  // The following verifies each node with output(s) invoked the callback
  // exactly once.
  ASSERT_GE(completed_nodes_w_outputs.size(), 4);  // There may be added nodes.

  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), a_));
  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), x_));
  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), y_));
  ASSERT_EQ(1, std::count(completed_nodes_w_outputs.begin(),
                          completed_nodes_w_outputs.end(), y_neg_));

  // Apart from nodes with outputs, there are also no-output (control) nodes.
  // They ought to be captured by the DebugGateway through
  // NodeOutputCallback as well.
  ASSERT_GT(completed_nodes_wo_outputs.size(), 0);

  // The DebugGateway should have captured the _SOURCE node.
  ASSERT_LE(1, std::count(completed_nodes_wo_outputs.begin(),
                          completed_nodes_wo_outputs.end(), "_SOURCE"));

  // Verify the calling history of the value callabck
  ASSERT_EQ(completed_nodes_w_outputs.size(), tensors_initialized.size());

  // In this graph, there is no uninitialized node value.
  ASSERT_EQ(
      tensors_initialized.end(),
      std::find(tensors_initialized.begin(), tensors_initialized.end(), false));

  ASSERT_EQ(completed_nodes_w_outputs.size(), tensor_vals.size());
  ASSERT_EQ(completed_nodes_w_outputs.size(), output_slots_val.size());
  ASSERT_EQ(completed_nodes_w_outputs.size(), is_refs_val.size());

  // Verify the intermediate tensor values captured through the value callback
  auto mat_a = tensor_vals[a_].matrix<float>();
  ASSERT_EQ(3.0, mat_a(0, 0));
  ASSERT_EQ(2.0, mat_a(0, 1));
  ASSERT_EQ(-1.0, mat_a(1, 0));
  ASSERT_EQ(0.0, mat_a(1, 1));

  auto mat_x = tensor_vals[x_].matrix<float>();
  ASSERT_EQ(1.0, mat_x(0, 0));
  ASSERT_EQ(1.0, mat_x(1, 0));

  auto mat_y = tensor_vals[y_].matrix<float>();
  ASSERT_EQ(5.0, mat_y(0, 0));
  ASSERT_EQ(-1.0, mat_y(1, 0));

  auto mat_y_neg = tensor_vals[y_neg_].matrix<float>();
  ASSERT_EQ(-5.0, mat_y_neg(0, 0));
  ASSERT_EQ(1.0, mat_y_neg(1, 0));

  // In this graph, all outputs are on the first slot
  ASSERT_EQ(output_slots_val.size(),
            std::count_if(output_slots_val.begin(), output_slots_val.end(),
                          [](int slot) { return slot == 0; }));

  // In this graph, there is no ref-type tensor.
  ASSERT_EQ(is_refs_val.end(),
            std::find(is_refs_val.begin(), is_refs_val.end(), true));
}

TEST_F(SessionDebugMinusAXTest, RunSimpleNetworkWithTwoDebugNodesInserted) {
  // Tensor contains one count of NaN
  Initialize({3, std::numeric_limits<float>::quiet_NaN(), -1, 0});
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  // Create debug tensor watch options with two debug ops:
  // DebugIdentity and DebugNanCount
  RunOptions run_opts;
  run_opts.set_output_partition_graphs(true);

  const string debug_identity = "DebugIdentity";
  const string debug_nan_count = "DebugNanCount";
  DebugTensorWatch* tensor_watch_opts =
      run_opts.mutable_debug_options()->add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(y_);
  tensor_watch_opts->set_output_slot(0);
  tensor_watch_opts->add_debug_ops(debug_identity);
  tensor_watch_opts->add_debug_ops(debug_nan_count);

  // Expected name of the inserted debug node
  string debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(y_, ":", 0), 0, debug_identity);
  string debug_nan_count_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(y_, ":", 0), 1, debug_nan_count);

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_debug_nodes;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [&mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &completed_debug_nodes](const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output && (node_name == debug_identity_node_name ||
                           node_name == debug_nan_count_node_name)) {
          completed_debug_nodes.push_back(node_name);
        }
      });

  std::vector<Tensor> watched_tensor_vals;
  std::vector<Tensor> debug_identity_tensor_vals;
  std::vector<Tensor> debug_nan_count_tensor_vals;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &watched_tensor_vals, &debug_identity_tensor_vals,
       &debug_nan_count_tensor_vals,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        if (node_name == y_) {
          watched_tensor_vals.push_back(tensor_value);
        } else if (node_name == debug_identity_node_name && output_slot == 0) {
          // output_slot == 0 carries the debug signal. Same below.
          debug_identity_tensor_vals.push_back(tensor_value);
        } else if (node_name == debug_nan_count_node_name && output_slot == 0) {
          debug_nan_count_tensor_vals.push_back(tensor_value);
        }

        // Set the notification once we have the value from the target node.
        if (node_name == y_neg_ && !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  TF_ASSERT_OK(session->Create(def_));

  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;

  RunMetadata run_metadata;
  Status s = session->Run(run_opts, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);

// Verify the correct number of partition graphs (GraphDefs) outputted
// through RunMetadata, given whether GPU is involved.
#if GOOGLE_CUDA
  ASSERT_EQ(2, run_metadata.partition_graphs().size());
#else
  ASSERT_EQ(1, run_metadata.partition_graphs().size());
#endif

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  // Verify that each of the two debug nodes has completed exactly once.
  ASSERT_EQ(2, completed_debug_nodes.size());
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_identity_node_name));
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_nan_count_node_name));

  // Verify that the tensor values from the watched node and the identity
  // debug node are received and they are equal (owing to the debug op being
  // "DebugIdentity")
  ASSERT_EQ(1, watched_tensor_vals.size());
  ASSERT_EQ(1, debug_identity_tensor_vals.size());
  auto mat_y = watched_tensor_vals[0].matrix<float>();
  auto mat_identity = debug_identity_tensor_vals[0].matrix<float>();
  // ASSERT_EQ doesn't work for nan == nan
  ASSERT_TRUE(std::isnan(mat_y(0, 0)));
  ASSERT_TRUE(std::isnan(mat_identity(0, 0)));
  ASSERT_EQ(-1, mat_identity(1, 0));

  // Verify that the output from the NaN-count debug node indicates exactly
  // one NaN.
  ASSERT_EQ(1, debug_nan_count_tensor_vals.size());
  ASSERT_EQ(1, debug_nan_count_tensor_vals[0].scalar<int64>()());
}

#ifndef GOOGLE_CUDA
// TODO(cais): Reinstate the following test for concurrent debugged runs on
//   a GPU once the root cause of the ~0.5% flakiness has been addressed.
//   (b/34081273)
TEST_F(SessionDebugMinusAXTest,
       RunSimpleNetworkConcurrentlyWithDifferentDebugTensorWatches) {
  // Test concurrent Run() calls on a graph with different debug watches.

  Initialize({3, 2, -1, 0});
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);
  TF_ASSERT_OK(session->Create(def_));

  // Number of concurrent Run() calls to launch.
  const int kConcurrentRuns = 3;
  thread::ThreadPool* tp =
      new thread::ThreadPool(Env::Default(), "test", kConcurrentRuns);

  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};

  mutex mu;
  DebugGateway debug_gateway(session.get());
  std::unordered_map<string, Tensor> debug_identity_tensor_vals;

  const string debug_identity = "DebugIdentity";

  const string a_debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(a_, ":", 0), 0, debug_identity);
  const string x_debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(x_, ":", 0), 0, debug_identity);
  const string y_debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(y_, ":", 0), 0, debug_identity);

  Notification callbacks_done;
  volatile int val_callback_count = 0;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &val_callback_count, &a_debug_identity_node_name,
       &x_debug_identity_node_name, &y_debug_identity_node_name,
       &debug_identity_tensor_vals, &callbacks_done, &kConcurrentRuns](
           const string& node_name, const int output_slot,
           const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);

        if (node_name == a_debug_identity_node_name && output_slot == 0) {
          debug_identity_tensor_vals["a"] = tensor_value;
          val_callback_count++;
        } else if (node_name == x_debug_identity_node_name &&
                   output_slot == 0) {
          // output_slot == 0 carries the debug signal.
          debug_identity_tensor_vals["x"] = tensor_value;
          val_callback_count++;
        } else if (node_name == y_debug_identity_node_name &&
                   output_slot == 0) {
          debug_identity_tensor_vals["y"] = tensor_value;
          val_callback_count++;
        }

        // Set the notification once we have the value from the callbacks from
        // all the concurrent Run() calls.
        if (val_callback_count == kConcurrentRuns &&
            !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  int run_counter = 0;
  mutex run_lock;

  // Function to be executed concurrently.
  auto fn = [this, &run_lock, &run_counter, &session, output_names,
             target_nodes, &debug_identity]() {
    // Create unique debug tensor watch options for each of the concurrent
    // run calls.
    RunOptions run_opts;
    run_opts.set_output_partition_graphs(true);

    DebugTensorWatch* tensor_watch_opts =
        run_opts.mutable_debug_options()->add_debug_tensor_watch_opts();
    tensor_watch_opts->set_output_slot(0);
    tensor_watch_opts->add_debug_ops(debug_identity);

    {
      // Let the concurrent runs watch different tensors.

      mutex_lock l(run_lock);

      if (run_counter == 0) {
        // Let the 1st concurrent run watch a.
        tensor_watch_opts->set_node_name(a_);
      } else if (run_counter == 1) {
        // Let the 2nd concurrent watch x.
        tensor_watch_opts->set_node_name(x_);
      } else if (run_counter == 2) {
        // Let the 3rd concurrent watch y.
        tensor_watch_opts->set_node_name(y_);
      }

      run_counter++;
    }

    // Run the graph.
    RunMetadata run_metadata;
    std::vector<std::pair<string, Tensor>> inputs;
    std::vector<Tensor> outputs;
    Status s = session->Run(run_opts, inputs, output_names, target_nodes,
                            &outputs, &run_metadata);
    TF_ASSERT_OK(s);

    ASSERT_EQ(1, run_metadata.partition_graphs().size());

    ASSERT_EQ(1, outputs.size());
    ASSERT_TRUE(outputs[0].IsInitialized());
    ASSERT_EQ(TensorShape({2, 1}), outputs[0].shape());
    auto mat = outputs[0].matrix<float>();
    EXPECT_FLOAT_EQ(5.0, mat(0, 0));
    EXPECT_FLOAT_EQ(-1.0, mat(1, 0));
  };

  for (int i = 0; i < kConcurrentRuns; ++i) {
    tp->Schedule(fn);
  }

  // Wait for the debug callbacks to finish.
  callbacks_done.WaitForNotification();

  // Wait for the concurrent functions with Run() calls to finish.
  delete tp;

  {
    mutex_lock l(mu);

    ASSERT_EQ(kConcurrentRuns, val_callback_count);
    ASSERT_EQ(kConcurrentRuns, debug_identity_tensor_vals.size());

    ASSERT_EQ(TensorShape({2, 2}), debug_identity_tensor_vals["a"].shape());
    auto a_mat_identity = debug_identity_tensor_vals["a"].matrix<float>();
    ASSERT_EQ(3.0, a_mat_identity(0, 0));
    ASSERT_EQ(2.0, a_mat_identity(0, 1));
    ASSERT_EQ(-1.0, a_mat_identity(1, 0));
    ASSERT_EQ(0.0, a_mat_identity(1, 1));

    ASSERT_EQ(TensorShape({2, 1}), debug_identity_tensor_vals["x"].shape());
    auto x_mat_identity = debug_identity_tensor_vals["x"].matrix<float>();
    ASSERT_EQ(1.0, x_mat_identity(0, 0));
    ASSERT_EQ(1.0, x_mat_identity(1, 0));

    ASSERT_EQ(TensorShape({2, 1}), debug_identity_tensor_vals["y"].shape());
    auto y_mat_identity = debug_identity_tensor_vals["y"].matrix<float>();
    ASSERT_EQ(5.0, y_mat_identity(0, 0));
    ASSERT_EQ(-1.0, y_mat_identity(1, 0));
  }
}
#endif

class SessionDebugOutputSlotWithoutOngoingEdgeTest : public ::testing::Test {
 public:
  void Initialize() {
    Graph graph(OpRegistry::Global());

#if GOOGLE_CUDA
    const string kDeviceName = "/job:localhost/replica:0/task:0/gpu:0";
#else
    const string kDeviceName = "/job:localhost/replica:0/task:0/cpu:0";
#endif

    Tensor a_tensor(DT_FLOAT, TensorShape({1, 1}));
    test::FillValues<float>(&a_tensor, {42.0});
    Node* a = test::graph::Constant(&graph, a_tensor);
    a->set_assigned_device_name(kDeviceName);

    Node* c = test::graph::Constant(&graph, a_tensor);
    c->set_assigned_device_name(kDeviceName);
    c_ = c->name();

    // Node c will be executed only because of the control edge from c to y.
    // Its output slot (slot 0) does not have an outgoing edge. This test
    // is for testing that the debugger can watch that slot properly.
    Node* y = test::graph::NoOp(&graph, {c});
    y->set_assigned_device_name(kDeviceName);
    y_ = y->name();

    test::graph::ToGraphDef(&graph, &def_);
  }

  string c_;
  string y_;
  GraphDef def_;
};

TEST_F(SessionDebugOutputSlotWithoutOngoingEdgeTest,
       WatchSlotWithoutOutgoingEdge) {
  Initialize();
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  // Supply completion and value callbacks
  mutex mu;

  string debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(c_, ":", 0), 0, "DebugIdentity");

  Notification callbacks_done;

  std::vector<Tensor> debug_identity_tensor_vals;
  debug_gateway.SetNodeValueCallback([this, &mu, &callbacks_done,
                                      &debug_identity_node_name,
                                      &debug_identity_tensor_vals](
      const string& node_name, const int output_slot,
      const Tensor& tensor_value, const bool is_ref) {
    mutex_lock l(mu);

    if (node_name == debug_identity_node_name && output_slot == 0) {
      debug_identity_tensor_vals.push_back(tensor_value);

      if (!callbacks_done.HasBeenNotified()) {
        callbacks_done.Notify();
      }
    }
  });

  // Add DebugIdentity watch on c:0, which does not have an outgoing edge.
  RunOptions run_opts;
  run_opts.set_output_partition_graphs(true);

  DebugTensorWatch* tensor_watch_opts =
      run_opts.mutable_debug_options()->add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(c_);
  tensor_watch_opts->set_output_slot(0);
  tensor_watch_opts->add_debug_ops("DebugIdentity");

  TF_ASSERT_OK(session->Create(def_));

  // Invoke Session::Run() on y.
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_names;
  std::vector<string> target_nodes = {y_};
  std::vector<Tensor> outputs;

  RunMetadata run_metadata;
  Status s = session->Run(run_opts, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  // Assert that DebugIdentity node watching the control edge has been run.
  ASSERT_EQ(1, debug_identity_tensor_vals.size());
  auto mat_identity = debug_identity_tensor_vals[0].matrix<float>();
  ASSERT_EQ(42.0, mat_identity(0, 0));
}

class SessionDebugVariableTest : public ::testing::Test {
 public:
  void Initialize() {
    Graph graph(OpRegistry::Global());

#if GOOGLE_CUDA
    const string kDeviceName = "/job:localhost/replica:0/task:0/gpu:0";
#else
    const string kDeviceName = "/job:localhost/replica:0/task:0/cpu:0";
#endif

    // Define variable node.
    var_node_name_ = "var";
    Node* var =
        test::graph::Var(&graph, DT_FLOAT, TensorShape({3}), var_node_name_);
    var->set_assigned_device_name(kDeviceName);

    // Define the initial value and the initial-value node.
    Tensor nan_nan_seven(DT_FLOAT, TensorShape({3}));
    nan_nan_seven.flat<float>()(0) = std::numeric_limits<float>::quiet_NaN();
    nan_nan_seven.flat<float>()(1) = std::numeric_limits<float>::quiet_NaN();
    nan_nan_seven.flat<float>()(2) = 7.0;

    init_val_node_name_ = "init_val";
    Node* init_val =
        test::graph::Constant(&graph, nan_nan_seven, init_val_node_name_);
    init_val->set_assigned_device_name(kDeviceName);

    // Define node for variable value initialization
    Node* init = test::graph::Assign(&graph, var, init_val);
    init->set_assigned_device_name(kDeviceName);
    init_node_name_ = init->name();

    // Define new value node
    Tensor nan_eight_eight(DT_FLOAT, TensorShape({3}));
    nan_eight_eight.flat<float>()(0) = std::numeric_limits<float>::quiet_NaN();
    nan_eight_eight.flat<float>()(1) = 8.0;
    nan_eight_eight.flat<float>()(2) = 8.0;

    Node* new_val = test::graph::Constant(&graph, nan_eight_eight);
    new_val->set_assigned_device_name(kDeviceName);
    new_val_node_name_ = new_val->name();

    // Define node for assigning new value
    Node* assign = test::graph::Assign(&graph, var, new_val);
    assign->set_assigned_device_name(kDeviceName);
    assign_node_name_ = assign->name();

    test::graph::ToGraphDef(&graph, &def_);
  }

  string var_node_name_;
  string init_val_node_name_;
  string init_node_name_;
  string new_val_node_name_;
  string assign_node_name_;
  GraphDef def_;
};

TEST_F(SessionDebugVariableTest, WatchUninitializedVariableWithDebugOps) {
  Initialize();
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  TF_ASSERT_OK(session->Create(def_));

  // Set up DebugTensorWatch for an uninitialized tensor (in node var).
  RunOptions run_opts;
  const string debug_identity = "DebugIdentity";
  DebugTensorWatch* tensor_watch_opts =
      run_opts.mutable_debug_options()->add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(var_node_name_);
  tensor_watch_opts->set_output_slot(0);
  tensor_watch_opts->add_debug_ops(debug_identity);

  // Expected name of the inserted debug node
  string debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(var_node_name_, ":", 0), 0, debug_identity);

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_debug_nodes;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [this, &mu, &debug_identity_node_name, &completed_debug_nodes,
       &callbacks_done](const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output && (node_name == debug_identity_node_name)) {
          completed_debug_nodes.push_back(node_name);
        }
      });

  std::vector<Tensor> debug_identity_tensor_vals;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &debug_identity_node_name, &debug_identity_tensor_vals,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        if (node_name == debug_identity_node_name && output_slot == 0) {
          // output_slot == 0 carries the debug signal. Same below.
          debug_identity_tensor_vals.push_back(tensor_value);
        }

        // Set the notification once we have the value from the target node.
        if (node_name == init_node_name_ && !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  // First run the initialization op
  std::vector<std::pair<string, Tensor>> inputs_init;
  std::vector<Tensor> outputs_init;

  RunMetadata run_metadata;
  Status s = session->Run(run_opts, inputs_init, {init_node_name_}, {},
                          &outputs_init, &run_metadata);
  TF_ASSERT_OK(s);

  callbacks_done.WaitForNotification();

  ASSERT_EQ(1, completed_debug_nodes.size());
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_identity_node_name));

  // Assert the output reflects the uninitialized nature of var's tensor.
  ASSERT_EQ(1, debug_identity_tensor_vals.size());
  ASSERT_FALSE(debug_identity_tensor_vals[0].IsInitialized());
  ASSERT_EQ(DT_FLOAT, debug_identity_tensor_vals[0].dtype());
  ASSERT_EQ(TensorShape({3}), debug_identity_tensor_vals[0].shape());
}

TEST_F(SessionDebugVariableTest, VariableAssignWithDebugOps) {
  // Tensor contains one count of NaN
  Initialize();
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  TF_ASSERT_OK(session->Create(def_));

  // First run the initialization op
  std::vector<std::pair<string, Tensor>> inputs_init;
  std::vector<Tensor> outputs_init;
  Status s = session->Run(inputs_init, {init_node_name_}, {}, &outputs_init);
  TF_ASSERT_OK(s);

  // Create debug tensor watch options with two ref-type debug ops:
  // DebugIdentity and DebugNanCount
  RunOptions run_opts;
  run_opts.set_output_partition_graphs(true);
  const string debug_identity = "DebugIdentity";
  const string debug_nan_count = "DebugNanCount";
  DebugTensorWatch* tensor_watch_opts =
      run_opts.mutable_debug_options()->add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(var_node_name_);
  tensor_watch_opts->set_output_slot(0);
  tensor_watch_opts->add_debug_ops(debug_identity);
  tensor_watch_opts->add_debug_ops(debug_nan_count);

  char tempdir_template[] = "/tmp/tfdbg_XXXXXX";
  string temp_dir(mkdtemp(tempdir_template));
  tensor_watch_opts->add_debug_urls(strings::StrCat("file://", temp_dir));

  // Expected name of the inserted debug node
  string debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(var_node_name_, ":", 0), 0, debug_identity);
  string debug_nan_count_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(var_node_name_, ":", 0), 1, debug_nan_count);

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_debug_nodes;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [this, &mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &completed_debug_nodes,
       &callbacks_done](const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output && (node_name == debug_identity_node_name ||
                           node_name == debug_nan_count_node_name)) {
          completed_debug_nodes.push_back(node_name);
        }
      });

  std::vector<Tensor> debug_identity_tensor_vals;
  std::vector<Tensor> debug_nan_count_tensor_vals;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &debug_identity_node_name, &debug_nan_count_node_name,
       &debug_identity_tensor_vals, &debug_nan_count_tensor_vals,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        if (node_name == debug_identity_node_name && output_slot == 0) {
          // output_slot == 0 carries the debug signal. Same below.
          debug_identity_tensor_vals.push_back(tensor_value);
        } else if (node_name == debug_nan_count_node_name && output_slot == 0) {
          debug_nan_count_tensor_vals.push_back(tensor_value);
        }

        // Set the notification once we have the value from the target node.
        if (node_name == assign_node_name_ &&
            !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  // // Request two targets: one fetch output and one non-fetched output.
  std::vector<std::pair<string, Tensor>> inputs;
  std::vector<string> output_names = {assign_node_name_ + ":0"};
  std::vector<string> target_nodes = {assign_node_name_};
  std::vector<Tensor> outputs;

  // Run with RunOptions that has tensor watches
  RunMetadata run_metadata;
  s = session->Run(run_opts, inputs, output_names, target_nodes, &outputs,
                   &run_metadata);
  TF_ASSERT_OK(s);

#if GOOGLE_CUDA
  ASSERT_EQ(2, run_metadata.partition_graphs().size());
#else
  ASSERT_EQ(1, run_metadata.partition_graphs().size());
#endif

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  // Verify that the update has happened properly.
  ASSERT_EQ(1, outputs.size());
  ASSERT_TRUE(std::isnan(outputs[0].vec<float>()(0)));
  ASSERT_EQ(8.0, outputs[0].vec<float>()(1));  // Expect new value
  ASSERT_EQ(8.0, outputs[0].vec<float>()(2));  // Expect new value

  // Verify that each of the two debug nodes has completed exactly once.
  ASSERT_EQ(2, completed_debug_nodes.size());
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_identity_node_name));
  ASSERT_EQ(
      1, std::count(completed_debug_nodes.begin(), completed_debug_nodes.end(),
                    debug_nan_count_node_name));

  // Verify that the values from the ref identity node reflects the value
  // before the new assign.
  ASSERT_EQ(1, debug_identity_tensor_vals.size());

  auto vec_identity = debug_identity_tensor_vals[0].vec<float>();
  ASSERT_TRUE(std::isnan(vec_identity(0)));
  ASSERT_TRUE(std::isnan(vec_identity(1)));
  ASSERT_EQ(7.0, vec_identity(2));

  // Verify that the output from the NaN-count debug node indicates exactly
  // two NaNs, i.e., reflecting the value before the new assign.
  ASSERT_EQ(1, debug_nan_count_tensor_vals.size());
  ASSERT_EQ(2, debug_nan_count_tensor_vals[0].scalar<int64>()());
}

#if GOOGLE_CUDA
class SessionDebugGPUSwitchTest : public ::testing::Test {
 public:
  void Initialize() {
    Graph graph(OpRegistry::Global());

    const string kDeviceName = "/job:localhost/replica:0/task:0/gpu:0";

    Tensor vb(DT_BOOL, TensorShape({}));
    vb.scalar<bool>()() = true;
    Tensor vi(DT_INT64, TensorShape({}));
    vi.scalar<int>()() = 42;
    // So vi is expected to be forwarded to the second output port of sw.

    Node* pred = test::graph::Constant(&graph, vb);
    pred->set_assigned_device_name(kDeviceName);
    pred_node_name_ = pred->name();

    Node* value = test::graph::Constant(&graph, vi);
    pred->set_assigned_device_name(kDeviceName);
    value_node_name_ = value->name();

    Node* sw = test::graph::Switch(&graph, value, pred);
    sw->set_assigned_device_name(kDeviceName);
    sw_node_name_ = sw->name();

    Node* z = test::graph::Identity(&graph, sw, 1);
    sw->set_assigned_device_name(kDeviceName);
    z_node_name_ = z->name();

    test::graph::ToGraphDef(&graph, &def_);
  }

  string pred_node_name_;
  string value_node_name_;
  string sw_node_name_;
  string z_node_name_;
  GraphDef def_;
};

// Test for debug-watching tensors marked as HOST_MEMORY on GPU.
TEST_F(SessionDebugGPUSwitchTest, RunSwitchWithHostMemoryDebugOp) {
  Initialize();
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  RunOptions run_opts;
  run_opts.set_output_partition_graphs(true);
  // This is the name of the boolean tensor fed as pred to the Switch node.
  // On GPU, this edge is HOST_MEMORY.
  const string watched_tensor = strings::StrCat(pred_node_name_, "/_1");

  const string debug_identity = "DebugIdentity";
  DebugTensorWatch* tensor_watch_opts =
      run_opts.mutable_debug_options()->add_debug_tensor_watch_opts();
  tensor_watch_opts->set_node_name(watched_tensor);
  tensor_watch_opts->set_output_slot(0);
  tensor_watch_opts->add_debug_ops(debug_identity);

  // Expected name of the inserted debug node
  string debug_identity_node_name = DebugNodeInserter::GetDebugNodeName(
      strings::StrCat(watched_tensor, ":", 0), 0, debug_identity);

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_nodes_w_outputs;
  std::vector<string> completed_nodes_wo_outputs;

  Notification callbacks_done;
  debug_gateway.SetNodeCompletionCallback(
      [&mu, &completed_nodes_w_outputs, &completed_nodes_wo_outputs](
          const string& node_name, const bool any_output) {
        mutex_lock l(mu);
        if (any_output) {
          completed_nodes_w_outputs.push_back(node_name);
        } else {
          completed_nodes_wo_outputs.push_back(node_name);
        }
      });

  std::vector<Tensor> debug_identity_tensor_vals;

  debug_gateway.SetNodeValueCallback(
      [this, &mu, &debug_identity_node_name, &debug_identity_tensor_vals,
       &callbacks_done](const string& node_name, const int output_slot,
                        const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        if (node_name == debug_identity_node_name && output_slot == 0) {
          debug_identity_tensor_vals.push_back(tensor_value);
        }

        // Set the notification once we have the value from the target node.
        if (node_name == z_node_name_ && !callbacks_done.HasBeenNotified()) {
          callbacks_done.Notify();
        }
      });

  TF_ASSERT_OK(session->Create(def_));

  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {z_node_name_ + ":0"};
  std::vector<string> target_nodes = {z_node_name_};
  std::vector<Tensor> outputs;

  RunMetadata run_metadata;
  Status s = session->Run(run_opts, inputs, output_names, target_nodes,
                          &outputs, &run_metadata);
  TF_ASSERT_OK(s);

  ASSERT_EQ(2, run_metadata.partition_graphs().size());

  // Wait for callbacks to complete.
  callbacks_done.WaitForNotification();

  ASSERT_EQ(1, debug_identity_tensor_vals.size());
  ASSERT_TRUE(debug_identity_tensor_vals[0].scalar<bool>()());
}
#endif  // GOOGLE_CUDA

}  // end namespace
}  // end namespace tensorflow
