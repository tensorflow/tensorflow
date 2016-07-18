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

#if GOOGLE_CUDA

#include "tensorflow/core/debug/debug_gateway.h"

#include <algorithm>
#include <unordered_map>

#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {
namespace {

DirectSession* CreateSession() {
  SessionOptions options;
  // Turn off graph optimizer so we can observe intermediate node states.
  options.config.mutable_graph_options()
      ->mutable_optimizer_options()
      ->set_opt_level(OptimizerOptions_Level_L0);

  (*options.config.mutable_device_count())["CPU"] = 1;
  (*options.config.mutable_device_count())["GPU"] = 1;
  options.config.set_allow_soft_placement(true);

  return dynamic_cast<DirectSession*>(NewSession(options));
}

class SessionDebugGPUMinusAXTest : public ::testing::Test {
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
    x->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");
    x_ = x->name();

    // y = A * x
    Node* y = test::graph::Matmul(&graph, a, x, false, false);
    y->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");
    y_ = y->name();

    Node* y_neg = test::graph::Unary(&graph, "Neg", y);
    y_neg_ = y_neg->name();
    y_neg->set_assigned_device_name("/job:localhost/replica:0/task:0/gpu:0");

    test::graph::ToGraphDef(&graph, &def_);
  }

  string a_;
  string x_;
  string y_;
  string y_neg_;
  GraphDef def_;
};

TEST_F(SessionDebugGPUMinusAXTest, RunSimpleNetwork) {
  Initialize({3, 2, -1, 0});
  std::unique_ptr<DirectSession> session(CreateSession());
  ASSERT_TRUE(session != nullptr);

  DebugGateway debug_gateway(session.get());

  // Supply completion and value callbacks
  mutex mu;
  // Completed nodes with and without outputs
  std::vector<string> completed_nodes_w_outputs;
  std::vector<string> completed_nodes_wo_outputs;

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
      [&mu, &tensors_initialized, &tensor_vals, &output_slots_val,
       &is_refs_val](const string& node_name, const int output_slot,
                     const Tensor& tensor_value, const bool is_ref) {
        mutex_lock l(mu);
        tensors_initialized.push_back(tensor_value.IsInitialized());
        tensor_vals.insert(std::make_pair(node_name, tensor_value));
        output_slots_val.push_back(output_slot);
        is_refs_val.push_back(is_ref);
      });

  TF_ASSERT_OK(session->Create(def_));

  std::vector<std::pair<string, Tensor>> inputs;

  // Request two targets: one fetch output and one non-fetched output.
  std::vector<string> output_names = {y_ + ":0"};
  std::vector<string> target_nodes = {y_neg_};
  std::vector<Tensor> outputs;
  Status s = session->Run(inputs, output_names, target_nodes, &outputs);
  TF_ASSERT_OK(s);

  Env::Default()->SleepForMicroseconds(10 * 1000);

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

  // The DebugGateway should have captured the _SOURCE and _SINK nodes.
  ASSERT_LE(1, std::count(completed_nodes_wo_outputs.begin(),
                          completed_nodes_wo_outputs.end(), "_SOURCE"));
  ASSERT_LE(1, std::count(completed_nodes_wo_outputs.begin(),
                          completed_nodes_wo_outputs.end(), "_SINK"));

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

}  // end namespace
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
