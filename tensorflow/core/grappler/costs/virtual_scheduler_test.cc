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

#include "tensorflow/core/grappler/costs/virtual_scheduler.h"

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/costs/virtual_placer.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
// Class for testing virtual scheduler.
class TestVirtualScheduler : public VirtualScheduler {
 public:
  TestVirtualScheduler(const GrapplerItem* grappler_item,
                       const bool use_static_shapes, Cluster* cluster)
      : VirtualScheduler(grappler_item, use_static_shapes, cluster) {}

  FRIEND_TEST(VirtualSchedulerTest, CalculateOutputSize);
  FRIEND_TEST(VirtualSchedulerTest, MemoryUsage);
  FRIEND_TEST(VirtualSchedulerTest, ControlDependency);
  FRIEND_TEST(VirtualSchedulerTest, ComplexDependency);
  FRIEND_TEST(VirtualSchedulerTest, Variable);
};

class VirtualSchedulerTest : public ::testing::Test {
 protected:
  const string kCPU0 = "/job:localhost/replica:0/task:0/cpu:0";

  void SetUp() override {
    // Initializes cluster_ and placer_.
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    devices[kCPU0] = cpu_device;

    cluster_.reset(new VirtualCluster(devices));
    placer_.reset(new VirtualPlacer(cluster_.get()));
  }

  // Three Conv2Ds with only two in fetch nodes.
  void CreateGrapplerItemWithConv2Ds() {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(kCPU0);
    auto x = tensorflow::ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto y = tensorflow::ops::RandomUniform(
        s.WithOpName("y"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto z = tensorflow::ops::RandomUniform(
        s.WithOpName("z"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto f = tensorflow::ops::RandomUniform(
        s.WithOpName("f"), {kernel_, kernel_, depth_in_, depth_out_}, DT_FLOAT);
    std::vector<int> strides = {1, 1, 1, 1};
    auto c0 =
        tensorflow::ops::Conv2D(s.WithOpName("c0"), x, f, strides, "SAME");
    auto c1 =
        tensorflow::ops::Conv2D(s.WithOpName("c1"), y, f, strides, "SAME");
    auto c2 =
        tensorflow::ops::Conv2D(s.WithOpName("c2"), z, f, strides, "SAME");
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_conv2d_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"c0", "c1"};

    dependency_["c0"] = {"x", "f"};
    dependency_["c1"] = {"y", "f"};
  }

  // A Conv2D with a variable.
  void CreateGrapplerItemWithConv2DAndVariable() {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(kCPU0);
    auto x = tensorflow::ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto f = tensorflow::ops::Variable(
        s.WithOpName("f"), {kernel_, kernel_, depth_in_, depth_out_}, DT_FLOAT);
    std::vector<int> strides = {1, 1, 1, 1};
    auto y = tensorflow::ops::Conv2D(s.WithOpName("y"), x, f, strides, "SAME");
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_conv2d_var_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"y"};

    dependency_["y"] = {"x", "f"};
  }

  // AddN that takes 4 tensors with 10x10x10x10.
  void CreateGrapplerItemWithAddN() {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(kCPU0);
    auto x = tensorflow::ops::RandomUniform(s.WithOpName("x"), {10, 10, 10, 10},
                                            DT_FLOAT);
    auto y = tensorflow::ops::RandomUniform(s.WithOpName("y"), {10, 10, 10, 10},
                                            DT_FLOAT);
    auto z = tensorflow::ops::RandomUniform(s.WithOpName("z"), {10, 10, 10, 10},
                                            DT_FLOAT);
    auto w = tensorflow::ops::RandomUniform(s.WithOpName("w"), {10, 10, 10, 10},
                                            DT_FLOAT);
    tensorflow::OutputList input_tensors = {x, y, z, w};
    auto out = tensorflow::ops::AddN(s.WithOpName("out"), input_tensors);
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_addn_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"out"};

    dependency_["out"] = {"x", "y", "z", "w"};
  }

  // NoOp that takes 7 NoOps as control dependency.
  void CreateGrapplerItemWithControlDependency() {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(kCPU0);
    std::vector<string> input_noop_names = {"x", "y", "z", "w", "u", "v", "t"};
    std::vector<tensorflow::Operation> input_tensors;
    for (const auto& input : input_noop_names) {
      auto x = tensorflow::ops::NoOp(s.WithOpName(input));
      input_tensors.push_back(x.operation);
    }
    auto out = tensorflow::ops::NoOp(
        s.WithControlDependencies(input_tensors).WithOpName("out"));
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_control_dependency_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"out"};

    dependency_["out"] = input_noop_names;
  }

  // FusedBN [an op with multiple outputs] with multiple consumers (including
  // control dependency).
  void CreateGrapplerItemWithBatchNorm() {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope().WithDevice(kCPU0);
    auto x = tensorflow::ops::RandomUniform(
        s.WithOpName("x"), {batch_size_, width_, height_, depth_in_}, DT_FLOAT);
    auto scale = tensorflow::ops::RandomUniform(s.WithOpName("scale"),
                                                {depth_in_}, DT_FLOAT);
    auto offset = tensorflow::ops::RandomUniform(s.WithOpName("offset"),
                                                 {depth_in_}, DT_FLOAT);
    auto mean =
        tensorflow::ops::RandomUniform(s.WithOpName("mean"), {0}, DT_FLOAT);
    auto var =
        tensorflow::ops::RandomUniform(s.WithOpName("var"), {0}, DT_FLOAT);

    auto batch_norm = tensorflow::ops::FusedBatchNorm(
        s.WithOpName("bn"), x, scale, offset, mean, var,
        ops::FusedBatchNorm::IsTraining(true).Epsilon(0.1f));
    auto y = batch_norm.y;
    auto batch_mean = batch_norm.batch_mean;
    auto batch_var = batch_norm.batch_variance;

    auto z1 = tensorflow::ops::Add(s.WithOpName("z1"), x, y);
    auto z2 = tensorflow::ops::Add(s.WithOpName("z2"), batch_var, batch_var);
    auto z3 = tensorflow::ops::Add(s.WithOpName("z3"), batch_var, batch_var);
    std::vector<tensorflow::Operation> input_tensors = {
        batch_mean.op(), z1.z.op(), z2.z.op(), z3.z.op(),
    };
    auto z4 = tensorflow::ops::NoOp(
        s.WithControlDependencies(batch_var).WithOpName("z4"));

    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_complex_dependency_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"z1", "z2", "z3", "z4"};

    dependency_["bn"] = {"x", "scale", "offset", "mean", "var"};
    dependency_["z1"] = {"x", "bn"};
    dependency_["z2"] = {"bn"};
    dependency_["z3"] = {"bn"};
    dependency_["z4"] = {"bn"};
  }

  // Call this after creating grappler_item_ and setting up dependency_.
  void InitScheduler() {
    scheduler_.reset(new TestVirtualScheduler(
        grappler_item_.get(), true /* use_static_shapes */, cluster_.get()));
    TF_CHECK_OK(scheduler_->Init());
  }

  // Call this after init scheduler_. Scheduler stops after executing
  // target_node.
  std::unordered_map<string, NodeInfo> RunScheduler(const string& target_node) {
    Costs zero_costs = Costs::ZeroCosts();
    std::unordered_map<string, NodeInfo> ops_executed;
    bool more_nodes = true;
    do {
      NodeInfo node_info = scheduler_->GetCurrNodeInfo();
      ops_executed[node_info.name] = node_info;

      // Check scheduling order.
      auto it = dependency_.find(node_info.name);
      if (it != dependency_.end()) {
        for (const auto& preceding_node : it->second) {
          EXPECT_GT(ops_executed.count(preceding_node), 0);
        }
      }
      more_nodes = scheduler_->MarkCurrNodeExecuted(zero_costs);

      if (node_info.name == target_node) {
        // Scheduler has the state after executing the target node.
        break;
      }
    } while (more_nodes);
    return ops_executed;
  }

  // Helper method for validating a vector.
  template <typename T>
  void ExpectVectorEq(const std::vector<T>& expected,
                      const std::vector<T>& test_elements) {
    // Set of expected elements for an easy comparison.
    std::set<T> expected_set(expected.begin(), expected.end());
    for (const auto& element : test_elements) {
      EXPECT_GT(expected_set.count(element), 0);
    }
    EXPECT_EQ(expected.size(), test_elements.size());
  }

  // Helper method that checks the name of nodes.
  void ValidateNodeDefs(const std::vector<string>& expected,
                        const std::vector<const NodeDef*>& node_defs) {
    std::vector<string> node_names;
    std::transform(node_defs.begin(), node_defs.end(),
                   std::back_inserter(node_names),
                   [](const NodeDef* node) { return node->name(); });
    ExpectVectorEq(expected, node_names);
  }

  // Helper method for validating a set.
  template <typename T>
  void ExpectSetEq(const std::set<T>& expected,
                   const std::set<T>& test_elements) {
    for (const auto& element : test_elements) {
      EXPECT_GT(expected.count(element), 0);
    }
    EXPECT_EQ(expected.size(), test_elements.size());
  }

  // Helper method tthat checks name - port pairs.
  void ValidateMemoryUsageSnapshot(
      const std::vector<string>& expected_names, const int port_num_expected,
      const std::set<std::pair<const NodeDef*, int>>& mem_usage_snapshot) {
    std::set<std::pair<string, int>> nodes_at_peak_mem_usage;
    std::transform(
        mem_usage_snapshot.begin(), mem_usage_snapshot.end(),
        std::inserter(nodes_at_peak_mem_usage, nodes_at_peak_mem_usage.begin()),
        [](const std::pair<const NodeDef*, int>& node_port) {
          return std::make_pair(node_port.first->name(), node_port.second);
        });
    std::set<std::pair<string, int>> expected;
    std::transform(expected_names.begin(), expected_names.end(),
                   std::inserter(expected, expected.begin()),
                   [port_num_expected](const string& name) {
                     return std::make_pair(name, port_num_expected);
                   });
    ExpectSetEq(expected, nodes_at_peak_mem_usage);
  }

  // Helper method for converting shape vector to TensorProperty.
  OpInfo::TensorProperties ShapeToTensorProperty(
      const std::vector<int> shape, const DataType& data_type) const {
    OpInfo::TensorProperties tensor_property;
    tensor_property.set_dtype(data_type);
    for (const auto& x : shape) {
      tensor_property.mutable_shape()->add_dim()->set_size(x);
    }
    return tensor_property;
  }

  // SetUp() inits cluster_ and placer_.
  std::unique_ptr<VirtualCluster> cluster_;
  std::unique_ptr<VirtualPlacer> placer_;

  // grappler_item_ and scheduler_ will be initialized differently for each test
  // case.
  std::unique_ptr<GrapplerItem> grappler_item_;
  std::unique_ptr<TestVirtualScheduler> scheduler_;
  // Node name -> its preceding nodes map for testing scheduling order.
  std::unordered_map<string, std::vector<string>> dependency_;

  // Shared params for Conv2D related graphs:
  const int batch_size_ = 4;
  const int width_ = 10;
  const int height_ = 10;
  const int depth_in_ = 8;
  const int kernel_ = 3;
  const int depth_out_ = 16;
};

TEST_F(VirtualSchedulerTest, InitAndBasicScheduling) {
  // Init.
  CreateGrapplerItemWithConv2Ds();
  InitScheduler();

  // Run the scheduler.
  auto ops_executed = RunScheduler("");  // Run all the nodes.

  // [const and rand] * (x, y, f), and c0 and c1. c2 and z shouldn't be
  // executed.
  EXPECT_EQ(8, ops_executed.size());

  // x, y, f, c0, and c1 should be in the ops executed.
  EXPECT_GT(ops_executed.count("x"), 0);
  EXPECT_GT(ops_executed.count("y"), 0);
  EXPECT_GT(ops_executed.count("f"), 0);
  EXPECT_GT(ops_executed.count("c0"), 0);
  EXPECT_GT(ops_executed.count("c1"), 0);

  // z and c2 shouldn't be part of it.
  EXPECT_EQ(ops_executed.count("z"), 0);
  EXPECT_EQ(ops_executed.count("c2"), 0);

  // Check input / output properties.
  EXPECT_EQ(1, ops_executed["x"].op_info.outputs_size());
  EXPECT_EQ(1, ops_executed["y"].op_info.outputs_size());
  EXPECT_EQ(1, ops_executed["f"].op_info.outputs_size());
  EXPECT_EQ(2, ops_executed["c0"].op_info.inputs_size());
  EXPECT_EQ(2, ops_executed["c1"].op_info.inputs_size());
}

TEST_F(VirtualSchedulerTest, CalculateOutputSize) {
  // Init.
  CreateGrapplerItemWithAddN();
  InitScheduler();

  // Create a set of tensor properties.
  std::vector<OpInfo::TensorProperties> output;
  output.push_back(ShapeToTensorProperty({4, 4}, DT_FLOAT));           // 0
  output.push_back(ShapeToTensorProperty({1}, DT_FLOAT));              // 1
  output.push_back(ShapeToTensorProperty({10, 10, 10}, DT_HALF));      // 2
  output.push_back(ShapeToTensorProperty({100, 7, 8, 99}, DT_FLOAT));  // 3
  output.push_back(ShapeToTensorProperty({-1, 7, 8, 99}, DT_FLOAT));   // 4
  output.push_back(ShapeToTensorProperty({-1, 7, -1, 99}, DT_FLOAT));  // 4

  // port_num -1 is for control dependency: hard coded 4B.
  EXPECT_EQ(4, scheduler_->CalculateOutputSize(output, -1));

  // Test valid outputs.
  EXPECT_EQ(4 * 4 * 4, scheduler_->CalculateOutputSize(output, 0));
  EXPECT_EQ(4 * 1, scheduler_->CalculateOutputSize(output, 1));
  EXPECT_EQ(2 * 10 * 10 * 10, scheduler_->CalculateOutputSize(output, 2));
  EXPECT_EQ(4 * 100 * 7 * 8 * 99, scheduler_->CalculateOutputSize(output, 3));

  // Any uknown shape (-1) shall yield zero output size.
  EXPECT_EQ(0, scheduler_->CalculateOutputSize(output, 4));
  EXPECT_EQ(0, scheduler_->CalculateOutputSize(output, 5));

  // Invalid port_num (though it may be an error) shall yield zero
  // output size.
  EXPECT_EQ(0, scheduler_->CalculateOutputSize(output, 6));
}

TEST_F(VirtualSchedulerTest, MemoryUsage) {
  // Init.
  CreateGrapplerItemWithAddN();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  const auto& device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states.at(kCPU0);

  // out node adds 4 tensors, each with 10x10x10x10, so the peak memory usage
  // is 4 x the input tensor size while executing the out node.
  int64 one_input_node_size = 4 * 10 * 10 * 10 * 10;
  const std::vector<string> expected_names = {"x", "y", "z", "w"};
  EXPECT_EQ(expected_names.size() * one_input_node_size,
            cpu_state.max_memory_usage);
  ValidateMemoryUsageSnapshot(expected_names, 0 /* port_num_expected */,
                              cpu_state.mem_usage_snapshot_at_peak);
}

TEST_F(VirtualSchedulerTest, ControlDependency) {
  // Init.
  CreateGrapplerItemWithControlDependency();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  const auto& device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states.at(kCPU0);

  // The graph has a NoOp that takes control dependency from 7 NoOps. The peak
  // memory usage is when executing the final NoOp.
  int64 one_input_node_size = 4;  // control dependency
  const std::vector<string> expected_names = {"x", "y", "z", "w",
                                              "u", "v", "t"};
  EXPECT_EQ(expected_names.size() * one_input_node_size,
            cpu_state.max_memory_usage);
  ValidateMemoryUsageSnapshot(expected_names, -1 /* port_num_expected */,
                              cpu_state.mem_usage_snapshot_at_peak);
}

TEST_F(VirtualSchedulerTest, ComplexDependency) {
  // Init.
  CreateGrapplerItemWithBatchNorm();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("bn");

  const auto& device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states.at(kCPU0);

  // The graph is
  //  bn = FusedBatchNorm(x, scale, offset, mean, var)
  //  z1 = bn.y + x
  //  z2 = bn.var + bn.var
  //  z3 = bn.var + bn.var
  //  z4 = control dependency from bn.
  //  Note that bn.mean doesn't have any consumer.
  const int x_size = batch_size_ * width_ * height_ * depth_in_;
  int64 expected_size =
      4 * (2 * x_size /* x and bn.y */ + depth_in_ /* bn.var */ +
           1 /* control dependency */);
  EXPECT_EQ(expected_size, cpu_state.memory_usage);

  // Nodes currently in memory: bn's port -1, 0, and 2, and x's port 0.
  std::set<std::pair<string, int>> nodes_in_memory;
  std::transform(
      cpu_state.nodes_in_memory.begin(), cpu_state.nodes_in_memory.end(),
      std::inserter(nodes_in_memory, nodes_in_memory.begin()),
      [](const std::pair<const NodeDef*, int>& node_port) {
        return std::make_pair(node_port.first->name(), node_port.second);
      });
  std::set<std::pair<string, int>> expected = {
      std::make_pair("bn", -1), std::make_pair("bn", 0),
      std::make_pair("bn", 2), std::make_pair("x", 0),
  };
  ExpectSetEq(expected, nodes_in_memory);

  const auto& node_states = scheduler_->GetNodeStates();
  const NodeState* bn_node = nullptr;
  const NodeState* x_node = nullptr;
  for (const auto& nodedef_node_state : node_states) {
    const NodeDef* node = nodedef_node_state.first;
    const NodeState& node_state = nodedef_node_state.second;
    if (node->name() == "bn") {
      bn_node = &node_state;
    }
    if (node->name() == "x") {
      x_node = &node_state;
    }
  }
  CHECK_NOTNULL(bn_node);
  CHECK_NOTNULL(x_node);

  ValidateNodeDefs({"bn", "z1"}, x_node->outputs.at(0));
  ValidateNodeDefs({"z4"}, bn_node->outputs.at(-1));
  ValidateNodeDefs({"z1"}, bn_node->outputs.at(0));
  // z2 and z3 are bn.var + bn.var, so they appear twice in bn's output port 2.
  ValidateNodeDefs({"z2", "z3", "z2", "z3"}, bn_node->outputs.at(2));
}

TEST_F(VirtualSchedulerTest, Variable) {
  // Init.
  CreateGrapplerItemWithConv2DAndVariable();
  InitScheduler();

  // Run the scheduler.
  RunScheduler("");

  const auto& device_states = scheduler_->GetDeviceStates();
  const auto& cpu_state = device_states.at(kCPU0);

  // There is one Conv2D that takes x and f, but f is variable, so it should be
  // in persistent nodes.
  // f is variable.
  ValidateMemoryUsageSnapshot({"f"}, 0 /* port_num_expected */,
                              cpu_state.persistent_nodes);
  // Only x in peak memory usage snapshot.
  ValidateMemoryUsageSnapshot({"x"}, 0 /* port_num_expected */,
                              cpu_state.mem_usage_snapshot_at_peak);
}
}  // end namespace grappler
}  // end namespace tensorflow
