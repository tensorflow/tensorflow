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

#include "tensorflow/core/grappler/optimizers/static_schedule.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class StaticScheduleTest : public ::testing::Test {
 public:
  std::unique_ptr<VirtualCluster> CreateVirtualCluster() const {
    // Invent a CPU so that predictions remain the same from machine to machine.
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_frequency(1000);
    cpu_device.set_num_cores(4);
    cpu_device.set_bandwidth(32);
    cpu_device.set_l1_cache_size(32 * 1024);
    cpu_device.set_l2_cache_size(256 * 1024);
    cpu_device.set_l3_cache_size(4 * 1024 * 1024);
    std::unordered_map<string, DeviceProperties> devices;
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    return std::unique_ptr<VirtualCluster>(new VirtualCluster(devices));
  }
};

TEST_F(StaticScheduleTest, BasicGraph) {
  // This trivial graph is so basic there's nothing to prune.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  std::unordered_map<const NodeDef*, Costs::NanoSeconds> completion_times;
  Status status =
      EstimateEarliestExecutionTimes(item, cluster.get(), &completion_times);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size(), completion_times.size());

  for (auto time : completion_times) {
    if (time.first->name() == "Const/Const") {
      EXPECT_EQ(Costs::NanoSeconds(1), time.second);
    } else if (time.first->name() == "x") {
      EXPECT_EQ(Costs::NanoSeconds(1500001), time.second);
    } else if (time.first->name() == "Square") {
      EXPECT_EQ(Costs::NanoSeconds(4000004), time.second);
    } else if (time.first->name() == "Square_1") {
      EXPECT_EQ(Costs::NanoSeconds(6500007), time.second);
    } else if (time.first->name() == "Square_2") {
      EXPECT_EQ(Costs::NanoSeconds(9000010), time.second);
    } else if (time.first->name() == "Square_3") {
      EXPECT_EQ(Costs::NanoSeconds(11500013), time.second);
    } else if (time.first->name() == "y") {
      EXPECT_EQ(Costs::NanoSeconds(14000013), time.second);
    }
  }
}

TEST_F(StaticScheduleTest, BasicGraphWithCtrlDependencies) {
  // Build a simple graph with a control dependency.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::AddN(s.WithOpName("b"), {a});
  Output c = ops::Identity(s.WithOpName("c"), b);
  Output d = ops::Identity(s.WithOpName("d"), c);
  Output e = ops::AddN(s.WithOpName("e"), {d});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  // Add a control dependency between c and e.
  EXPECT_EQ("c", item.graph.node(2).name());
  EXPECT_EQ("e", item.graph.node(4).name());
  *item.graph.mutable_node(4)->add_input() = "^c";

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  std::unordered_map<const NodeDef*, Costs::NanoSeconds> completion_times;
  Status status =
      EstimateEarliestExecutionTimes(item, cluster.get(), &completion_times);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size(), completion_times.size());

  for (auto time : completion_times) {
    if (time.first->name() == "a") {
      EXPECT_EQ(Costs::NanoSeconds(1), time.second);
    } else if (time.first->name() == "b") {
      EXPECT_EQ(Costs::NanoSeconds(25000001), time.second);
    } else if (time.first->name() == "c") {
      EXPECT_EQ(Costs::NanoSeconds(25000002), time.second);
    } else if (time.first->name() == "d") {
      EXPECT_EQ(Costs::NanoSeconds(25000003), time.second);
    } else if (time.first->name() == "e") {
      EXPECT_EQ(Costs::NanoSeconds(50000003), time.second);
    }
  }
}

TEST_F(StaticScheduleTest, RequiredTimes) {
  // This trivial graph is so basic there's nothing to prune.
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false, {"CPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  std::unordered_map<const NodeDef*, Costs::NanoSeconds> execution_times;
  for (const NodeDef& node : item.graph.node()) {
    execution_times[&node] = 0;
  }
  std::unordered_map<const NodeDef*, Costs::NanoSeconds> required_times;
  Status status = EstimateRequiredTimes(item, cluster.get(), execution_times,
                                        &required_times);
  TF_EXPECT_OK(status);

  EXPECT_EQ(item.graph.node_size(), required_times.size());

  for (auto time : required_times) {
    if (time.first->name() == "Const/Const") {
      EXPECT_EQ(Costs::NanoSeconds(-14000012), time.second);
    } else if (time.first->name() == "x") {
      EXPECT_EQ(Costs::NanoSeconds(-12500012), time.second);
    } else if (time.first->name() == "Square") {
      EXPECT_EQ(Costs::NanoSeconds(-10000009), time.second);
    } else if (time.first->name() == "Square_1") {
      EXPECT_EQ(Costs::NanoSeconds(-7500006), time.second);
    } else if (time.first->name() == "Square_2") {
      EXPECT_EQ(Costs::NanoSeconds(-5000003), time.second);
    } else if (time.first->name() == "Square_3") {
      EXPECT_EQ(Costs::NanoSeconds(-2500000), time.second);
    } else if (time.first->name() == "y") {
      EXPECT_EQ(Costs::NanoSeconds(0), time.second);
    }
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
