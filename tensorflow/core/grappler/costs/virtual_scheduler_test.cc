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

class VirtualSchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Initializes cluster_ and placer_.
    std::unordered_map<string, DeviceProperties> devices;
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    DeviceProperties gpu_device;
    gpu_device.set_type("GPU");
    devices["/job:localhost/replica:0/task:0/gpu:0"] = gpu_device;

    cluster_.reset(new VirtualCluster(devices));
    placer_.reset(new VirtualPlacer(cluster_.get()));
  }

  void CreateSchedulerWithConv2Ds() {
    // Create a scheduler with a simple graph: 3 Conv2Ds, where only 2 are in
    // fetch nodes.
    const int bs = 4;
    const int width = 10;
    const int height = 10;
    const int depth_in = 8;
    const int kernel = 3;
    const int depth_out = 16;

    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    auto x = tensorflow::ops::RandomUniform(
        s.WithOpName("x"), {bs, width, height, depth_in}, DT_FLOAT);
    auto y = tensorflow::ops::RandomUniform(
        s.WithOpName("y"), {bs, width, height, depth_in}, DT_FLOAT);
    auto z = tensorflow::ops::RandomUniform(
        s.WithOpName("z"), {bs, width, height, depth_in}, DT_FLOAT);
    auto f = tensorflow::ops::RandomUniform(
        s.WithOpName("f"), {kernel, kernel, depth_in, depth_out}, DT_FLOAT);
    std::vector<int> strides = {1, 1, 1, 1};
    auto c0 =
        tensorflow::ops::Conv2D(s.WithOpName("c0"), x, f, strides, "SAME");
    auto c1 =
        tensorflow::ops::Conv2D(s.WithOpName("c1"), y, f, strides, "SAME");
    auto c2 =
        tensorflow::ops::Conv2D(s.WithOpName("c2"), z, f, strides, "SAME");
    GraphDef def;
    TF_CHECK_OK(s.ToGraphDef(&def));
    LOG(INFO) << def.DebugString();

    grappler_item_.reset(new GrapplerItem);
    grappler_item_->id = "test_conv2d_graph";
    grappler_item_->graph = def;
    grappler_item_->fetch = {"c0", "c1"};

    scheduler_.reset(new VirtualScheduler(
        grappler_item_.get(), true /* use_static_shapes */,
        "CPU" /* default_device_type */, cluster_.get(), placer_.get()));
    TF_CHECK_OK(scheduler_->Init());
  }

  // SetUp() inits cluster_ and placer_.
  std::unique_ptr<VirtualCluster> cluster_;
  std::unique_ptr<VirtualPlacer> placer_;

  // grappler_item_ and scheduler_ will be initialized differently for each test
  // case
  std::unique_ptr<GrapplerItem> grappler_item_;
  std::unique_ptr<VirtualScheduler> scheduler_;
};

TEST_F(VirtualSchedulerTest, InitAndBasicScheduling) {
  CreateSchedulerWithConv2Ds();  // init scheduler_.

  Costs zero_costs = Costs::ZeroCosts();
  std::unordered_map<string, NodeInfo> ops_executed;
  do {
    NodeInfo node_info = scheduler_->GetCurrNodeInfo();
    ops_executed[node_info.name] = node_info;

    // Check scheduling order: x and f before c0, and y and f before c1.
    if (node_info.name == "c0") {
      EXPECT_GT(ops_executed.count("x"), 0);
      EXPECT_GT(ops_executed.count("f"), 0);
    } else if (node_info.name == "c1") {
      EXPECT_GT(ops_executed.count("y"), 0);
      EXPECT_GT(ops_executed.count("f"), 0);
    }
  } while (scheduler_->MarkCurrNodeExecuted(zero_costs));

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
  EXPECT_EQ(1, ops_executed["x"].outputs.size());
  EXPECT_EQ(1, ops_executed["y"].outputs.size());
  EXPECT_EQ(1, ops_executed["f"].outputs.size());
  EXPECT_EQ(2, ops_executed["c0"].op_info.inputs_size());
  EXPECT_EQ(2, ops_executed["c1"].op_info.inputs_size());
}
}  // end namespace grappler
}  // end namespace tensorflow
