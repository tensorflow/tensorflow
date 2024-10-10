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

#include "tensorflow/core/grappler/clusters/virtual_cluster.h"

#include <memory>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class VirtualClusterTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Invent a CPU so that predictions remain the same from machine to machine.
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_frequency(1000);
    cpu_device.set_num_cores(4);
    cpu_device.set_bandwidth(32);
    cpu_device.set_l1_cache_size(32 * 1024);
    cpu_device.set_l2_cache_size(256 * 1024);
    cpu_device.set_l3_cache_size(4 * 1024 * 1024);
    cpu_device.set_memory_size(1024 * 1024);
    std::unordered_map<string, DeviceProperties> devices;
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    cluster_ = std::make_unique<VirtualCluster>(devices);
    TF_CHECK_OK(cluster_->Provision());
  }

  void TearDown() override {
    TF_CHECK_OK(cluster_->Shutdown());
    cluster_.reset();
  }

 protected:
  std::unique_ptr<VirtualCluster> cluster_;
};

TEST_F(VirtualClusterTest, ClusterType) {
  CHECK_EQ("virtual", cluster_->type());
}

TEST_F(VirtualClusterTest, CostModel) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_CHECK_OK(cluster_->Initialize(item));

  RunMetadata metadata;
  TF_CHECK_OK(cluster_->Run(item.graph, item.feed, item.fetch, &metadata));

  // There should be at least 4 nodes corresponding to the 4 stages we created
  // in the fake input.
  EXPECT_LE(4, metadata.cost_graph().node_size());
  for (const auto& node : metadata.cost_graph().node()) {
    // Skip the constant node that configures the random number generator.
    if (node.name().find("Const/Const") != string::npos) {
      continue;
    }
    EXPECT_EQ(1, node.output_info_size());
    EXPECT_EQ(40, node.output_info(0).size());
    const TensorShapeProto& shape = node.output_info(0).shape();
    EXPECT_EQ(2, shape.dim_size());
    EXPECT_EQ(10, shape.dim(0).size());
    EXPECT_EQ(1, shape.dim(1).size());
    if (node.name() == "x") {
      EXPECT_EQ(1500, node.compute_cost());
    } else {
      EXPECT_EQ(2500, node.compute_cost());
    }
  }

  for (const auto& dev_stat : metadata.step_stats().dev_stats()) {
    EXPECT_EQ("/job:localhost/replica:0/task:0/cpu:0", dev_stat.device());
    for (const auto& node : dev_stat.node_stats()) {
      if (node.node_name() == "AddN") {
        EXPECT_EQ(2500, node.op_end_rel_micros());
      }
    }
  }
}

TEST_F(VirtualClusterTest, OutOfMemory) {
  tensorflow::Scope root = tensorflow::Scope::NewRootScope();
  // Create a large variable that can't fit in memory.
  auto zero = ops::Variable(root.WithOpName("zero"), {1024, 1024}, DT_FLOAT);
  auto identity = ops::Identity(root.WithOpName("i"), zero);
  auto identity2 = ops::Identity(root.WithOpName("i2"), identity);
  GrapplerItem item;
  TF_CHECK_OK(root.ToGraphDef(&item.graph));
  item.fetch.push_back("i2");

  TF_CHECK_OK(cluster_->Initialize(item));
  absl::Status s = cluster_->Run(item.graph, item.feed, item.fetch, nullptr);
  EXPECT_EQ(error::RESOURCE_EXHAUSTED, s.code());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
