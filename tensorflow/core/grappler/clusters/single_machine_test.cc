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

#include "tensorflow/core/grappler/clusters/single_machine.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class SingleMachineTest : public ::testing::Test {
 public:
  void SetUp() override {
    // Provision a single machine with 3 cpu cores
    cluster_.reset(new SingleMachine(5 * 60, 3, 0));
    TF_CHECK_OK(cluster_->Provision());
  }

  void TearDown() override {
    cluster_.reset();
  }

 protected:
  std::unique_ptr<SingleMachine> cluster_;
};

TEST_F(SingleMachineTest, CostModel) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_CHECK_OK(cluster_->Initialize(item));

  RunMetadata metadata;
  const int64 start_micros = Env::Default()->NowMicros();
  TF_CHECK_OK(cluster_->Run(item.graph, item.feed, item.fetch, &metadata));
  const int64 run_duration_micros = Env::Default()->NowMicros() - start_micros;

  // There should be at least 4 nodes corresponding to the 4 stages we created
  // in the fake input.
  EXPECT_LE(4, metadata.cost_graph().node_size());
  for (const auto& node : metadata.cost_graph().node()) {
    // Skip the special nodes inserted by TF: these are prefixed with an
    // underscore.
    if (node.name()[0] == '_' || node.name().find("/_") != string::npos) {
      continue;
    }
    EXPECT_EQ(1, node.output_info_size());
    EXPECT_LE(8, node.output_info(0).size());
    const TensorShapeProto& shape = node.output_info(0).shape();
    EXPECT_EQ(2, shape.dim_size());
    EXPECT_EQ(10, shape.dim(0).size());
    EXPECT_EQ(1, shape.dim(1).size());
    EXPECT_LE(0, node.compute_cost());
    EXPECT_GE(run_duration_micros, node.compute_cost());
  }
}

TEST_F(SingleMachineTest, Queue) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, true,
                                          cluster_->GetDeviceNames());
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  TF_CHECK_OK(cluster_->Initialize(item));
  RunMetadata metadata;
  TF_CHECK_OK(cluster_->Run(item.graph, item.feed, item.fetch, &metadata));
}

TEST_F(SingleMachineTest, MultipleItems) {
  TrivialTestGraphInputYielder fake_input(4, 1, 10, false,
                                          cluster_->GetDeviceNames());

  for (int i = 0; i < 3; ++i) {
    GrapplerItem item;
    CHECK(fake_input.NextItem(&item));
    TF_CHECK_OK(cluster_->Initialize(item));
    RunMetadata metadata1;
    TF_CHECK_OK(cluster_->Run(item.graph, item.feed, item.fetch, &metadata1));
    RunMetadata metadata2;
    TF_CHECK_OK(cluster_->Run(item.graph, item.feed, item.fetch, &metadata2));

    // There should be at least 4 nodes corresponding to the 4 stages we created
    // in the fake input, plus 1 enqueue and 1 dequeue node.
    EXPECT_LE(6, metadata1.cost_graph().node_size());
    for (const auto& node : metadata1.cost_graph().node()) {
      if (node.name()[0] == '_' || node.name().find("/_") != string::npos ||
          node.name() == "queue") {
        continue;
      }
      EXPECT_EQ(1, node.output_info_size());
      const TensorShapeProto& shape = node.output_info(0).shape();
      EXPECT_EQ(2, shape.dim_size());
      EXPECT_EQ(10, shape.dim(0).size());
      EXPECT_EQ(1, shape.dim(1).size());
    }

    for (int i = 0; i < metadata1.cost_graph().node_size(); ++i) {
      metadata1.mutable_cost_graph()->mutable_node(i)->set_compute_cost(0);
      metadata1.clear_step_stats();
    }
    for (int i = 0; i < metadata2.cost_graph().node_size(); ++i) {
      metadata2.mutable_cost_graph()->mutable_node(i)->set_compute_cost(0);
      metadata2.clear_step_stats();
    }
    string s1;
    ::tensorflow::protobuf::TextFormat::PrintToString(metadata1, &s1);
    string s2;
    ::tensorflow::protobuf::TextFormat::PrintToString(metadata2, &s2);
    EXPECT_EQ(s1, s2);
  }
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
