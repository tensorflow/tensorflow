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

#include "tensorflow/core/grappler/optimizers/memory_optimizer.h"

#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/clusters/virtual_cluster.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class RecomputeSubgraphTest : public ::testing::Test {};

TEST_F(RecomputeSubgraphTest, SimpleSubgraph) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Identity(s.WithOpName("b"), a);  // Recomputed
  Output c = ops::Identity(s.WithOpName("c"), b);
  Output d = ops::AddN(s.WithOpName("gradients/d"), {c});
  Output e = ops::AddN(s.WithOpName("gradients/e"), {d, b});
  Output f = ops::AddN(s.WithOpName("gradients/f"), {e, a});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(6, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  (*pre_transform_node_map.GetNode("b")->mutable_attr())["_recompute_hint"]
      .set_i(0);

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);

  TF_EXPECT_OK(status);
  NodeMap post_transform_node_map(&output);
  EXPECT_EQ(8, output.node_size());
  NodeDef* transformed_e = post_transform_node_map.GetNode(e.name());
  EXPECT_EQ(2, transformed_e->input_size());
  EXPECT_EQ("gradients/d", transformed_e->input(0));
  EXPECT_EQ("Recomputed/b", transformed_e->input(1));
  NodeDef* recomputed_b = post_transform_node_map.GetNode("Recomputed/b");
  EXPECT_EQ(2, recomputed_b->input_size());
  EXPECT_EQ("a", recomputed_b->input(0));
  EXPECT_EQ("^RecomputeTrigger/b", recomputed_b->input(1));
  NodeDef* recompute_trigger =
      post_transform_node_map.GetNode("RecomputeTrigger/b");
  EXPECT_EQ(1, recompute_trigger->input_size());
  EXPECT_EQ("^gradients/d", recompute_trigger->input(0));
}

TEST_F(RecomputeSubgraphTest, NoFeedsRecomputed) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Identity(s.WithOpName("b"), a);  // Would be recomputed, but
                                                   // for being fed
  Output c = ops::Identity(s.WithOpName("c"), b);
  Output d = ops::AddN(s.WithOpName("gradients/d"), {c});
  Output e = ops::AddN(s.WithOpName("gradients/e"), {d, b});
  Output f = ops::AddN(s.WithOpName("gradients/f"), {e, a});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  item.feed.emplace_back("b", Tensor());
  EXPECT_EQ(6, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  (*pre_transform_node_map.GetNode("b")->mutable_attr())["_recompute_hint"]
      .set_i(0);

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);

  TF_EXPECT_OK(status);
  EXPECT_EQ(6, output.node_size());
}

TEST_F(RecomputeSubgraphTest, TwoInputSubgraphs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Variable(s.WithOpName("b"), {2, 3, 4}, DT_FLOAT);
  Output d = ops::AddN(s.WithOpName("gradients/two_subgraph_inputs"), {a, b});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(3, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  (*pre_transform_node_map.GetNode("a")->mutable_attr())["_recompute_hint"]
      .set_i(0);
  (*pre_transform_node_map.GetNode("b")->mutable_attr())["_recompute_hint"]
      .set_i(0);

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);

  TF_EXPECT_OK(status);
  NodeMap post_transform_node_map(&output);
  // Mostly checking that this case does not crash.
  EXPECT_EQ(7, output.node_size());
  EXPECT_NE(post_transform_node_map.GetNode("Recomputed/a"), nullptr);
  EXPECT_NE(post_transform_node_map.GetNode("Recomputed/b"), nullptr);
  EXPECT_NE(post_transform_node_map.GetNode("RecomputeTrigger/a"), nullptr);
  EXPECT_NE(post_transform_node_map.GetNode("RecomputeTrigger/b"), nullptr);
}

TEST_F(RecomputeSubgraphTest, MultiNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("Conv"), {2, 3, 4}, DT_FLOAT);
  Output b = ops::Identity(s.WithOpName("BN"), a);    // Recomputed
  Output c = ops::Identity(s.WithOpName("ReLU"), b);  // Recomputed
  Output d = ops::Identity(s.WithOpName("Conv1"), c);

  // The "gradients/" prefix means the heuristic will pick these up as
  // candidates to have their inputs recomputed.
  Output trigger = ops::AddN(s.WithOpName("gradients/BN1Grad"), {d});
  Output e = ops::AddN(s.WithOpName("gradients/Conv1Grad"), {trigger, c});
  Output f = ops::AddN(s.WithOpName("gradients/ReLUGrad"), {e, c});
  Output g = ops::AddN(s.WithOpName("gradients/BNGrad"), {f, a});
  Output h = ops::AddN(s.WithOpName("gradients/ConvGrad"), {g});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(9, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  // Set op types so that the heuristic will pick these nodes up to be
  // recomputed
  pre_transform_node_map.GetNode("BN")->set_op("FusedBatchNorm");
  pre_transform_node_map.GetNode("ReLU")->set_op("Relu");

  MemoryOptimizer optimizer(RewriterConfig::HEURISTICS);
  GraphDef first_pass_output;
  Status first_pass_status =
      optimizer.Optimize(nullptr, item, &first_pass_output);
  TF_EXPECT_OK(first_pass_status);

  NodeMap post_transform_node_map(&first_pass_output);
  EXPECT_EQ(13, first_pass_output.node_size());
  NodeDef* transformed_e = post_transform_node_map.GetNode(e.name());
  EXPECT_EQ(2, transformed_e->input_size());
  EXPECT_EQ("gradients/BN1Grad", transformed_e->input(0));
  EXPECT_EQ("Recomputed/ReLU", transformed_e->input(1));
  NodeDef* transformed_f = post_transform_node_map.GetNode(f.name());
  EXPECT_EQ(2, transformed_f->input_size());
  EXPECT_EQ("gradients/Conv1Grad", transformed_f->input(0));
  EXPECT_EQ("Recomputed/ReLU", transformed_f->input(1));
  NodeDef* transformed_g = post_transform_node_map.GetNode(g.name());
  EXPECT_EQ(2, transformed_g->input_size());
  EXPECT_EQ("gradients/ReLUGrad", transformed_g->input(0));
  EXPECT_EQ("Conv", transformed_g->input(1));

  NodeDef* recomputed_b = post_transform_node_map.GetNode("Recomputed/BN");
  EXPECT_EQ(2, recomputed_b->input_size());
  EXPECT_EQ("Conv", recomputed_b->input(0));
  EXPECT_EQ("^RecomputeTrigger/BN", recomputed_b->input(1));
  NodeDef* recompute_trigger_b =
      post_transform_node_map.GetNode("RecomputeTrigger/BN");
  EXPECT_EQ(1, recompute_trigger_b->input_size());
  EXPECT_EQ("^RecomputeTrigger/ReLU", recompute_trigger_b->input(0));

  NodeDef* recomputed_c = post_transform_node_map.GetNode("Recomputed/ReLU");
  EXPECT_EQ(2, recomputed_c->input_size());
  EXPECT_EQ("Recomputed/BN", recomputed_c->input(0));
  EXPECT_EQ("^RecomputeTrigger/ReLU", recomputed_c->input(1));
  NodeDef* recompute_trigger_c =
      post_transform_node_map.GetNode("RecomputeTrigger/ReLU");
  EXPECT_EQ(1, recompute_trigger_c->input_size());
  EXPECT_EQ("^gradients/BN1Grad", recompute_trigger_c->input(0));
}

class MemoryOptimizerTest : public ::testing::Test {
 public:
  static std::unique_ptr<VirtualCluster> CreateVirtualCluster() {
    DeviceProperties cpu_device;
    cpu_device.set_type("CPU");
    cpu_device.set_frequency(1000);
    cpu_device.set_num_cores(4);
    cpu_device.set_bandwidth(32);
    std::unordered_map<string, DeviceProperties> devices;
    devices["/job:localhost/replica:0/task:0/cpu:0"] = cpu_device;
    return std::unique_ptr<VirtualCluster>(new VirtualCluster(devices));
  }
};

TEST_F(MemoryOptimizerTest, SimpleSwapping) {
  // Build a simple graph with an op that's marked for swapping.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Variable(s.WithOpName("a"), {10, 10}, DT_FLOAT);
  Output b = ops::AddN(s.WithOpName("b"), {a});
  Output c = ops::AddN(s.WithOpName("c"), {b});
  Output d = ops::AddN(s.WithOpName("d"), {c});
  Output e = ops::AddN(s.WithOpName("e"), {b, d});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  EXPECT_EQ(5, item.graph.node_size());
  EXPECT_EQ(NodeName(e.name()), item.graph.node(4).name());
  AttrValue& val =
      (*item.graph.mutable_node(4)->mutable_attr())["_swap_to_host"];
  val.mutable_list()->add_i(0);

  std::unique_ptr<VirtualCluster> cluster(CreateVirtualCluster());

  MemoryOptimizer optimizer(RewriterConfig::MANUAL);
  GraphDef output;
  Status status = optimizer.Optimize(cluster.get(), item, &output);
  TF_EXPECT_OK(status);

  EXPECT_EQ(7, output.node_size());
  const NodeDef& new_e = output.node(4);
  EXPECT_EQ(NodeName(e.name()), new_e.name());

  EXPECT_EQ(2, new_e.input_size());
  EXPECT_EQ(NodeName(d.name()), new_e.input(1));
  EXPECT_EQ("swap_in_e_0", new_e.input(0));

  const NodeDef& swap_out = output.node(5);
  EXPECT_EQ("swap_out_e_0", swap_out.name());

  const NodeDef& swap_in = output.node(6);
  EXPECT_EQ("swap_in_e_0", swap_in.name());

  EXPECT_EQ(NodeName(b.name()), swap_out.input(0));
  EXPECT_EQ(NodeName(swap_out.name()), swap_in.input(0));
  EXPECT_EQ("^c", swap_in.input(1));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
