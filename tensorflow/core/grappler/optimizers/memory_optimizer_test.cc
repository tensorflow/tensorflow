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

  Output a = ops::Const(s.WithOpName("a"), 1.f, {2, 3, 4});
  Output b = ops::AddN(s.WithOpName("b"), {a});  // Recomputed
  Output c = ops::AddN(s.WithOpName("c"), {b});
  Output d = ops::AddN(s.WithOpName("d"), {c});
  Output e = ops::AddN(s.WithOpName("e"), {d, b});
  Output f = ops::AddN(s.WithOpName("f"), {e, a});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(6, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  std::vector<const NodeDef*> recomputed_source_nodes;
  recomputed_source_nodes.push_back(pre_transform_node_map.GetNode(b.name()));
  std::vector<NodeDef*> target_nodes;
  target_nodes.push_back(pre_transform_node_map.GetNode(e.name()));
  RecomputeSubgraph(recomputed_source_nodes, d.name(), target_nodes,
                    &item.graph);
  NodeMap post_transform_node_map(&item.graph);
  EXPECT_EQ(7, item.graph.node_size());
  NodeDef* transformed_e = post_transform_node_map.GetNode(e.name());
  EXPECT_EQ(2, transformed_e->input_size());
  EXPECT_EQ("d", transformed_e->input(0));
  EXPECT_EQ("Recomputed/b", transformed_e->input(1));
  NodeDef* recomputed_b = post_transform_node_map.GetNode("Recomputed/b");
  EXPECT_EQ(2, recomputed_b->input_size());
  EXPECT_EQ("a", recomputed_b->input(0));
  EXPECT_EQ("^d", recomputed_b->input(1).substr(0, 2));
}

TEST_F(RecomputeSubgraphTest, MultiNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("Conv"), 1.f, {2, 3, 4});
  Output b = ops::AddN(s.WithOpName("BN"), {a});    // Recomputed
  Output c = ops::AddN(s.WithOpName("ReLU"), {b});  // Recomputed
  Output d = ops::AddN(s.WithOpName("Conv1"), {c});

  Output trigger = ops::Const(s.WithOpName("BN1Grad"), 0.f, {2, 3, 4});
  Output e = ops::AddN(s.WithOpName("Conv1Grad"), {trigger, c});
  Output f = ops::AddN(s.WithOpName("ReLUGrad"), {e, c});
  Output g = ops::AddN(s.WithOpName("BNGrad"), {f, a});
  Output h = ops::AddN(s.WithOpName("ConvGrad"), {g});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  EXPECT_EQ(9, item.graph.node_size());
  NodeMap pre_transform_node_map(&item.graph);
  std::vector<const NodeDef*> recomputed_source_nodes;
  recomputed_source_nodes.push_back(pre_transform_node_map.GetNode(b.name()));
  recomputed_source_nodes.push_back(pre_transform_node_map.GetNode(c.name()));
  std::vector<NodeDef*> target_nodes;
  target_nodes.push_back(pre_transform_node_map.GetNode(e.name()));
  target_nodes.push_back(pre_transform_node_map.GetNode(f.name()));
  target_nodes.push_back(pre_transform_node_map.GetNode(g.name()));
  RecomputeSubgraph(recomputed_source_nodes, trigger.name(), target_nodes,
                    &item.graph);
  NodeMap post_transform_node_map(&item.graph);
  EXPECT_EQ(11, item.graph.node_size());
  NodeDef* transformed_e = post_transform_node_map.GetNode(e.name());
  EXPECT_EQ(2, transformed_e->input_size());
  EXPECT_EQ("BN1Grad", transformed_e->input(0));
  EXPECT_EQ("Recomputed/ReLU", transformed_e->input(1));
  NodeDef* transformed_f = post_transform_node_map.GetNode(f.name());
  EXPECT_EQ(2, transformed_f->input_size());
  EXPECT_EQ("Conv1Grad", transformed_f->input(0));
  EXPECT_EQ("Recomputed/ReLU", transformed_f->input(1));
  NodeDef* transformed_g = post_transform_node_map.GetNode(g.name());
  EXPECT_EQ(2, transformed_g->input_size());
  EXPECT_EQ("ReLUGrad", transformed_g->input(0));
  EXPECT_EQ("Conv", transformed_g->input(1));

  NodeDef* recomputed_b = post_transform_node_map.GetNode("Recomputed/BN");
  EXPECT_EQ(2, recomputed_b->input_size());
  EXPECT_EQ("Conv", recomputed_b->input(0));
  EXPECT_EQ("^BN1Grad", recomputed_b->input(1).substr(0, 8));
  NodeDef* recomputed_c = post_transform_node_map.GetNode("Recomputed/ReLU");
  EXPECT_EQ(2, recomputed_c->input_size());
  EXPECT_EQ("Recomputed/BN", recomputed_c->input(0));
  EXPECT_EQ("^BN1Grad", recomputed_c->input(1).substr(0, 8));
}

class MemoryOptimizerTest : public ::testing::Test {};

TEST_F(MemoryOptimizerTest, SimpleSwapping) {
  // Build a simple graph with an op that's marked for swapping.
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::AddN(s.WithOpName("b"), {a});
  Output c = ops::AddN(s.WithOpName("c"), {b});
  Output d = ops::AddN(s.WithOpName("d"), {c});
  Output e = ops::AddN(s.WithOpName("e"), {b, d});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));

  EXPECT_EQ(5, item.graph.node_size());
  EXPECT_EQ(NodeName(e.name()), item.graph.node(4).name());
  AttrValue& val =
      (*item.graph.mutable_node(4)->mutable_attr())["swap_to_host"];
  val.mutable_list()->add_i(0);

  MemoryOptimizer optimizer;
  GraphDef output;
  Status status = optimizer.Optimize(nullptr, item, &output);
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
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
