/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/graph/collective_order.h"

#include <gmock/gmock.h>
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ::testing::UnorderedElementsAreArray;

REGISTER_OP("TestParams").Output("o: float");

// Verifies that the list of collective nodes in `graph` matches
// `expected_collective_nodes`, and that the list of control edges between these
// collective nodes matches `expected_collective_control_edges`.
void VerifyGraph(const Graph& graph,
                 const std::vector<string>& expected_collective_nodes,
                 const std::vector<std::pair<string, string>>&
                     expected_collective_control_edges) {
  std::vector<string> actual_collective_nodes;
  std::vector<std::pair<string, string>> actual_collective_control_edges;
  for (const Node* src : graph.nodes()) {
    if (!src->IsCollective()) {
      continue;
    }
    actual_collective_nodes.push_back(src->name());
    for (const Edge* edge : src->out_edges()) {
      VLOG(2) << "collective edge " << edge->src()->name() << " -> "
              << edge->dst()->name();
      // Add all control edges found except those to `_SINK`.
      if (!edge->IsControlEdge() || edge->dst()->name() == "_SINK") {
        continue;
      }
      actual_collective_control_edges.emplace_back(src->name(),
                                                   edge->dst()->name());
    }
  }
  EXPECT_THAT(actual_collective_nodes,
              UnorderedElementsAreArray(expected_collective_nodes));
  EXPECT_THAT(actual_collective_control_edges,
              UnorderedElementsAreArray(expected_collective_control_edges));
}

// Verifies that the `wait_for` attribute on collective nodes matches
// `wait_for_map`.
void VerifyAttrs(
    const Graph& graph,
    const std::unordered_map<string, std::vector<int32>> wait_for_map) {
  for (const Node* node : graph.nodes()) {
    if (node->IsCollective() ||
        wait_for_map.find(node->name()) == wait_for_map.end()) {
      continue;
    }
    std::vector<int32> wait_for_actual;
    TF_EXPECT_OK(GetNodeAttr(node->attrs(), "wait_for", &wait_for_actual));
    auto wait_for_expected = wait_for_map.at(node->name());
    EXPECT_THAT(wait_for_actual, UnorderedElementsAreArray(wait_for_expected));
  }
}

Node* CollectiveReduceNode(GraphDefBuilder* builder, Node* input,
                           const string& name, const string& device,
                           int instance_key) {
  Node* collective_node =
      ops::UnaryOp("CollectiveReduce", input,
                   builder->opts()
                       .WithName(name)
                       .WithDevice(device)
                       .WithAttr("T", DT_FLOAT)
                       .WithAttr("group_size", 2)
                       .WithAttr("group_key", 1)
                       .WithAttr("instance_key", instance_key)
                       .WithAttr("merge_op", "Add")
                       .WithAttr("final_op", "Id")
                       .WithAttr("subdiv_offsets", {1}));
  return collective_node;
}

// Initialize the following graph:
//
//       (cpu0) (cpu1)
//         a      b
//         |      |
//         c1     c1
//         |      |
//         id     id
//        /  \   /  \
//       c2  c3 c2  c3
//
// Here ci denotes a collective node with `instance_key` i.  `a` and `b` are
// inputs, `id` is identity node.
std::unique_ptr<Graph> InitGraph() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  const string dev1 = "/job:localhost/replica:0/task:0/device:CPU:1";
  Node* a = ops::SourceOp("TestParams",
                          builder.opts().WithName("a").WithDevice(dev0));
  Node* b = ops::SourceOp("TestParams",
                          builder.opts().WithName("b").WithDevice(dev1));
  Node* c1_0 = CollectiveReduceNode(&builder, a, "c1_0", dev0, 1);
  Node* c1_1 = CollectiveReduceNode(&builder, b, "c1_1", dev1, 1);
  Node* id0 = ops::UnaryOp(
      "Identity", c1_0,
      builder.opts().WithName("id0").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  Node* id1 = ops::UnaryOp(
      "Identity", c1_1,
      builder.opts().WithName("id1").WithDevice(dev1).WithAttr("T", DT_FLOAT));
  CollectiveReduceNode(&builder, id0, "c2_0", dev0, 2);
  CollectiveReduceNode(&builder, id1, "c2_1", dev1, 2);
  CollectiveReduceNode(&builder, id0, "c3_0", dev0, 3);
  CollectiveReduceNode(&builder, id1, "c3_1", dev1, 3);

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// Tests that in the graph created by `InitGraph`, exactly 2 control edges are
// added after calling `OrderCollectives`: c3_0 -> c2_0 and c3_1 -> c2_1.
TEST(CollectiveOrderTest, SimpleOrder) {
  std::unique_ptr<Graph> graph = InitGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  VerifyGraph(*graph, {"c1_0", "c1_1", "c2_0", "c2_1", "c3_0", "c3_1"},
              {{"c3_0", "c2_0"}, {"c3_1", "c2_1"}});
}

TEST(CollectiveOrderTest, SimpleOrderAttr) {
  std::unique_ptr<Graph> graph = InitGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kAttrs));
  VerifyAttrs(*graph, {{"c2_0", {3}}, {"c2_1", {3}}});
}

// Initialize the following graph:
//
//         a
//         |
//         c1
//        /  \
//       c4  id
//          /  \
//         c2  c3
//
// Here ci denotes a collective node with `instance_key` i.  `a` is an input,
// `id` is identity node.
std::unique_ptr<Graph> InitGraph2() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* a = ops::SourceOp("TestParams",
                          builder.opts().WithName("a").WithDevice(dev0));
  Node* c1 = CollectiveReduceNode(&builder, a, "c1", dev0, 1);
  CollectiveReduceNode(&builder, c1, "c4", dev0, 4);
  Node* id = ops::UnaryOp(
      "Identity", c1,
      builder.opts().WithName("id").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  CollectiveReduceNode(&builder, id, "c2", dev0, 2);
  CollectiveReduceNode(&builder, id, "c3", dev0, 3);

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// Tests that in the graph created by `InitGraph2`, we add the following control
// edges after calling `OrderCollectives`: c4 -> c3, c3 -> c2.  c4->c2 is
// pruned because it follows from the other two edges.
TEST(CollectiveOrderTest, SimpleOrder2) {
  std::unique_ptr<Graph> graph = InitGraph2();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  VerifyGraph(*graph, {"c1", "c2", "c3", "c4"}, {{"c4", "c3"}, {"c3", "c2"}});
}

// Initialize the following graph:
//
//         w   x   y   z
//         |   |   |   |
//         c1  c2  c3  c4
//
std::unique_ptr<Graph> InitGraphForPruning() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* w = ops::SourceOp("TestParams",
                          builder.opts().WithName("w").WithDevice(dev0));
  Node* x = ops::SourceOp("TestParams",
                          builder.opts().WithName("x").WithDevice(dev0));
  Node* y = ops::SourceOp("TestParams",
                          builder.opts().WithName("y").WithDevice(dev0));
  Node* z = ops::SourceOp("TestParams",
                          builder.opts().WithName("z").WithDevice(dev0));
  CollectiveReduceNode(&builder, w, "c1", dev0, 1);
  CollectiveReduceNode(&builder, x, "c2", dev0, 2);
  CollectiveReduceNode(&builder, y, "c3", dev0, 3);
  CollectiveReduceNode(&builder, z, "c4", dev0, 4);

  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// Tests that in the graph created by `InitGraphForPruning`, we only add c4 ->
// c3, c3 -> c2, c2 -> c1, and other edges are pruned away.
TEST(CollectiveOrderTest, Pruning) {
  std::unique_ptr<Graph> graph = InitGraphForPruning();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kAttrs));
  VerifyAttrs(*graph, {{"c3", {4}}, {"c2", {3}}, {"c1", {2}}});
}

}  // namespace
}  // namespace tensorflow
