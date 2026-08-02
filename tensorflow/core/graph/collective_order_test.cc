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

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/common_runtime/graph_def_builder_util.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder.h"
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
                 const std::vector<std::string>& expected_collective_nodes,
                 const std::vector<std::pair<std::string, std::string>>&
                     expected_collective_control_edges) {
  std::vector<std::string> actual_collective_nodes;
  std::vector<std::pair<std::string, std::string>>
      actual_collective_control_edges;
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
    const std::unordered_map<std::string, std::vector<int32_t>> wait_for_map) {
  for (const Node* node : graph.nodes()) {
    if (node->IsCollective() ||
        wait_for_map.find(node->name()) == wait_for_map.end()) {
      continue;
    }
    std::vector<int32_t> wait_for_actual;
    TF_EXPECT_OK(GetNodeAttr(node->attrs(), "wait_for", &wait_for_actual));
    auto wait_for_expected = wait_for_map.at(node->name());
    EXPECT_THAT(wait_for_actual, UnorderedElementsAreArray(wait_for_expected));
  }
}

Node* CollectiveReduceNode(GraphDefBuilder* builder, Node* input,
                           const std::string& name, const std::string& device,
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
  const std::string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  const std::string dev1 = "/job:localhost/replica:0/task:0/device:CPU:1";
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

  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = GraphDefBuilderToGraph(builder, graph.get());
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

// Initialize the following graph, all on one device:
//
//              a
//            /   \
//          c3     c4
//          |      |
//         id3    id4
//          |      |
//          c1     c2
//
// Here ci denotes a collective node with `instance_key` i, and `id` is an
// identity node.  The data dependencies are chosen so that exactly two of the
// six collective pairs already have a data path -- (c3, c1) and (c4, c2) --
// and the remaining four become control edges oriented from the higher
// instance key to the lower one:
//
//   c4 -> c3,  c4 -> c1,  c3 -> c2,  c2 -> c1
//
// c4 -> c1 is redundant: the path c4 -> c3 -> c2 -> c1 already orders them.
// The transitive reduction of a DAG is unique, so the only correct output is
// {c4 -> c3, c3 -> c2, c2 -> c1}.
//
// The instance keys are assigned so that they descend along the order in which
// `DiscoverDataDependencies` discovers the collectives.  That makes the pair
// (c3, c2) reach the enumeration after (c4, c1), so `all_paths[c3]` is filled
// before c2 acquires c1, and the redundancy of c4 -> c1 is only visible to a
// closure that back-propagates.
std::unique_ptr<Graph> InitDiamondGraph() {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const std::string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* a = ops::SourceOp("TestParams",
                          builder.opts().WithName("a").WithDevice(dev0));
  Node* c3 = CollectiveReduceNode(&builder, a, "c3", dev0, 3);
  Node* c4 = CollectiveReduceNode(&builder, a, "c4", dev0, 4);
  Node* id3 = ops::UnaryOp(
      "Identity", c3,
      builder.opts().WithName("id3").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  Node* id4 = ops::UnaryOp(
      "Identity", c4,
      builder.opts().WithName("id4").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  CollectiveReduceNode(&builder, id3, "c1", dev0, 1);
  CollectiveReduceNode(&builder, id4, "c2", dev0, 2);

  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) {
    LOG(FATAL) << "Error building graph " << s;
  }
  return graph;
}

// The emitted control edges must be the transitive reduction of the dependency
// graph.  That object is unique for a DAG, so this is the only admissible
// output and it cannot depend on iteration order.
//
// Before the transitive closure fix, `all_paths[c3]` was populated from
// `all_paths[c2]` at the moment the edge c3 -> c2 was created, which happens
// before c2 -> c1 exists.  Nothing back-propagated c1 into it afterwards, so
// the pruning step never saw the path c4 -> c3 -> c2 -> c1 and kept the
// redundant edge c4 -> c1.
TEST(CollectiveOrderTest, TransitiveReductionIsExact) {
  std::unique_ptr<Graph> graph = InitDiamondGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  VerifyGraph(*graph, {"c1", "c2", "c3", "c4"},
              {{"c2", "c1"}, {"c3", "c2"}, {"c4", "c3"}});
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
  const std::string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* a = ops::SourceOp("TestParams",
                          builder.opts().WithName("a").WithDevice(dev0));
  Node* c1 = CollectiveReduceNode(&builder, a, "c1", dev0, 1);
  CollectiveReduceNode(&builder, c1, "c4", dev0, 4);
  Node* id = ops::UnaryOp(
      "Identity", c1,
      builder.opts().WithName("id").WithDevice(dev0).WithAttr("T", DT_FLOAT));
  CollectiveReduceNode(&builder, id, "c2", dev0, 2);
  CollectiveReduceNode(&builder, id, "c3", dev0, 3);

  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = GraphDefBuilderToGraph(builder, graph.get());
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
  const std::string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
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

  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = GraphDefBuilderToGraph(builder, graph.get());
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

// Control edges between collective nodes, as (source name, target name), sorted
// so the result is comparable across runs and across processes.
std::vector<std::pair<std::string, std::string>> CollectiveControlEdges(
    const Graph& graph) {
  std::vector<std::pair<std::string, std::string>> edges;
  for (const Node* src : graph.nodes()) {
    if (!src->IsCollective()) continue;
    for (const Edge* edge : src->out_edges()) {
      if (!edge->IsControlEdge()) continue;
      if (!edge->dst()->IsCollective()) continue;
      edges.emplace_back(src->name(), edge->dst()->name());
    }
  }
  std::sort(edges.begin(), edges.end());
  return edges;
}

// Transitive closure of `edges`, as target sets keyed by source.
std::map<std::string, std::set<std::string>> Reachability(
    const std::vector<std::pair<std::string, std::string>>& edges) {
  std::map<std::string, std::set<std::string>> reach;
  for (const auto& e : edges) reach[e.first].insert(e.second);
  // Repeat to a fixed point; the graphs here are tiny.
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto& e : edges) {
      const auto it = reach.find(e.second);
      if (it == reach.end()) continue;
      for (const std::string& downstream : it->second) {
        if (reach[e.first].insert(downstream).second) changed = true;
      }
    }
  }
  return reach;
}

// A transitive reduction contains no edge implied by the others: for every
// emitted edge u -> v there must be no path u -> ... -> v avoiding it.
void ExpectMinimal(
    const std::vector<std::pair<std::string, std::string>>& edges) {
  for (const auto& removed : edges) {
    std::vector<std::pair<std::string, std::string>> rest;
    for (const auto& e : edges) {
      if (e != removed) rest.push_back(e);
    }
    const auto reach = Reachability(rest);
    const auto it = reach.find(removed.first);
    const bool still_reachable =
        it != reach.end() && it->second.count(removed.second) > 0;
    EXPECT_FALSE(still_reachable)
        << "edge " << removed.first << " -> " << removed.second
        << " is implied by the remaining edges, so the emitted set is not a "
           "transitive reduction";
  }
}

// The emitted graph must stay acyclic, otherwise the collectives deadlock.
void ExpectAcyclic(
    const std::vector<std::pair<std::string, std::string>>& edges) {
  const auto reach = Reachability(edges);
  for (const auto& [source, targets] : reach) {
    EXPECT_EQ(targets.count(source), 0)
        << "node " << source << " reaches itself";
  }
}

// `n` collectives on one device, none of them ordered by data.  Every pair is a
// candidate, so the dependency graph is the complete DAG on the instance keys
// and its transitive reduction is the descending chain.
std::unique_ptr<Graph> InitIndependentCollectives(int n) {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const std::string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  for (int i = 1; i <= n; ++i) {
    Node* src = ops::SourceOp(
        "TestParams",
        builder.opts().WithName(absl::StrCat("p", i)).WithDevice(dev0));
    CollectiveReduceNode(&builder, src, absl::StrCat("c", i), dev0, i);
  }
  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) LOG(FATAL) << "Error building graph " << s;
  return graph;
}

// `n` collectives strung together by data, so every pair is already ordered and
// no control edge is warranted.  This is the negative control: an
// implementation that manufactures ordering where none is needed is as wrong as
// one that drops it.
std::unique_ptr<Graph> InitDataChainCollectives(int n) {
  GraphDefBuilder builder(GraphDefBuilder::kFailImmediately);
  const std::string dev0 = "/job:localhost/replica:0/task:0/device:CPU:0";
  Node* prev = ops::SourceOp("TestParams",
                             builder.opts().WithName("a").WithDevice(dev0));
  for (int i = 1; i <= n; ++i) {
    prev = CollectiveReduceNode(&builder, prev, absl::StrCat("c", i), dev0, i);
  }
  std::unique_ptr<Graph> graph = std::make_unique<Graph>(OpRegistry::Global());
  absl::Status s = GraphDefBuilderToGraph(builder, graph.get());
  if (!s.ok()) LOG(FATAL) << "Error building graph " << s;
  return graph;
}

// The emitted set must be minimal on the graph that motivated the fix.
TEST(CollectiveOrderTest, DiamondGraphReductionIsMinimal) {
  std::unique_ptr<Graph> graph = InitDiamondGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  const auto edges = CollectiveControlEdges(*graph);
  ExpectMinimal(edges);
  ExpectAcyclic(edges);
}

// Every existing fixture must also satisfy the invariants, not merely match its
// recorded expectation.
TEST(CollectiveOrderTest, ExistingGraphsAreMinimalAndAcyclic) {
  for (auto& factory : {&InitGraph, &InitGraph2, &InitGraphForPruning}) {
    std::unique_ptr<Graph> graph = (*factory)();
    TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
    const auto edges = CollectiveControlEdges(*graph);
    ExpectMinimal(edges);
    ExpectAcyclic(edges);
  }
}

// Ground truth.  The transitive reduction of a complete DAG on a total order is
// the chain of consecutive elements, so `n` unordered collectives must yield
// exactly `n - 1` edges, each stepping down one instance key.
TEST(CollectiveOrderTest, IndependentCollectivesFormDescendingChain) {
  for (int n : {2, 3, 5, 8}) {
    std::unique_ptr<Graph> graph = InitIndependentCollectives(n);
    TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
    const auto edges = CollectiveControlEdges(*graph);

    std::vector<std::pair<std::string, std::string>> expected;
    for (int i = n; i > 1; --i) {
      expected.emplace_back(absl::StrCat("c", i), absl::StrCat("c", i - 1));
    }
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(edges, expected) << "n = " << n;
    ExpectMinimal(edges);
    ExpectAcyclic(edges);
  }
}

// Negative control.  Collectives already ordered by data need no control edges;
// emitting any would over-serialize a graph that was already correct.
TEST(CollectiveOrderTest, DataOrderedCollectivesGetNoControlEdges) {
  for (int n : {2, 4, 6}) {
    std::unique_ptr<Graph> graph = InitDataChainCollectives(n);
    TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
    EXPECT_THAT(CollectiveControlEdges(*graph), ::testing::IsEmpty())
        << "n = " << n;
  }
}

// A lone collective has nothing to be ordered against.
TEST(CollectiveOrderTest, SingleCollectiveEmitsNoEdges) {
  std::unique_ptr<Graph> graph = InitIndependentCollectives(1);
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  EXPECT_THAT(CollectiveControlEdges(*graph), ::testing::IsEmpty());
}

// The emitted set must be a function of the graph alone.  Two structurally
// identical graphs built in one process occupy different heap addresses, so a
// result that depends on `Node*` hashing differs between them.
TEST(CollectiveOrderTest, EmittedEdgesAreIndependentOfNodeAddresses) {
  for (auto& factory : {&InitDiamondGraph, &InitGraph2}) {
    std::unique_ptr<Graph> first = (*factory)();
    std::unique_ptr<Graph> second = (*factory)();
    TF_EXPECT_OK(OrderCollectives(first.get(), GraphCollectiveOrder::kEdges));
    TF_EXPECT_OK(OrderCollectives(second.get(), GraphCollectiveOrder::kEdges));
    EXPECT_EQ(CollectiveControlEdges(*first), CollectiveControlEdges(*second));
  }
  for (int n : {4, 7}) {
    std::unique_ptr<Graph> first = InitIndependentCollectives(n);
    std::unique_ptr<Graph> second = InitIndependentCollectives(n);
    TF_EXPECT_OK(OrderCollectives(first.get(), GraphCollectiveOrder::kEdges));
    TF_EXPECT_OK(OrderCollectives(second.get(), GraphCollectiveOrder::kEdges));
    EXPECT_EQ(CollectiveControlEdges(*first), CollectiveControlEdges(*second));
  }
}

// Ordering is idempotent: re-running over a graph that already carries its
// control edges must not add more.
TEST(CollectiveOrderTest, OrderingTwiceAddsNothing) {
  std::unique_ptr<Graph> graph = InitDiamondGraph();
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  const auto after_first = CollectiveControlEdges(*graph);
  TF_EXPECT_OK(OrderCollectives(graph.get(), GraphCollectiveOrder::kEdges));
  EXPECT_EQ(CollectiveControlEdges(*graph), after_first);
}

}  // namespace
}  // namespace tensorflow
