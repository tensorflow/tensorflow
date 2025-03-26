/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/optimize_cross_host_control_deps.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph_to_functiondef.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

Node* GetNodeByName(const string& name, Graph* graph) {
  for (Node* node : graph->op_nodes()) {
    if (node->name() == name) return node;
  }
  return nullptr;
}

TEST(OptimizeCrossHostControlDepsTest, OptimizeCrossHostControlOutputEdges) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  auto a = ops::Const(scope.WithOpName("a"), 1.0f);
  a.node()->set_assigned_device_name("/job:worker/task:0/CPU:0");

  auto b = ops::Const(scope.WithOpName("b").WithControlDependencies(a), 2.0f);
  b.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");
  auto c = ops::Const(scope.WithOpName("c").WithControlDependencies(a), 3.0f);
  c.node()->set_assigned_device_name("/job:worker/task:1/CPU:1");
  auto d = ops::Const(scope.WithOpName("d").WithControlDependencies(a), 4.0f);
  d.node()->set_assigned_device_name("/job:worker/task:1/CPU:2");

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  ASSERT_EQ(graph.num_op_nodes(), 4);

  // No optimizations if the cross_host_edges_threshold is set too high.
  TF_ASSERT_OK(OptimizeCrossHostControlOutputEdges(
      &graph, /*cross_host_edges_threshold=*/10));
  ASSERT_EQ(graph.num_op_nodes(), 4);

  // Check the optimization is performed and control after node is created.
  TF_ASSERT_OK(OptimizeCrossHostControlOutputEdges(
      &graph, /*cross_host_edges_threshold=*/2));
  ASSERT_EQ(graph.num_op_nodes(), 5);

  Node* control_after = GetNodeByName("a/control_after/_0", &graph);
  ASSERT_NE(control_after, nullptr);
  EXPECT_EQ(control_after->op_def().name(), "NoOp");
  EXPECT_EQ(control_after->assigned_device_name(),
            "/job:worker/task:1/device:CPU:0");
}

TEST(OptimizeCrossHostControlDepsTest, OptimizeCrossHostDataOutputEdges) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  auto c1 = ops::Const(scope.WithOpName("c1"), 1.0f);
  auto c2 = ops::Const(scope.WithOpName("c2"), 2.0f);

  // For testing, we need a node with two outputs. The easiest way to do
  // that is by creating (yet another) IdentityN.
  auto a = ops::IdentityN(scope.WithOpName("a"), {c1, c2});
  a.operation.node()->set_assigned_device_name("/job:worker/task:0/CPU:0");

  auto b = ops::Identity(scope.WithOpName("b"), a[0]);
  b.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");
  auto c = ops::Identity(scope.WithOpName("c"), a[1]);
  c.node()->set_assigned_device_name("/job:worker/task:1/CPU:1");
  auto d = ops::Identity(scope.WithOpName("d"), a[0]);
  d.node()->set_assigned_device_name("/job:worker/task:2/CPU:0");
  auto e = ops::Identity(scope.WithOpName("e"), a[1]);
  e.node()->set_assigned_device_name("/job:worker/task:2/CPU:1");

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  ASSERT_EQ(graph.num_op_nodes(), 7);

  // No optimizations if the cross_host_edges_threshold is set too high.
  TF_ASSERT_OK(OptimizeCrossHostDataOutputEdges(
      &graph, /*cross_host_edges_threshold=*/10));
  ASSERT_EQ(graph.num_op_nodes(), 7);

  // Check the optimization is performed.
  TF_ASSERT_OK(OptimizeCrossHostDataOutputEdges(
      &graph, /*cross_host_edges_threshold=*/2));

  ASSERT_EQ(graph.num_op_nodes(), 9);

  Node* data_after1 = GetNodeByName("a/data_after/_0", &graph);
  Node* data_after2 = GetNodeByName("a/data_after/_1", &graph);
  // These two nodes are unordered, so "sort" them by device.
  if (data_after1->assigned_device_name() ==
      "/job:worker/task:2/device:CPU:0") {
    std::swap(data_after1, data_after2);
  }

  ASSERT_NE(data_after1, nullptr);
  EXPECT_EQ(data_after1->op_def().name(), "IdentityN");
  EXPECT_EQ(data_after1->assigned_device_name(),
            "/job:worker/task:1/device:CPU:0");
  EXPECT_EQ(data_after1->def().input_size(), 2);
  EXPECT_EQ(data_after1->def().input(0), "a");
  EXPECT_EQ(data_after1->def().input(1), "a:1");
  EXPECT_EQ(data_after1->op_def().name(), "IdentityN");

  ASSERT_NE(data_after2, nullptr);
  EXPECT_EQ(data_after2->op_def().name(), "IdentityN");
  EXPECT_EQ(data_after2->assigned_device_name(),
            "/job:worker/task:2/device:CPU:0");
  EXPECT_EQ(data_after2->def().input_size(), 2);
  EXPECT_EQ(data_after2->def().input(0), "a");
  EXPECT_EQ(data_after2->def().input(1), "a:1");
  EXPECT_EQ(data_after2->op_def().name(), "IdentityN");

  // Adding edges to the graph doesn't update the nodes, so go
  // through the graph to verify inputs.
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  std::unordered_map<string, const NodeDef*> map;
  for (auto& node : graph_def.node()) {
    map[node.name()] = &node;
  }
  EXPECT_EQ(map["b"]->input(0), data_after1->name());
  EXPECT_EQ(map["c"]->input(0), data_after1->name() + ":1");
  EXPECT_EQ(map["d"]->input(0), data_after2->name());
  EXPECT_EQ(map["e"]->input(0), data_after2->name() + ":1");
}

TEST(OptimizeCrossHostControlDepsTest,
     CreatesIdentityNodesWhenInputsIdentical) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  auto c1 = ops::Const(scope.WithOpName("c1"), 1.0f);
  auto c2 = ops::Const(scope.WithOpName("c2"), 2.0f);
  auto a = ops::IdentityN(scope.WithOpName("a"), {c1, c2});
  a.operation.node()->set_assigned_device_name("/job:worker/task:0/CPU:0");

  auto b = ops::Identity(scope.WithOpName("b"), a[0]);
  auto c = ops::Identity(scope.WithOpName("c"), a[0]);
  auto d = ops::Identity(scope.WithOpName("d"), a[0]);
  auto e = ops::Identity(scope.WithOpName("e"), a[0]);
  b.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");
  c.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");
  d.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");
  e.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  ASSERT_EQ(graph.num_op_nodes(), 7);

  TF_ASSERT_OK(OptimizeCrossHostDataOutputEdges(
      &graph, /*cross_host_edges_threshold=*/2));

  ASSERT_EQ(graph.num_op_nodes(), 8);

  Node* data_after = GetNodeByName("a/data_after/_0", &graph);

  ASSERT_NE(data_after, nullptr);
  EXPECT_EQ(data_after->op_def().name(), "Identity");
  EXPECT_EQ(data_after->assigned_device_name(),
            "/job:worker/task:1/device:CPU:0");
  EXPECT_EQ(data_after->def().input_size(), 1);
  EXPECT_EQ(data_after->def().input(0)[0], 'a');
  EXPECT_EQ(data_after->op_def().name(), "Identity");

  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  std::unordered_map<string, const NodeDef*> map;
  for (auto& node : graph_def.node()) {
    map[node.name()] = &node;
  }
  EXPECT_EQ(map["b"]->input(0), data_after->name());
  EXPECT_EQ(map["c"]->input(0), data_after->name());
  EXPECT_EQ(map["d"]->input(0), data_after->name());
  EXPECT_EQ(map["e"]->input(0), data_after->name());
}

TEST(OptimizeCrossHostControlDepsTest, OptimizeCrossHostControlInputEdges) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();
  auto a = ops::Const(scope.WithOpName("a"), 1.0f);
  a.node()->set_assigned_device_name("/job:worker/task:0/CPU:0");
  auto b = ops::Const(scope.WithOpName("b"), 2.0f);
  b.node()->set_assigned_device_name("/job:worker/task:0/CPU:1");
  auto c = ops::Const(scope.WithOpName("c"), 1.0f);
  c.node()->set_assigned_device_name("/job:worker/task:0/CPU:2");

  auto d = ops::Const(
      scope.WithOpName("d").WithControlDependencies({a.op(), b.op(), c.op()}),
      4.0f);
  d.node()->set_assigned_device_name("/job:worker/task:1/CPU:0");

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  ASSERT_EQ(graph.num_op_nodes(), 4);

  // No optimizations if the cross_host_edges_threshold is set too high.
  TF_ASSERT_OK(OptimizeCrossHostControlOutputEdges(
      &graph, /*cross_host_edges_threshold=*/10));
  ASSERT_EQ(graph.num_op_nodes(), 4);

  // Check the optimization is performed and control before node is created.
  TF_ASSERT_OK(OptimizeCrossHostControlInputEdges(
      &graph, /*cross_host_edges_threshold=*/2));
  ASSERT_EQ(graph.num_op_nodes(), 5);

  Node* control_before = GetNodeByName("d/control_before/_0", &graph);
  ASSERT_NE(control_before, nullptr);
  EXPECT_EQ(control_before->op_def().name(), "NoOp");
  EXPECT_EQ(control_before->assigned_device_name(),
            "/job:worker/task:0/device:CPU:0");
}

TEST(OptimizeCrossHostControlDepsTest, LargeGraph) {
  tensorflow::Scope scope = tensorflow::Scope::NewRootScope();

  constexpr int size = 750;

  std::vector<Operation> layer1;
  for (int i = 0; i < size; ++i) {
    auto n = ops::Const(scope, 1.0f);
    n.node()->set_assigned_device_name("/job:worker/task:0/CPU:0");
    layer1.push_back(n.op());
  }

  for (int j = 0; j < size; ++j) {
    auto d = ops::Const(scope.WithControlDependencies(layer1), 1.0f);
    d.node()->set_assigned_device_name("/job:worker/task:0/CPU:0");
  }

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(scope.ToGraph(&graph));
  ASSERT_EQ(graph.num_op_nodes(), size * 2);

  TF_ASSERT_OK(OptimizeCrossHostControlInputEdges(
      &graph, /*cross_host_edges_threshold=*/size));
  TF_ASSERT_OK(OptimizeCrossHostControlOutputEdges(
      &graph, /*cross_host_edges_threshold=*/size));
  ASSERT_EQ(graph.num_op_nodes(), size * 4);
}

}  // namespace
}  // namespace tensorflow
