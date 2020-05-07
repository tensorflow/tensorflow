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

#include "tensorflow/core/grappler/graph_view.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/cc/ops/parsing_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

namespace tensorflow {
namespace grappler {
namespace {

class GraphViewTest : public ::testing::Test {};

TEST_F(GraphViewTest, OpPortIdToArgIdShapeN) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  ops::ShapeN b(s.WithOpName("b"), {a, a, a});

  GraphDef graph_def;
  TF_CHECK_OK(s.ToGraphDef(&graph_def));
  GraphView graph_view(&graph_def);

  const NodeDef& a_node_def = *graph_view.GetNode("a");
  const NodeDef& b_node_def = *graph_view.GetNode("b");

  const OpDef* a_op_def = nullptr;
  const OpDef* b_op_def = nullptr;
  TF_EXPECT_OK(OpRegistry::Global()->LookUpOpDef(a_node_def.op(), &a_op_def));
  TF_EXPECT_OK(OpRegistry::Global()->LookUpOpDef(b_node_def.op(), &b_op_def));

  // Const has 0 inputs, 1 output.
  EXPECT_EQ(OpInputPortIdToArgId(a_node_def, *a_op_def, 0), -1);
  EXPECT_EQ(OpOutputPortIdToArgId(a_node_def, *a_op_def, 0), 0);
  EXPECT_EQ(OpOutputPortIdToArgId(a_node_def, *a_op_def, 1), -1);

  // ShapeN has N=3 inputs and outputs.
  EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 0), 0);
  EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 1), 0);
  EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 2), 0);
  EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 3), -1);
  EXPECT_EQ(OpOutputPortIdToArgId(b_node_def, *b_op_def, 0), 0);
  EXPECT_EQ(OpOutputPortIdToArgId(b_node_def, *b_op_def, 1), 0);
  EXPECT_EQ(OpOutputPortIdToArgId(b_node_def, *b_op_def, 2), 0);
  EXPECT_EQ(OpOutputPortIdToArgId(b_node_def, *b_op_def, 3), -1);
  EXPECT_EQ(OpOutputPortIdToArgId(b_node_def, *b_op_def, 4), -1);
}

TEST_F(GraphViewTest, OpPortIdToArgIdSparseSplit) {
  for (int num_splits : {1, 2}) {
    tensorflow::Scope s = tensorflow::Scope::NewRootScope();
    Output a = ops::Const<int64>(s.WithOpName("a"), 1, {10, 10});
    ops::SparseSplit b(s.WithOpName("b"), a, a, a, a, num_splits);

    GraphDef graph_def;
    TF_CHECK_OK(s.ToGraphDef(&graph_def));
    GraphView graph_view(&graph_def);

    const NodeDef& b_node_def = *graph_view.GetNode("b");
    const OpDef* b_op_def = nullptr;
    TF_EXPECT_OK(OpRegistry::Global()->LookUpOpDef(b_node_def.op(), &b_op_def));

    // We have 4 inputs.
    EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 0), 0);
    EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 1), 1);
    EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 2), 2);
    EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 3), 3);
    EXPECT_EQ(OpInputPortIdToArgId(b_node_def, *b_op_def, 4), -1);

    for (int port_id = 0; port_id <= num_splits * 3; ++port_id) {
      int arg_id = -1;
      if (port_id < num_splits * 3) {
        arg_id = port_id / num_splits;
      }
      EXPECT_EQ(OpOutputPortIdToArgId(b_node_def, *b_op_def, port_id), arg_id);
    }
  }
}

TEST_F(GraphViewTest, ParseSingleExample) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const<tstring>(s.WithOpName("a"), "", {});
  Output b = ops::Const<int64>(s.WithOpName("b"), 1, {1, 1});
  ops::ParseSingleExample c(s.WithOpName("c"), a, {b, b}, 2, {"w", "x"},
                            {"y", "z"}, {DT_INT64, DT_INT64}, {{1}, {1}});

  GraphDef graph_def;
  TF_CHECK_OK(s.ToGraphDef(&graph_def));
  GraphView graph_view(&graph_def);

  const NodeDef& c_node_def = *graph_view.GetNode("c");

  const OpDef* c_op_def = nullptr;
  TF_EXPECT_OK(OpRegistry::Global()->LookUpOpDef(c_node_def.op(), &c_op_def));

  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 0), 0);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 1), 0);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 2), 1);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 3), 1);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 4), 2);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 5), 2);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 6), 3);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 7), 3);
  EXPECT_EQ(OpOutputPortIdToArgId(c_node_def, *c_op_def, 8), -1);
}

TEST_F(GraphViewTest, BasicGraph) {
  TrivialTestGraphInputYielder fake_input(4, 2, 2, false, {"/CPU:0", "/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphView graph(&item.graph);

  GraphView::InputPort input = graph.GetInputPort("AddN", 0);
  EXPECT_EQ(input.node->name(), "AddN");
  EXPECT_EQ(input.port_id, 0);
  GraphView::OutputPort fanin = graph.GetRegularFanin(input);
  EXPECT_EQ(fanin.node->name(), "Sign");
  EXPECT_EQ(fanin.port_id, 0);

  input = graph.GetInputPort("AddN", 1);
  EXPECT_EQ(input.node->name(), "AddN");
  EXPECT_EQ(input.port_id, 1);
  fanin = graph.GetRegularFanin(input);
  EXPECT_EQ(fanin.node->name(), "Sign_1");
  EXPECT_EQ(fanin.port_id, 0);

  GraphView::OutputPort output = graph.GetOutputPort("AddN", 0);
  EXPECT_EQ(output.node->name(), "AddN");
  EXPECT_EQ(output.port_id, 0);
  EXPECT_EQ(graph.GetFanout(output).size(), 2);
  for (auto fanout : graph.GetFanout(output)) {
    if (fanout.node->name() == "AddN_2" || fanout.node->name() == "AddN_3") {
      EXPECT_EQ(fanout.port_id, 0);
    } else {
      // Invalid fanout
      EXPECT_FALSE(true);
    }
  }

  const NodeDef* add_node = graph.GetNode("AddN");
  EXPECT_NE(add_node, nullptr);

  absl::flat_hash_set<string> fanouts;
  absl::flat_hash_set<string> expected_fanouts = {"AddN_2:0", "AddN_3:0"};
  for (const auto& fo : graph.GetFanouts(*add_node, false)) {
    fanouts.insert(absl::StrCat(fo.node->name(), ":", fo.port_id));
  }
  EXPECT_EQ(graph.NumFanouts(*add_node, false), 2);
  EXPECT_EQ(fanouts, expected_fanouts);

  absl::flat_hash_set<string> fanins;
  absl::flat_hash_set<string> expected_fanins = {"Sign_1:0", "Sign:0"};
  for (const auto& fi : graph.GetFanins(*add_node, false)) {
    fanins.insert(absl::StrCat(fi.node->name(), ":", fi.port_id));
  }
  EXPECT_EQ(graph.NumFanins(*add_node, false), 2);
  EXPECT_EQ(fanins, expected_fanins);
}

TEST_F(GraphViewTest, ControlDependencies) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::Square(s.WithOpName("b"), {a});
  Output c = ops::Sqrt(s.WithOpName("c"), {b});
  Output d = ops::AddN(s.WithOpName("d").WithControlDependencies(a), {b, c});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphView graph(&item.graph);

  GraphView::OutputPort output = graph.GetOutputPort("a", -1);
  EXPECT_EQ(output.node->name(), "a");
  EXPECT_EQ(output.port_id, -1);
  auto fanout = graph.GetFanout(output);
  EXPECT_EQ(fanout.size(), 1);
  EXPECT_EQ((*fanout.begin()).node->name(), "d");
  EXPECT_EQ((*fanout.begin()).port_id, -1);

  output = graph.GetOutputPort("a", 0);
  EXPECT_EQ(output.node->name(), "a");
  EXPECT_EQ(output.port_id, 0);
  fanout = graph.GetFanout(output);
  EXPECT_EQ(fanout.size(), 1);
  EXPECT_EQ((*fanout.begin()).node->name(), "b");
  EXPECT_EQ((*fanout.begin()).port_id, 0);

  GraphView::InputPort input = graph.GetInputPort("d", -1);
  EXPECT_EQ(input.node->name(), "d");
  EXPECT_EQ(input.port_id, -1);
  auto fanin = graph.GetFanin(input);
  EXPECT_EQ(fanin.size(), 1);
  EXPECT_EQ((*fanin.begin()).node->name(), "a");
  EXPECT_EQ((*fanin.begin()).port_id, -1);

  input = graph.GetInputPort("d", 0);
  EXPECT_EQ(input.node->name(), "d");
  EXPECT_EQ(input.port_id, 0);
  fanin = graph.GetFanin(input);
  EXPECT_EQ(fanin.size(), 1);
  EXPECT_EQ((*fanin.begin()).node->name(), "b");
  EXPECT_EQ((*fanin.begin()).port_id, 0);

  input = graph.GetInputPort("d", 1);
  EXPECT_EQ(input.node->name(), "d");
  EXPECT_EQ(input.port_id, 1);
  fanin = graph.GetFanin(input);
  EXPECT_EQ(fanin.size(), 1);
  EXPECT_EQ((*fanin.begin()).node->name(), "c");
  EXPECT_EQ((*fanin.begin()).port_id, 0);
}

TEST_F(GraphViewTest, HasNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphView graph(&item.graph);

  EXPECT_EQ(graph.HasNode("a"), true);
  EXPECT_EQ(graph.HasNode("b"), false);
}

TEST_F(GraphViewTest, HasFanin) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::Square(s.WithOpName("b"), {a});
  Output c = ops::Sqrt(s.WithOpName("c"), {b});
  Output d = ops::AddN(s.WithOpName("d").WithControlDependencies(a), {b, c});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphView graph(&item.graph);

  const NodeDef* d_node = graph.GetNode("d");
  EXPECT_NE(d_node, nullptr);

  EXPECT_EQ(graph.HasFanin(*d_node, {"a", Graph::kControlSlot}), true);
  EXPECT_EQ(graph.HasFanin(*d_node, {"a", 0}), false);
  EXPECT_EQ(graph.HasFanin(*d_node, {"b", 0}), true);
  EXPECT_EQ(graph.HasFanin(*d_node, {"b", Graph::kControlSlot}), false);
  EXPECT_EQ(graph.HasFanin(*d_node, {"c", 0}), true);
  EXPECT_EQ(graph.HasFanin(*d_node, {"c", Graph::kControlSlot}), false);
}

TEST_F(GraphViewTest, GetRegularFaninPortOutOfBounds) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});
  Output b = ops::Square(s.WithOpName("b"), {});
  Output c = ops::Sqrt(s.WithOpName("c"), {b});
  Output d = ops::AddN(s.WithOpName("d").WithControlDependencies(a), {b, c});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphView graph(&item.graph);

  const NodeDef* b_node = graph.GetNode("b");
  EXPECT_NE(b_node, nullptr);
  const NodeDef* c_node = graph.GetNode("c");
  EXPECT_NE(c_node, nullptr);
  const NodeDef* d_node = graph.GetNode("d");
  EXPECT_NE(d_node, nullptr);

  auto d_output_0 = graph.GetRegularFanin({d_node, 0});
  EXPECT_EQ(d_output_0, GraphView::OutputPort(b_node, 0));
  auto d_output_1 = graph.GetRegularFanin({d_node, 1});
  EXPECT_EQ(d_output_1, GraphView::OutputPort(c_node, 0));
  auto d_output_2 = graph.GetRegularFanin({d_node, 2});
  EXPECT_EQ(d_output_2, GraphView::OutputPort());
  auto d_output_control = graph.GetRegularFanin({d_node, Graph::kControlSlot});
  EXPECT_EQ(d_output_control, GraphView::OutputPort());
}

static void BM_GraphViewConstruction(int iters, int num_nodes,
                                     int num_edges_per_node) {
  testing::StopTiming();
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, num_edges_per_node);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    GraphView graph_view(&graph_def);
  }
  testing::StopTiming();
}

BENCHMARK(BM_GraphViewConstruction)
    ->ArgPair(10, 2)
    ->ArgPair(100, 2)
    ->ArgPair(1000, 2)
    ->ArgPair(10000, 2)
    ->ArgPair(25000, 2)
    ->ArgPair(50000, 2)
    ->ArgPair(100000, 2)
    ->ArgPair(10, 4)
    ->ArgPair(100, 4)
    ->ArgPair(1000, 4)
    ->ArgPair(10000, 4)
    ->ArgPair(25000, 4)
    ->ArgPair(50000, 4)
    ->ArgPair(100000, 4)
    ->ArgPair(10, 8)
    ->ArgPair(100, 8)
    ->ArgPair(1000, 8)
    ->ArgPair(10000, 8)
    ->ArgPair(25000, 8)
    ->ArgPair(50000, 8)
    ->ArgPair(100000, 8)
    ->ArgPair(10, 16)
    ->ArgPair(100, 16)
    ->ArgPair(1000, 16)
    ->ArgPair(10000, 16)
    ->ArgPair(25000, 16)
    ->ArgPair(50000, 16)
    ->ArgPair(100000, 16);

static void BM_GraphViewGetNode(int iters, int num_nodes) {
  testing::StopTiming();
  const GraphDef graph_def =
      test::CreateGraphDef(num_nodes, /*num_edges_per_node=*/16);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    graph_view.GetNode("out");
  }
  testing::StopTiming();
}

BENCHMARK(BM_GraphViewGetNode)
    ->Arg(10)
    ->Arg(100)
    ->Arg(1000)
    ->Arg(10000)
    ->Arg(25000)
    ->Arg(50000)
    ->Arg(100000);

#define RUN_FANIN_FANOUT_BENCHMARK(name) \
  BENCHMARK(name)                        \
      ->ArgPair(10, 10)                  \
      ->ArgPair(10, 100)                 \
      ->ArgPair(10, 1000)                \
      ->ArgPair(10, 10000)               \
      ->ArgPair(10, 100000)              \
      ->ArgPair(100, 10)                 \
      ->ArgPair(100, 100)                \
      ->ArgPair(100, 1000)               \
      ->ArgPair(100, 10000)              \
      ->ArgPair(100, 100000)             \
      ->ArgPair(1000, 10)                \
      ->ArgPair(1000, 100)               \
      ->ArgPair(1000, 1000)              \
      ->ArgPair(1000, 10000)             \
      ->ArgPair(1000, 100000)            \
      ->ArgPair(10000, 10)               \
      ->ArgPair(10000, 100)              \
      ->ArgPair(10000, 1000)             \
      ->ArgPair(10000, 10000)            \
      ->ArgPair(10000, 100000)           \
      ->ArgPair(100000, 10)              \
      ->ArgPair(100000, 100)             \
      ->ArgPair(100000, 1000)            \
      ->ArgPair(100000, 10000)           \
      ->ArgPair(100000, 100000);

static void BM_GraphViewGetFanout(int iters, int num_fanins, int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetFanout({node, 0});
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetFanout);

static void BM_GraphViewGetFanin(int iters, int num_fanins, int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetFanin({node, 0});
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetFanin);

static void BM_GraphViewGetRegularFanin(int iters, int num_fanins,
                                        int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetRegularFanin({node, 0});
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetRegularFanin);

static void BM_GraphViewGetFanouts(int iters, int num_fanins, int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetFanouts(*node, /*include_controlled_nodes=*/false);
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetFanouts);

static void BM_GraphViewGetFanins(int iters, int num_fanins, int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetFanins(*node, /*include_controlling_nodes=*/false);
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetFanins);

static void BM_GraphViewGetFanoutEdges(int iters, int num_fanins,
                                       int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetFanoutEdges(*node, /*include_controlled_edges=*/false);
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetFanoutEdges);

static void BM_GraphViewGetFaninEdges(int iters, int num_fanins,
                                      int num_fanouts) {
  testing::StopTiming();
  const GraphDef graph_def = test::CreateFaninFanoutNodeGraph(
      num_fanins, num_fanouts, num_fanins, num_fanouts,
      /*fanout_unique_index=*/true);
  GraphView graph_view(&graph_def);

  testing::StartTiming();
  for (int i = 0; i < iters; ++i) {
    const NodeDef* node = graph_view.GetNode("node");
    graph_view.GetFaninEdges(*node, /*include_controlling_edges=*/false);
  }
  testing::StopTiming();
}

RUN_FANIN_FANOUT_BENCHMARK(BM_GraphViewGetFaninEdges);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
