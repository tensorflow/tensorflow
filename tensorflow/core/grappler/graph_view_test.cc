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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

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
  EXPECT_TRUE(
      OpRegistry::Global()->LookUpOpDef(a_node_def.op(), &a_op_def).ok());
  EXPECT_TRUE(
      OpRegistry::Global()->LookUpOpDef(b_node_def.op(), &b_op_def).ok());

  // Const has 0 inputs, 1 output.
  EXPECT_EQ(-1, OpInputPortIdToArgId(a_node_def, *a_op_def, 0));
  EXPECT_EQ(0, OpOutputPortIdToArgId(a_node_def, *a_op_def, 0));
  EXPECT_EQ(-1, OpOutputPortIdToArgId(a_node_def, *a_op_def, 1));

  // ShapeN has N=3 inputs and outputs.
  EXPECT_EQ(0, OpInputPortIdToArgId(b_node_def, *b_op_def, 0));
  EXPECT_EQ(0, OpInputPortIdToArgId(b_node_def, *b_op_def, 1));
  EXPECT_EQ(0, OpInputPortIdToArgId(b_node_def, *b_op_def, 2));
  EXPECT_EQ(-1, OpInputPortIdToArgId(b_node_def, *b_op_def, 3));
  EXPECT_EQ(0, OpOutputPortIdToArgId(b_node_def, *b_op_def, 0));
  EXPECT_EQ(0, OpOutputPortIdToArgId(b_node_def, *b_op_def, 1));
  EXPECT_EQ(0, OpOutputPortIdToArgId(b_node_def, *b_op_def, 2));
  EXPECT_EQ(-1, OpOutputPortIdToArgId(b_node_def, *b_op_def, 3));
  EXPECT_EQ(-1, OpOutputPortIdToArgId(b_node_def, *b_op_def, 4));
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
    EXPECT_TRUE(
        OpRegistry::Global()->LookUpOpDef(b_node_def.op(), &b_op_def).ok());

    // We have 4 inputs.
    EXPECT_EQ(0, OpInputPortIdToArgId(b_node_def, *b_op_def, 0));
    EXPECT_EQ(1, OpInputPortIdToArgId(b_node_def, *b_op_def, 1));
    EXPECT_EQ(2, OpInputPortIdToArgId(b_node_def, *b_op_def, 2));
    EXPECT_EQ(3, OpInputPortIdToArgId(b_node_def, *b_op_def, 3));
    EXPECT_EQ(-1, OpInputPortIdToArgId(b_node_def, *b_op_def, 4));

    for (int port_id = 0; port_id <= num_splits * 3; ++port_id) {
      int arg_id = -1;
      if (port_id < num_splits * 3) {
        arg_id = port_id / num_splits;
      }
      EXPECT_EQ(arg_id, OpOutputPortIdToArgId(b_node_def, *b_op_def, port_id));
    }
  }
}

TEST_F(GraphViewTest, ParseSingleExample) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const<string>(s.WithOpName("a"), "", {});
  Output b = ops::Const<int64>(s.WithOpName("b"), 1, {1, 1});
  ops::ParseSingleExample c(s.WithOpName("c"), a, {b, b}, 2, {"w", "x"},
                            {"y", "z"}, {DT_INT64, DT_INT64}, {{1}, {1}});

  GraphDef graph_def;
  TF_CHECK_OK(s.ToGraphDef(&graph_def));
  GraphView graph_view(&graph_def);

  const NodeDef& c_node_def = *graph_view.GetNode("c");

  const OpDef* c_op_def = nullptr;
  EXPECT_TRUE(
      OpRegistry::Global()->LookUpOpDef(c_node_def.op(), &c_op_def).ok());

  EXPECT_EQ(0, OpOutputPortIdToArgId(c_node_def, *c_op_def, 0));
  EXPECT_EQ(0, OpOutputPortIdToArgId(c_node_def, *c_op_def, 1));
  EXPECT_EQ(1, OpOutputPortIdToArgId(c_node_def, *c_op_def, 2));
  EXPECT_EQ(1, OpOutputPortIdToArgId(c_node_def, *c_op_def, 3));
  EXPECT_EQ(2, OpOutputPortIdToArgId(c_node_def, *c_op_def, 4));
  EXPECT_EQ(2, OpOutputPortIdToArgId(c_node_def, *c_op_def, 5));
  EXPECT_EQ(3, OpOutputPortIdToArgId(c_node_def, *c_op_def, 6));
  EXPECT_EQ(3, OpOutputPortIdToArgId(c_node_def, *c_op_def, 7));
  EXPECT_EQ(-1, OpOutputPortIdToArgId(c_node_def, *c_op_def, 8));
}

TEST_F(GraphViewTest, BasicGraph) {
  TrivialTestGraphInputYielder fake_input(4, 2, 2, false, {"/CPU:0", "/GPU:0"});
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphView graph(&item.graph);

  GraphView::InputPort input = graph.GetInputPort("AddN", 0);
  EXPECT_EQ("AddN", input.node->name());
  EXPECT_EQ(0, input.port_id);
  GraphView::OutputPort fanin = graph.GetRegularFanin(input);
  EXPECT_EQ("Square", fanin.node->name());
  EXPECT_EQ(0, fanin.port_id);

  input = graph.GetInputPort("AddN", 1);
  EXPECT_EQ("AddN", input.node->name());
  EXPECT_EQ(1, input.port_id);
  fanin = graph.GetRegularFanin(input);
  EXPECT_EQ("Square_1", fanin.node->name());
  EXPECT_EQ(0, fanin.port_id);

  GraphView::OutputPort output = graph.GetOutputPort("AddN", 0);
  EXPECT_EQ("AddN", output.node->name());
  EXPECT_EQ(0, output.port_id);
  EXPECT_EQ(2, graph.GetFanout(output).size());
  for (auto fanout : graph.GetFanout(output)) {
    if (fanout.node->name() == "AddN_2" || fanout.node->name() == "AddN_3") {
      EXPECT_EQ(0, fanout.port_id);
    } else {
      // Invalid fanout
      EXPECT_FALSE(true);
    }
  }

  const NodeDef* add_node = graph.GetNode("AddN");
  EXPECT_NE(nullptr, add_node);

  absl::flat_hash_set<string> fanouts;
  absl::flat_hash_set<string> expected_fanouts = {"AddN_2:0", "AddN_3:0"};
  for (const auto& fo : graph.GetFanouts(*add_node, false)) {
    fanouts.insert(absl::StrCat(fo.node->name(), ":", fo.port_id));
  }
  EXPECT_EQ(graph.NumFanouts(*add_node, false), 2);
  EXPECT_EQ(fanouts, expected_fanouts);

  absl::flat_hash_set<string> fanins;
  absl::flat_hash_set<string> expected_fanins = {"Square_1:0", "Square:0"};
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
  EXPECT_EQ("a", output.node->name());
  EXPECT_EQ(-1, output.port_id);
  auto fanout = graph.GetFanout(output);
  EXPECT_EQ(1, fanout.size());
  EXPECT_EQ("d", (*fanout.begin()).node->name());
  EXPECT_EQ(-1, (*fanout.begin()).port_id);

  output = graph.GetOutputPort("a", 0);
  EXPECT_EQ("a", output.node->name());
  EXPECT_EQ(0, output.port_id);
  fanout = graph.GetFanout(output);
  EXPECT_EQ(1, fanout.size());
  EXPECT_EQ("b", (*fanout.begin()).node->name());
  EXPECT_EQ(0, (*fanout.begin()).port_id);

  GraphView::InputPort input = graph.GetInputPort("d", -1);
  EXPECT_EQ("d", input.node->name());
  EXPECT_EQ(-1, input.port_id);
  auto fanin = graph.GetFanin(input);
  EXPECT_EQ(1, fanin.size());
  EXPECT_EQ("a", (*fanin.begin()).node->name());
  EXPECT_EQ(-1, (*fanin.begin()).port_id);

  input = graph.GetInputPort("d", 0);
  EXPECT_EQ("d", input.node->name());
  EXPECT_EQ(0, input.port_id);
  fanin = graph.GetFanin(input);
  EXPECT_EQ(1, fanin.size());
  EXPECT_EQ("b", (*fanin.begin()).node->name());
  EXPECT_EQ(0, (*fanin.begin()).port_id);

  input = graph.GetInputPort("d", 1);
  EXPECT_EQ("d", input.node->name());
  EXPECT_EQ(1, input.port_id);
  fanin = graph.GetFanin(input);
  EXPECT_EQ(1, fanin.size());
  EXPECT_EQ("c", (*fanin.begin()).node->name());
  EXPECT_EQ(0, (*fanin.begin()).port_id);
}

TEST_F(GraphViewTest, HasNode) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output a = ops::Const(s.WithOpName("a"), 0.0f, {10, 10});

  GrapplerItem item;
  TF_CHECK_OK(s.ToGraphDef(&item.graph));
  GraphView graph(&item.graph);

  EXPECT_EQ(true, graph.HasNode("a"));
  EXPECT_EQ(false, graph.HasNode("b"));
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
  EXPECT_NE(nullptr, d_node);

  EXPECT_EQ(true, graph.HasFanin(*d_node, {"a", Graph::kControlSlot}));
  EXPECT_EQ(false, graph.HasFanin(*d_node, {"a", 0}));
  EXPECT_EQ(true, graph.HasFanin(*d_node, {"b", 0}));
  EXPECT_EQ(false, graph.HasFanin(*d_node, {"b", Graph::kControlSlot}));
  EXPECT_EQ(true, graph.HasFanin(*d_node, {"c", 0}));
  EXPECT_EQ(false, graph.HasFanin(*d_node, {"c", Graph::kControlSlot}));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
