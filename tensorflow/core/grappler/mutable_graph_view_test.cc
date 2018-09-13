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

#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

bool FindChildWithName(const MutableGraphView& graph,
                       const string& output_port_name,
                       const string& input_name) {
  GraphView::OutputPort output_port = graph.GetOutputPort(output_port_name, 0);
  auto fanout = graph.GetFanout(output_port);
  for (auto& input_port : fanout) {
    if (input_port.node->name() == input_name) return true;
  }
  return false;
}

TrivialTestGraphInputYielder SimpleGraph() {
  // This outputs simple graph like:
  //        x
  //       / \
  // Square   Square_1
  //   |   \  /    |
  //   |    \/     |
  //   |    /\     |
  //   |   /  \    |
  //  AddN     AddN_1
  //      \   /
  //        y
  TrivialTestGraphInputYielder simple_graph(2, 2, 2, false,
                                            {"/CPU:0", "/GPU:0"});
  return simple_graph;
}

TEST(MutableGraphViewTest, AddAndReplaceInput) {
  TrivialTestGraphInputYielder fake_input = SimpleGraph();
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphDef new_graph = item.graph;
  MutableGraphView graph(&new_graph);

  GraphView::InputPort input = graph.GetInputPort("AddN", 0);
  EXPECT_EQ("AddN", input.node->name());
  EXPECT_EQ(0, input.port_id);
  GraphView::OutputPort fanin = graph.GetRegularFanin(input);
  EXPECT_EQ("Square", fanin.node->name());
  EXPECT_EQ(0, fanin.port_id);

  EXPECT_FALSE(FindChildWithName(graph, "Square", "new_node"));

  NodeDef new_node = *input.node;
  new_node.set_name("new_node");

  EXPECT_EQ(graph.GetNode("new_node"), nullptr);
  NodeDef* node_in_graph = graph.AddNode(std::move(new_node));
  EXPECT_NE(graph.GetNode("new_node"), nullptr);

  graph.ReplaceInput(*input.node, *node_in_graph);
  EXPECT_TRUE(FindChildWithName(graph, "Square", "new_node"));
  EXPECT_TRUE(FindChildWithName(graph, "new_node", "y"));
}

TEST(MutableGraphViewTest, InsertNodes) {
  TrivialTestGraphInputYielder fake_input = SimpleGraph();

  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphDef new_graph = item.graph;
  MutableGraphView graph(&new_graph);

  GraphView::InputPort input = graph.GetInputPort("AddN", 0);

  NodeDef new_node = *input.node;
  new_node.set_name("new_node");
  new_node.set_input(0, input.node->name());

  EXPECT_EQ(graph.GetNode("new_node"), nullptr);
  graph.InsertNode(*input.node, std::move(new_node));
  EXPECT_NE(graph.GetNode("new_node"), nullptr);
  EXPECT_TRUE(FindChildWithName(graph, "Square", "AddN"));
  EXPECT_TRUE(FindChildWithName(graph, "Square", "AddN_1"));
  EXPECT_TRUE(FindChildWithName(graph, "Square_1", "AddN"));
  EXPECT_TRUE(FindChildWithName(graph, "Square_1", "AddN_1"));
  EXPECT_TRUE(FindChildWithName(graph, "AddN", "new_node"));
  EXPECT_TRUE(FindChildWithName(graph, "AddN_1", "y"));
  EXPECT_TRUE(FindChildWithName(graph, "new_node", "y"));
}

TEST(MutableGraphViewTest, DeleteNodes) {
  // Outputs simple graph as described in first test.
  TrivialTestGraphInputYielder fake_input = SimpleGraph();
  GrapplerItem item;
  CHECK(fake_input.NextItem(&item));

  GraphDef new_graph = item.graph;
  MutableGraphView graph(&new_graph);

  EXPECT_NE(graph.GetNode("AddN"), nullptr);
  graph.DeleteNodes({"AddN"});

  EXPECT_EQ(graph.GetNode("AddN"), nullptr);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
