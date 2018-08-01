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

TEST(MutableGraphViewTest, AddAndReplaceInput) {
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
  TrivialTestGraphInputYielder fake_input(2, 2, 2, false, {"/CPU:0", "/GPU:0"});
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

  auto find_child_with_name = [&graph](string output_port_name,
                                       string input_name) {
    GraphView::OutputPort output_port =
        graph.GetOutputPort(output_port_name, 0);
    auto fanout = graph.GetFanout(output_port);
    for (auto& input_port : fanout) {
      if (input_port.node->name() == input_name) return true;
    }
    return false;
  };

  EXPECT_FALSE(find_child_with_name("Square", "new_node"));

  NodeDef new_node = *input.node;
  new_node.set_name("new_node");

  EXPECT_EQ(graph.GetNode("new_node"), nullptr);
  NodeDef* node_in_graph = graph.AddNode(std::move(new_node));
  EXPECT_NE(graph.GetNode("new_node"), nullptr);

  graph.ReplaceInput(*input.node, *node_in_graph);
  EXPECT_TRUE(find_child_with_name("Square", "new_node"));
  EXPECT_TRUE(find_child_with_name("new_node", "y"));
}

TEST(MutableGraphViewTest, DeleteNodes) {
  // Outputs simple graph as described in first test.
  TrivialTestGraphInputYielder fake_input(2, 2, 2, false, {"/CPU:0", "/GPU:0"});
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
