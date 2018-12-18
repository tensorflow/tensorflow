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
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::test::function::NDef;

TEST(MutableGraphViewTest, AddAndUpdateFanouts) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("bar", "NotImportant", {}, {}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar", "other", "bar:1", "^bar"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar:2", "^bar"})},
      /* empty function library */ {});

  MutableGraphView graph(&graph_def);

  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "NotImportant", {}, {}));
  NodeDef* bar = graph.GetNode("bar");

  graph.UpdateFanouts(bar->name(), new_bar->name());

  // Fanout nodes must have their inputs updated.
  NodeDef* foo_1 = graph.GetNode("foo_1");
  ASSERT_NE(foo_1, nullptr);
  ASSERT_EQ(foo_1->input_size(), 4);
  EXPECT_EQ(foo_1->input(0), "new_bar");
  EXPECT_EQ(foo_1->input(1), "other");
  EXPECT_EQ(foo_1->input(2), "new_bar:1");
  EXPECT_EQ(foo_1->input(3), "^new_bar");

  NodeDef* foo_2 = graph.GetNode("foo_2");
  ASSERT_NE(foo_2, nullptr);
  ASSERT_EQ(foo_2->input_size(), 3);
  EXPECT_EQ(foo_2->input(0), "other:1");
  EXPECT_EQ(foo_2->input(1), "new_bar:2");
  EXPECT_EQ(foo_2->input(2), "^new_bar");

  // And fanouts mapping must be also updated for both nodes.
  bool include_control_fanouts = true;
  auto old_node_fanouts = graph.GetFanouts(*bar, include_control_fanouts);
  auto new_node_fanouts = graph.GetFanouts(*new_bar, include_control_fanouts);

  EXPECT_TRUE(old_node_fanouts.empty());
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_1, 0)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_1, 2)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_1, -1)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_2, 1)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_2, -1)), 1);
}

TEST(MutableGraphViewTest, AddAndUpdateFanoutsWithoutSelfLoops) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def =
      test::function::GDef({NDef("bar", "NotImportant", {}, {}),
                            NDef("foo", "NotImportant", {"bar", "^bar"})},
                           /* empty function library */ {});

  MutableGraphView graph(&graph_def);

  // `new_bar` reads the output of an original `bar` node.
  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "NewBar", {"bar"}, {}));
  NodeDef* bar = graph.GetNode("bar");

  graph.UpdateFanouts("bar", new_bar->name());

  // Foo node must read from `new_bar`.
  NodeDef* foo = graph.GetNode("foo");
  ASSERT_NE(foo, nullptr);
  ASSERT_EQ(foo->input_size(), 2);
  EXPECT_EQ(foo->input(0), "new_bar");
  EXPECT_EQ(foo->input(1), "^new_bar");

  // And the `new_bar` should read from the original `bar`.
  ASSERT_EQ(new_bar->input_size(), 1);
  ASSERT_EQ(new_bar->input(0), "bar");

  // And fanouts mapping must be also updated for both nodes.
  bool include_control_fanouts = true;
  auto bar_fanouts = graph.GetFanouts(*bar, include_control_fanouts);
  auto new_bar_fanouts = graph.GetFanouts(*new_bar, include_control_fanouts);

  EXPECT_EQ(bar_fanouts.size(), 1);
  EXPECT_EQ(bar_fanouts.count(MutableGraphView::InputPort(new_bar, 0)), 1);

  EXPECT_EQ(new_bar_fanouts.size(), 2);
  EXPECT_EQ(new_bar_fanouts.count(MutableGraphView::InputPort(foo, 0)), 1);
  EXPECT_EQ(new_bar_fanouts.count(MutableGraphView::InputPort(foo, -1)), 1);
}

GraphDef SimpleMutateFaninGraph() {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {}, {}), NDef("d", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"a"}),
       NDef("foo_2", "NotImportant", {"b", "^a", "^c"}),
       NDef("foo_3", "NotImportant", {"b", "a:1", "a:1"}),
       NDef("foo_4", "NotImportant", {"a", "b:2", "b:2", "^c", "^d"}),
       NDef("foo_5", "NotImportant", {}),
       NDef("foo_6", "NotImportant", {"^a", "^b"})},
      /*funcs=*/{});
  return graph_def;
}

void CompareNodeInputs(const MutableGraphView& graph, const NodeDef* expected,
                       NodeDef* actual) {
  ASSERT_EQ(actual->input_size(), expected->input_size());
  int port;
  for (int i = 0; i < actual->input_size(); ++i) {
    EXPECT_EQ(actual->input(i), expected->input(i));
    TensorId tensor_id = ParseTensorName(expected->input(i));
    if (tensor_id.index() == Graph::kControlSlot) {
      port = Graph::kControlSlot;
    } else {
      port = i;
    }
    MutableGraphView::InputPort input_port(actual, port);
    MutableGraphView::OutputPort output_port =
        graph.GetOutputPort(tensor_id.node(), tensor_id.index());
    EXPECT_EQ(graph.GetFanin(input_port).contains(output_port), true);
    EXPECT_EQ(graph.GetFanout(output_port).contains(input_port), true);
  }
}

void TestAddFanin(absl::string_view node_name, const TensorId& fanin_to_add,
                  bool modified, const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  if (expected_node == nullptr) {
    EXPECT_EQ(node, nullptr);
  } else {
    EXPECT_NE(node, nullptr);
  }

  EXPECT_EQ(modified, graph.AddFanin(node_name, fanin_to_add));
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
  }
}

TEST(MutableGraphViewTest, AddFanin) {
  NodeDef expected_node;
  // Add input to node with 1 input 0 controls.
  expected_node = NDef("", "", {"a", "b:1"});
  TestAddFanin("foo_1", {"b", 1}, /*modified=*/true, &expected_node);
  // Add input to node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b", "a:1", "a:1", "b:2"});
  TestAddFanin("foo_3", {"b", 2}, /*modified=*/true, &expected_node);
  // Add input to node with 1 input multiple controls.
  expected_node = NDef("", "", {"b", "a", "^c", "^a"});
  TestAddFanin("foo_2", {"a", 0}, /*modified=*/true, &expected_node);
  // Add input to node with multiple inputs and controls.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "a:1", "^d", "^c"});
  TestAddFanin("foo_4", {"a", 1}, /*modified=*/true, &expected_node);
  // Add input to node with 0 inputs 0 controls.
  expected_node = NDef("", "", {"a:1"});
  TestAddFanin("foo_5", {"a", 1}, /*modified=*/true, &expected_node);
  // Add input to node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"c:1", "^b", "^a"});
  TestAddFanin("foo_6", {"c", 1}, /*modified=*/true, &expected_node);

  // Add control to node with 1 input 0 controls.
  expected_node = NDef("", "", {"a", "^b"});
  TestAddFanin("foo_1", {"b", Graph::kControlSlot}, /*modified=*/true,
               &expected_node);
  // Add control to node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b", "a:1", "a:1", "^c"});
  TestAddFanin("foo_3", {"c", Graph::kControlSlot}, /*modified=*/true,
               &expected_node);
  // Add control to node with 1 input multiple controls.
  expected_node = NDef("", "", {"b", "^a", "^c", "^d"});
  TestAddFanin("foo_2", {"d", Graph::kControlSlot}, /*modified=*/true,
               &expected_node);
  // Add control to node with multiple input multiple controls.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^c", "^d", "^a"});
  TestAddFanin("foo_4", {"a", Graph::kControlSlot}, /*modified=*/true,
               &expected_node);
  // Add control to node with 0 inputs 0 controls.
  expected_node = NDef("", "", {"^a"});
  TestAddFanin("foo_5", {"a", Graph::kControlSlot}, /*modified=*/true,
               &expected_node);
  // Add control to node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"^a", "^b", "^c"});
  TestAddFanin("foo_6", {"c", Graph::kControlSlot}, /*modified=*/true,
               &expected_node);
  // Add control to node with control that already exists.
  expected_node = NDef("", "", {"b", "^a", "^c"});
  TestAddFanin("foo_2", {"a", Graph::kControlSlot}, /*modified=*/false,
               &expected_node);

  // Add fanin to node where node is missing.
  TestAddFanin("foo_missing", {"a", 0}, /*modified=*/false, nullptr);
  // Add fanin to node where fanin is missing.
  expected_node = NDef("", "", {"a"});
  TestAddFanin("foo_1", {"bar_missing", 0}, /*modified=*/false, &expected_node);
  // Add fanin to node where node and fanin are missing.
  TestAddFanin("foo_missing", {"bar_missing", 0}, /*modified=*/false,
               /*expected_node=*/nullptr);
}

void CheckFanout(const MutableGraphView& graph, const TensorId& fanin,
                 absl::string_view node_name) {
  MutableGraphView::OutputPort output_port =
      graph.GetOutputPort(fanin.node(), fanin.index());
  auto fanouts = graph.GetFanout(output_port);
  for (auto fanout : fanouts) {
    EXPECT_NE(fanout.node->name(), fanin.node());
  }
}

void TestRemoveFanin(absl::string_view node_name,
                     const TensorId& fanin_to_remove, bool modified,
                     const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  if (expected_node == nullptr) {
    EXPECT_EQ(nullptr, node);
  } else {
    EXPECT_NE(nullptr, node);
  }

  EXPECT_EQ(modified, graph.RemoveFanin(node_name, fanin_to_remove));
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
    if (modified) {
      CheckFanout(graph, fanin_to_remove, node_name);
    }
  }
}

TEST(MutableGraphViewTest, RemoveFanin) {
  NodeDef expected_node;
  // Remove input from node with 1 input 0 controls.
  expected_node = NDef("", "", {});
  TestRemoveFanin("foo_1", {"a", 0}, /*modified=*/true, &expected_node);
  // Remove input from node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b"});
  TestRemoveFanin("foo_3", {"a", 1}, /*modified=*/true, &expected_node);
  // Remove input from node with 1 input multiple controls.
  expected_node = NDef("", "", {"^a", "^c"});
  TestRemoveFanin("foo_2", {"b", 0}, /*modified=*/true, &expected_node);
  // Remove input from node with multiple inputs and controls.
  expected_node = NDef("", "", {"a", "^c", "^d"});
  TestRemoveFanin("foo_4", {"b", 2}, /*modified=*/true, &expected_node);

  // Remove control from node with 1 input multiple controls.
  expected_node = NDef("", "", {"b", "^c"});
  TestRemoveFanin("foo_2", {"a", Graph::kControlSlot}, /*modified=*/true,
                  &expected_node);
  // Remove control from node with multiple input multiple controls.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^c"});
  TestRemoveFanin("foo_4", {"d", Graph::kControlSlot}, /*modified=*/true,
                  &expected_node);
  // Remove control from node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"^b"});
  TestRemoveFanin("foo_6", {"a", Graph::kControlSlot}, /*modified=*/true,
                  &expected_node);

  // Remove input from node with 0 inputs 0 controls.
  expected_node = NDef("", "", {});
  TestRemoveFanin("foo_5", {"a", 1}, /*modified=*/false, &expected_node);
  // Remove input from node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"^a", "^b"});
  TestRemoveFanin("foo_6", {"a", 1}, /*modified=*/false, &expected_node);
  // Remove control from node with 1 input 0 controls.
  expected_node = NDef("", "", {"a"});
  TestRemoveFanin("foo_1", {"b", Graph::kControlSlot}, /*modified=*/false,
                  &expected_node);
  // Remove control from node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b", "a:1", "a:1"});
  TestRemoveFanin("foo_3", {"c", Graph::kControlSlot}, /*modified=*/false,
                  &expected_node);
  // Remove control from node with 0 inputs 0 controls.
  expected_node = NDef("", "", {});
  TestRemoveFanin("foo_5", {"a", Graph::kControlSlot}, /*modified=*/false,
                  &expected_node);

  // Remove fanin from node where node is missing.
  TestRemoveFanin("foo_missing", {"a", 0}, /*modified=*/false,
                  /*expected_node=*/nullptr);
  // Remove fanin from node where fanin is missing.
  expected_node = NDef("", "", {"a"});
  TestRemoveFanin("foo_1", {"bar_missing", 0}, /*modified=*/false,
                  &expected_node);
  // Remove fanin from node where node and fanin are missing.
  TestRemoveFanin("foo_missing", {"bar_missing", 0}, /*modified=*/false,
                  /*expected_node=*/nullptr);
}

void TestRemoveAllFanins(absl::string_view node_name,
                         bool keep_controlling_nodes, bool modified,
                         const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  absl::flat_hash_set<string> fanin_strings;
  if (expected_node == nullptr) {
    EXPECT_EQ(node, nullptr);
  } else {
    EXPECT_NE(node, nullptr);
    fanin_strings.insert(node->input().begin(), node->input().end());
  }

  EXPECT_EQ(modified, graph.RemoveAllFanins(node_name, keep_controlling_nodes));
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
    if (modified) {
      TensorId tensor_id;
      auto retained_inputs = absl::flat_hash_set<string>(node->input().begin(),
                                                         node->input().end());
      for (const string& fanin : fanin_strings) {
        if (!retained_inputs.contains(fanin)) {
          tensor_id = ParseTensorName(fanin);
          CheckFanout(graph, tensor_id, node_name);
        }
      }
    }
  }
}

TEST(MutableGraphViewTest, RemoveAllFanins) {
  NodeDef expected_node;
  // Remove all fanins from node with no control dependencies.
  expected_node = NDef("", "", {});
  TestRemoveAllFanins("foo_3", /*keep_controlling_nodes=*/false,
                      /*modified=*/true, &expected_node);
  // Remove all fanins from node with control dependencies.
  TestRemoveAllFanins("foo_4", /*keep_controlling_nodes=*/false,
                      /*modified=*/true, &expected_node);

  // Remove all fanins from node with no control dependencies and preserve
  // control dependencies.
  TestRemoveAllFanins("foo_3", /*keep_controlling_nodes=*/true,
                      /*modified=*/true, &expected_node);
  // Remove all fanins from node with control dependencies and preserve control
  // dependencies.
  expected_node = NDef("", "", {"^c", "^d"});
  TestRemoveAllFanins("foo_4", /*keep_controlling_nodes=*/true,
                      /*modified=*/true, &expected_node);

  // Remove all fanins from node with no fanins.
  expected_node = NDef("", "", {});
  TestRemoveAllFanins("foo_5", /*keep_controlling_nodes=*/false,
                      /*modified=*/false, &expected_node);
  TestRemoveAllFanins("foo_5", /*keep_controlling_nodes=*/true,
                      /*modified=*/false, &expected_node);

  // Remove all fanins from node with only control dependencies.
  TestRemoveAllFanins("foo_6", /*keep_controlling_nodes=*/false,
                      /*modified=*/true, &expected_node);
  expected_node = NDef("", "", {"^a", "^b"});
  TestRemoveAllFanins("foo_6", /*keep_controlling_nodes=*/true,
                      /*modified=*/false, &expected_node);

  // Remove all fanins from node where node is missing.
  TestRemoveAllFanins("foo_missing", /*keep_controlling_nodes=*/false,
                      /*modified=*/false, /*expected_node=*/nullptr);
  TestRemoveAllFanins("foo_missing", /*keep_controlling_nodes=*/true,
                      /*modified=*/false, /*expected_node=*/nullptr);
}

void TestUpdateFanin(absl::string_view node_name, const TensorId& from_fanin,
                     const TensorId& to_fanin, bool modified,
                     const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  if (expected_node == nullptr) {
    EXPECT_EQ(node, nullptr);
  } else {
    EXPECT_NE(node, nullptr);
  }

  EXPECT_EQ(modified, graph.UpdateFanin(node_name, from_fanin, to_fanin));
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
    if (modified) {
      CheckFanout(graph, from_fanin, node_name);
    }
  }
}

TEST(MutableGraphViewTest, UpdateFanin) {
  NodeDef expected_node;
  // Update fanin from non control to non control.
  expected_node = NDef("", "", {"a", "b:3", "b:3", "^c", "^d"});
  TestUpdateFanin("foo_4", {"b", 2}, {"b", 3}, /*modified=*/true,
                  &expected_node);
  // Update fanin from non control to control.
  expected_node = NDef("", "", {"a", "^c", "^d", "^b"});
  TestUpdateFanin("foo_4", {"b", 2}, {"b", Graph::kControlSlot},
                  /*modified=*/true, &expected_node);
  // Update fanin from control to non control.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "d:1", "^c"});
  TestUpdateFanin("foo_4", {"d", Graph::kControlSlot}, {"d", 1},
                  /*modified=*/true, &expected_node);
  // Update fanin from control to control.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^d", "^b"});
  TestUpdateFanin("foo_4", {"c", Graph::kControlSlot},
                  {"b", Graph::kControlSlot}, /*modified=*/true,
                  &expected_node);
  // Update fanin from control to existing control.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^d"});
  TestUpdateFanin("foo_4", {"c", Graph::kControlSlot},
                  {"d", Graph::kControlSlot}, /*modified=*/true,
                  &expected_node);

  // Update fanin of node where from and to fanins are the same.
  expected_node = NDef("", "", {"a"});
  TestUpdateFanin("foo_1", {"a", -1}, {"a", -1}, /*modified=*/false,
                  &expected_node);
  TestUpdateFanin("foo_1", {"a", 0}, {"a", 0}, /*modified=*/false,
                  &expected_node);
  TestUpdateFanin("foo_1", {"a", 1}, {"a", 1}, /*modified=*/false,
                  &expected_node);
  // Update fanin of node where node is missing.
  TestUpdateFanin("foo_missing", {"a", 0}, {"a", 1}, /*modified=*/false,
                  /*expected_node=*/nullptr);
  // Update fanin of node where from fanin is missing.
  TestUpdateFanin("foo_1", {"from_bar_missing", 0}, {"a", 1},
                  /*modified=*/false, &expected_node);
  // Update fanin of node where to fanin is missing.
  TestUpdateFanin("foo_1", {"a", 0}, {"to_bar_missing", 1}, /*modified=*/false,
                  &expected_node);
  // Update fanin of node where from/to fanins and node are missing.
  TestUpdateFanin("foo_missing", {"from_bar_missing", 0}, {"to_bar_missing", 1},
                  /*modified=*/false, /*expected_node=*/nullptr);
}

TEST(MutableGraphViewTest, DeleteNodes) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("bar", "NotImportant", {}, {}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar", "other", "bar:1", "^bar"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar:2", "^bar"})},
      /* empty function library */ {});

  MutableGraphView graph(&graph_def);

  EXPECT_NE(graph.GetNode("foo_1"), nullptr);
  graph.DeleteNodes({"foo_1"});

  EXPECT_EQ(graph.GetNode("foo_1"), nullptr);

  NodeDef* bar = graph.GetNode("bar");
  NodeDef* other = graph.GetNode("other");
  NodeDef* foo_2 = graph.GetNode("foo_2");

  bool include_control_fanouts = true;
  auto bar_fanouts = graph.GetFanouts(*bar, include_control_fanouts);
  auto other_fanouts = graph.GetFanouts(*other, include_control_fanouts);

  EXPECT_EQ(bar_fanouts.size(), 2);
  EXPECT_EQ(bar_fanouts.count(MutableGraphView::InputPort(foo_2, 1)), 1);
  EXPECT_EQ(bar_fanouts.count(MutableGraphView::InputPort(foo_2, -1)), 1);

  EXPECT_EQ(other_fanouts.size(), 1);
  EXPECT_EQ(other_fanouts.count(MutableGraphView::InputPort(foo_2, 0)), 1);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
