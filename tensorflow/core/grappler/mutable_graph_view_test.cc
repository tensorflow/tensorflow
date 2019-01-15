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
#include "absl/strings/substitute.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/types.pb.h"
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
       NDef("foo_2", "NotImportant", {"other:1", "bar:2", "^bar"}),
       NDef("foo_3", "NotImportant", {"other:2", "^bar"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "NotImportant", {}, {}));
  NodeDef* bar = graph.GetNode("bar");

  EXPECT_TRUE(graph.UpdateFanouts(bar->name(), new_bar->name()).ok());

  // Fanout nodes must have their inputs updated.
  NodeDef* foo_1 = graph.GetNode("foo_1");
  ASSERT_NE(foo_1, nullptr);
  ASSERT_EQ(foo_1->input_size(), 3);
  EXPECT_EQ(foo_1->input(0), "new_bar");
  EXPECT_EQ(foo_1->input(1), "other");
  EXPECT_EQ(foo_1->input(2), "new_bar:1");

  NodeDef* foo_2 = graph.GetNode("foo_2");
  ASSERT_NE(foo_2, nullptr);
  ASSERT_EQ(foo_2->input_size(), 2);
  EXPECT_EQ(foo_2->input(0), "other:1");
  EXPECT_EQ(foo_2->input(1), "new_bar:2");

  NodeDef* foo_3 = graph.GetNode("foo_3");
  ASSERT_NE(foo_3, nullptr);
  ASSERT_EQ(foo_3->input_size(), 2);
  EXPECT_EQ(foo_3->input(0), "other:2");
  EXPECT_EQ(foo_3->input(1), "^new_bar");

  // And fanouts mapping must be also updated for both nodes.
  bool include_control_fanouts = true;
  auto old_node_fanouts = graph.GetFanouts(*bar, include_control_fanouts);
  auto new_node_fanouts = graph.GetFanouts(*new_bar, include_control_fanouts);

  EXPECT_TRUE(old_node_fanouts.empty());

  EXPECT_EQ(new_node_fanouts.size(), 4);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_1, 0)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_1, 2)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_2, 1)), 1);
  EXPECT_EQ(new_node_fanouts.count(MutableGraphView::InputPort(foo_3, -1)), 1);
}

TEST(MutableGraphViewTest, AddAndUpdateFanoutsKeepControls) {
  GraphDef graph_def = test::function::GDef(
      {NDef("bar_1", "Switch", {}, {}), NDef("bar_2", "Identity", {"bar_1:1"}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar_2", "other", "bar_2:1", "^bar_2"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar_2:2", "^bar_2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "Identity", {"bar_1:2"}));
  NodeDef* bar_2 = graph.GetNode("bar_2");

  EXPECT_TRUE(graph.UpdateFanouts(bar_2->name(), new_bar->name()).ok());

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
  auto old_node_fanouts = graph.GetFanouts(*bar_2, include_control_fanouts);
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
                            NDef("foo_1", "NotImportant", {"bar", "^bar"}),
                            NDef("foo_2", "NotImportant", {"^bar"})},
                           /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  // `new_bar` reads the output of an original `bar` node.
  NodeDef* new_bar = graph.AddNode(NDef("new_bar", "NewBar", {"bar"}, {}));
  NodeDef* bar = graph.GetNode("bar");

  EXPECT_TRUE(graph.UpdateFanouts("bar", new_bar->name()).ok());

  // Foo node must read from `new_bar`.
  NodeDef* foo_1 = graph.GetNode("foo_1");
  ASSERT_NE(foo_1, nullptr);
  ASSERT_EQ(foo_1->input_size(), 1);
  EXPECT_EQ(foo_1->input(0), "new_bar");

  NodeDef* foo_2 = graph.GetNode("foo_2");
  ASSERT_NE(foo_2, nullptr);
  ASSERT_EQ(foo_2->input_size(), 1);
  EXPECT_EQ(foo_2->input(0), "^new_bar");

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
  EXPECT_EQ(new_bar_fanouts.count(MutableGraphView::InputPort(foo_1, 0)), 1);
  EXPECT_EQ(new_bar_fanouts.count(MutableGraphView::InputPort(foo_2, -1)), 1);
}

void CompareNodeInputs(const MutableGraphView& graph, const NodeDef* expected,
                       NodeDef* actual) {
  ASSERT_EQ(actual->input_size(), expected->input_size());
  for (int i = 0; i < actual->input_size(); ++i) {
    EXPECT_EQ(actual->input(i), expected->input(i));
    TensorId tensor_id = ParseTensorName(expected->input(i));
    int port;
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

TEST(MutableGraphViewTest, UpdateFanoutsToSwitchWithControlFromSwitch) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {}),
       NDef("c", "NotImportant", {}, {}), NDef("d", "NotImportant", {}, {}),
       NDef("e", "NotImportant", {"c", "b", "^a", "^d"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.UpdateFanouts("a", "b");
  EXPECT_FALSE(s.ok());
  string expected_msg =
      "Can't update fanouts from 'a' to 'b', to node is being added as a "
      "Switch control dependency.";
  EXPECT_EQ(s.error_message(), expected_msg);
  s = graph.UpdateFanouts("d", "b");
  EXPECT_FALSE(s.ok());
  expected_msg =
      "Can't update fanouts from 'd' to 'b', to node is being added as a "
      "Switch control dependency.";
  EXPECT_EQ(s.error_message(), expected_msg);

  EXPECT_EQ(graph.graph()->node_size(), 5);

  NodeDef expected = NDef("", "", {}, {});
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  CompareNodeInputs(graph, &expected, b);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  CompareNodeInputs(graph, &expected, c);

  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  CompareNodeInputs(graph, &expected, d);

  NodeDef* e = graph.GetNode("e");
  ASSERT_NE(e, nullptr);
  expected = NDef("", "", {"c", "b", "^a", "^d"});
  CompareNodeInputs(graph, &expected, e);
}

TEST(MutableGraphViewTest, UpdateFanoutsToSwitchWithNoControlFromSwitch) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {}),
       NDef("c", "NotImportant", {}, {}), NDef("d", "NotImportant", {}, {}),
       NDef("e", "NotImportant", {"c", "b", "^a", "^d"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.UpdateFanouts("c", "b").ok());

  EXPECT_EQ(graph.graph()->node_size(), 5);

  NodeDef expected = NDef("", "", {}, {});
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  CompareNodeInputs(graph, &expected, b);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  CompareNodeInputs(graph, &expected, c);

  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  CompareNodeInputs(graph, &expected, d);

  NodeDef* e = graph.GetNode("e");
  ASSERT_NE(e, nullptr);
  expected = NDef("", "", {"b", "b", "^a", "^d"});
  CompareNodeInputs(graph, &expected, e);
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

void TestAddRegularFanin(absl::string_view node_name,
                         const TensorId& fanin_to_add, bool success,
                         const string& error_msg,
                         const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  if (expected_node == nullptr) {
    EXPECT_EQ(node, nullptr);
  } else {
    EXPECT_NE(node, nullptr);
  }

  Status s = graph.AddRegularFanin(node_name, fanin_to_add);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
  }
}

TEST(MutableGraphViewTest, AddRegularFanin) {
  NodeDef expected_node;
  string error_msg;
  // Add input to node with 1 input 0 controls.
  expected_node = NDef("", "", {"a", "b:1"});
  TestAddRegularFanin("foo_1", {"b", 1}, /*success=*/true, error_msg,
                      &expected_node);
  // Add input to node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b", "a:1", "a:1", "b:2"});
  TestAddRegularFanin("foo_3", {"b", 2}, /*success=*/true, error_msg,
                      &expected_node);
  // Add input to node with 1 input multiple controls.
  expected_node = NDef("", "", {"b", "a", "^c"});
  TestAddRegularFanin("foo_2", {"a", 0}, /*success=*/true, error_msg,
                      &expected_node);
  // Add input to node with multiple inputs and controls.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "a:1", "^d", "^c"});
  TestAddRegularFanin("foo_4", {"a", 1}, /*success=*/true, error_msg,
                      &expected_node);
  // Add input to node with 0 inputs 0 controls.
  expected_node = NDef("", "", {"a:1"});
  TestAddRegularFanin("foo_5", {"a", 1}, /*success=*/true, error_msg,
                      &expected_node);
  // Add input to node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"c:1", "^b", "^a"});
  TestAddRegularFanin("foo_6", {"c", 1}, /*success=*/true, error_msg,
                      &expected_node);

  // Add control to node with 1 input 0 controls.
  expected_node = NDef("", "", {"a"});
  error_msg = "Can't add invalid fanin '^b' as regular fanin to node 'foo_1'.";
  TestAddRegularFanin("foo_1", {"b", Graph::kControlSlot}, /*success=*/false,
                      error_msg, &expected_node);
  // Add control to node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b", "a:1", "a:1"});
  error_msg = "Can't add invalid fanin '^c' as regular fanin to node 'foo_3'.";
  TestAddRegularFanin("foo_3", {"c", Graph::kControlSlot}, /*success=*/false,
                      error_msg, &expected_node);
  // Add control to node with 1 input multiple controls.
  expected_node = NDef("", "", {"b", "^a", "^c"});
  error_msg = "Can't add invalid fanin '^d' as regular fanin to node 'foo_2'.";
  TestAddRegularFanin("foo_2", {"d", Graph::kControlSlot}, /*success=*/false,
                      error_msg, &expected_node);
  // Add control to node with multiple input multiple controls.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^c", "^d"});
  error_msg = "Can't add invalid fanin '^a' as regular fanin to node 'foo_4'.";
  TestAddRegularFanin("foo_4", {"a", Graph::kControlSlot},
                      /*success=*/false, error_msg, &expected_node);
  // Add control to node with 0 inputs 0 controls.
  expected_node = NDef("", "", {});
  error_msg = "Can't add invalid fanin '^a' as regular fanin to node 'foo_5'.";
  TestAddRegularFanin("foo_5", {"a", Graph::kControlSlot}, /*success=*/false,
                      error_msg, &expected_node);
  // Add control to node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"^a", "^b"});
  error_msg = "Can't add invalid fanin '^c' as regular fanin to node 'foo_6'.";
  TestAddRegularFanin("foo_6", {"c", Graph::kControlSlot}, /*success=*/false,
                      error_msg, &expected_node);
  // Add control to node with control that already exists.
  expected_node = NDef("", "", {"b", "^a", "^c"});
  error_msg = "Can't add invalid fanin '^a' as regular fanin to node 'foo_2'.";
  TestAddRegularFanin("foo_2", {"a", Graph::kControlSlot},
                      /*success=*/false, error_msg, &expected_node);

  // Add fanin to node where node is missing.
  error_msg =
      "Can't add fanin 'a:0' as regular fanin to missing node 'foo_missing'.";
  TestAddRegularFanin("foo_missing", {"a", 0}, /*success=*/false, error_msg,
                      nullptr);
  // Add fanin to node where fanin is missing.
  expected_node = NDef("", "", {"a"});
  error_msg =
      "Can't add missing fanin 'bar_missing:0' as regular fanin to node "
      "'foo_1'.";
  TestAddRegularFanin("foo_1", {"bar_missing", 0}, /*success=*/false, error_msg,
                      &expected_node);
  // Add fanin to node where node and fanin are missing.
  error_msg =
      "Can't add missing fanin 'bar_missing:0' as regular fanin to missing "
      "node 'foo_missing'.";
  TestAddRegularFanin("foo_missing", {"bar_missing", 0}, /*success=*/false,
                      error_msg, /*expected_node=*/nullptr);
  // Add control fanin to node where node and fanin are missing.
  error_msg =
      "Can't add invalid/missing fanin '^bar_missing' as regular fanin to "
      "missing node 'foo_missing'.";
  TestAddRegularFanin("foo_missing", {"bar_missing", Graph::kControlSlot},
                      /*success=*/false, error_msg, /*expected_node=*/nullptr);
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

void TestRemoveRegularFanin(absl::string_view node_name,
                            const TensorId& fanin_to_remove, bool success,
                            const string& error_msg,
                            const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  if (expected_node == nullptr) {
    EXPECT_EQ(nullptr, node);
  } else {
    EXPECT_NE(nullptr, node);
  }

  Status s = graph.RemoveRegularFanin(node_name, fanin_to_remove);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
    if (success) {
      CheckFanout(graph, fanin_to_remove, node_name);
    }
  }
}

TEST(MutableGraphViewTest, RemoveRegularFanin) {
  NodeDef expected_node;
  string error_msg;
  // Remove input from node with 1 input 0 controls.
  expected_node = NDef("", "", {});
  TestRemoveRegularFanin("foo_1", {"a", 0}, /*success=*/true, error_msg,
                         &expected_node);
  // Remove input from node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b"});
  TestRemoveRegularFanin("foo_3", {"a", 1}, /*success=*/true, error_msg,
                         &expected_node);
  // Remove input from node with 1 input multiple controls.
  expected_node = NDef("", "", {"^a", "^c"});
  TestRemoveRegularFanin("foo_2", {"b", 0}, /*success=*/true, error_msg,
                         &expected_node);
  // Remove input from node with multiple inputs and controls.
  expected_node = NDef("", "", {"a", "^c", "^d"});
  TestRemoveRegularFanin("foo_4", {"b", 2}, /*success=*/true, error_msg,
                         &expected_node);
  // Remove input from node with multiple inputs and controls, and results in
  // shifting of ports.
  expected_node = NDef("", "", {"b:2", "b:2", "^c", "^d"});
  TestRemoveRegularFanin("foo_4", {"a", 0}, /*success=*/true, error_msg,
                         &expected_node);

  // Remove control from node with 1 input multiple controls.
  expected_node = NDef("", "", {"b", "^a", "^c"});
  error_msg =
      "Can't remove invalid fanin '^a' as regular fanin from node 'foo_2'.";
  TestRemoveRegularFanin("foo_2", {"a", Graph::kControlSlot},
                         /*success=*/false, error_msg, &expected_node);
  // Remove control from node with multiple input multiple controls.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^c", "^d"});
  error_msg =
      "Can't remove invalid fanin '^d' as regular fanin from node 'foo_4'.";
  TestRemoveRegularFanin("foo_4", {"d", Graph::kControlSlot},
                         /*success=*/false, error_msg, &expected_node);
  // Remove control from node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"^a", "^b"});
  error_msg =
      "Can't remove invalid fanin '^a' as regular fanin from node 'foo_6'.";
  TestRemoveRegularFanin("foo_6", {"a", Graph::kControlSlot},
                         /*success=*/false, error_msg, &expected_node);

  // Remove input from node with 0 inputs 0 controls.
  expected_node = NDef("", "", {});
  error_msg = "";
  TestRemoveRegularFanin("foo_5", {"a", 1}, /*success=*/true, error_msg,
                         &expected_node);
  // Remove input from node with 0 inputs multiple controls.
  expected_node = NDef("", "", {"^a", "^b"});
  TestRemoveRegularFanin("foo_6", {"a", 1}, /*success=*/true, error_msg,
                         &expected_node);

  // Remove control from node with 1 input 0 controls.
  expected_node = NDef("", "", {"a"});
  error_msg =
      "Can't remove invalid fanin '^b' as regular fanin from node 'foo_1'.";
  TestRemoveRegularFanin("foo_1", {"b", Graph::kControlSlot},
                         /*success=*/false, error_msg, &expected_node);
  // Remove control from node with multiple inputs and 0 controls.
  expected_node = NDef("", "", {"b", "a:1", "a:1"});
  error_msg =
      "Can't remove invalid fanin '^c' as regular fanin from node 'foo_3'.";
  TestRemoveRegularFanin("foo_3", {"c", Graph::kControlSlot},
                         /*success=*/false, error_msg, &expected_node);
  // Remove control from node with 0 inputs 0 controls.
  expected_node = NDef("", "", {});
  error_msg =
      "Can't remove invalid fanin '^a' as regular fanin from node 'foo_5'.";
  TestRemoveRegularFanin("foo_5", {"a", Graph::kControlSlot},
                         /*success=*/false, error_msg, &expected_node);

  // Remove fanin from node where node is missing.
  error_msg =
      "Can't remove fanin 'a:0' as regular fanin from missing node "
      "'foo_missing'.";
  TestRemoveRegularFanin("foo_missing", {"a", 0}, /*success=*/false, error_msg,
                         /*expected_node=*/nullptr);
  // Remove fanin from node where fanin is missing.
  expected_node = NDef("", "", {"a"});
  error_msg =
      "Can't remove missing fanin 'bar_missing:0' as regular fanin from node "
      "'foo_1'.";
  TestRemoveRegularFanin("foo_1", {"bar_missing", 0}, /*success=*/false,
                         error_msg, &expected_node);
  // Remove fanin from node where node and fanin are missing.
  error_msg =
      "Can't remove missing fanin 'bar_missing:0' as regular fanin from "
      "missing node 'foo_missing'.";
  TestRemoveRegularFanin("foo_missing", {"bar_missing", 0}, /*success=*/false,
                         error_msg,
                         /*expected_node=*/nullptr);
  // Remove control from node where node and fanin are missing.
  error_msg =
      "Can't remove invalid/missing fanin '^bar_missing' as regular fanin from "
      "missing node 'foo_missing'.";
  TestRemoveRegularFanin("foo_missing", {"bar_missing", Graph::kControlSlot},
                         /*success=*/false, error_msg,
                         /*expected_node=*/nullptr);
}

void TestRemoveAllFanins(absl::string_view node_name,
                         bool keep_controlling_nodes, bool success,
                         const string& error_msg,
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

  Status s = graph.RemoveAllFanins(node_name, keep_controlling_nodes);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
    if (success) {
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
  string error_msg;
  // Remove all fanins from node with no control dependencies.
  expected_node = NDef("", "", {});
  TestRemoveAllFanins("foo_3", /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, &expected_node);
  // Remove all fanins from node with control dependencies.
  TestRemoveAllFanins("foo_4", /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, &expected_node);

  // Remove all fanins from node with no control dependencies and preserve
  // control dependencies.
  TestRemoveAllFanins("foo_3", /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, &expected_node);
  // Remove all fanins from node with control dependencies and preserve control
  // dependencies.
  expected_node = NDef("", "", {"^c", "^d"});
  TestRemoveAllFanins("foo_4", /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, &expected_node);

  // Remove all fanins from node with no fanins.
  expected_node = NDef("", "", {});
  TestRemoveAllFanins("foo_5", /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, &expected_node);
  TestRemoveAllFanins("foo_5", /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, &expected_node);

  // Remove all fanins from node with only control dependencies.
  TestRemoveAllFanins("foo_6", /*keep_controlling_nodes=*/false,
                      /*success=*/true, error_msg, &expected_node);
  expected_node = NDef("", "", {"^a", "^b"});
  TestRemoveAllFanins("foo_6", /*keep_controlling_nodes=*/true,
                      /*success=*/true, error_msg, &expected_node);

  // Remove all fanins from node where node is missing.
  error_msg = "Can't remove all fanins from missing node 'foo_missing'.";
  TestRemoveAllFanins("foo_missing", /*keep_controlling_nodes=*/false,
                      /*success=*/false, error_msg, /*expected_node=*/nullptr);
  TestRemoveAllFanins("foo_missing", /*keep_controlling_nodes=*/true,
                      /*success=*/false, error_msg, /*expected_node=*/nullptr);
}

void TestUpdateFanin(absl::string_view node_name, const TensorId& from_fanin,
                     const TensorId& to_fanin, bool success,
                     const string& error_msg, const NodeDef* expected_node) {
  GraphDef graph_def = SimpleMutateFaninGraph();

  MutableGraphView graph(&graph_def);

  auto node = graph.GetNode(node_name);
  if (expected_node == nullptr) {
    EXPECT_EQ(node, nullptr);
  } else {
    EXPECT_NE(node, nullptr);
  }

  Status s = graph.UpdateFanin(node_name, from_fanin, to_fanin);
  EXPECT_EQ(s.ok(), success);
  if (!success) {
    EXPECT_EQ(s.error_message(), error_msg);
  }
  if (expected_node != nullptr) {
    CompareNodeInputs(graph, expected_node, node);
    if (success) {
      CheckFanout(graph, from_fanin, node_name);
    }
  }
}

TEST(MutableGraphViewTest, UpdateFanin) {
  NodeDef expected_node;
  string error_msg;
  // Update fanin from non control to non control.
  expected_node = NDef("", "", {"a", "b:3", "b:3", "^c", "^d"});
  TestUpdateFanin("foo_4", {"b", 2}, {"b", 3}, /*success=*/true, error_msg,
                  &expected_node);
  // Update fanin from non control to control.
  expected_node = NDef("", "", {"a", "^c", "^d", "^b"});
  TestUpdateFanin("foo_4", {"b", 2}, {"b", Graph::kControlSlot},
                  /*success=*/true, error_msg, &expected_node);
  // Update fanin from control to non control.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "d:1", "^c"});
  TestUpdateFanin("foo_4", {"d", Graph::kControlSlot}, {"d", 1},
                  /*success=*/true, error_msg, &expected_node);
  // Update fanin from control to control.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^d"});
  TestUpdateFanin("foo_4", {"c", Graph::kControlSlot},
                  {"b", Graph::kControlSlot}, /*success=*/true, error_msg,
                  &expected_node);
  // Update fanin from control to existing control.
  expected_node = NDef("", "", {"a", "b:2", "b:2", "^d"});
  TestUpdateFanin("foo_4", {"c", Graph::kControlSlot},
                  {"d", Graph::kControlSlot}, /*success=*/true, error_msg,
                  &expected_node);

  // Update fanin of node where from and to fanins are the same.
  expected_node = NDef("", "", {"a"});
  TestUpdateFanin("foo_1", {"a", -1}, {"a", -1}, /*success=*/true, error_msg,
                  &expected_node);
  TestUpdateFanin("foo_1", {"a", 0}, {"a", 0}, /*success=*/true, error_msg,
                  &expected_node);
  TestUpdateFanin("foo_1", {"a", 1}, {"a", 1}, /*success=*/true, error_msg,
                  &expected_node);

  // Update fanin of node where node is missing.
  error_msg =
      "Can't update fanin 'a:0' to fanin 'a:1' in missing node 'foo_missing'.";
  TestUpdateFanin("foo_missing", {"a", 0}, {"a", 1}, /*success=*/false,
                  error_msg,
                  /*expected_node=*/nullptr);
  // Update fanin of node where from fanin is missing.
  error_msg =
      "Can't update missing fanin 'from_bar_missing:0' to fanin 'a:1' in node "
      "'foo_1'.";
  TestUpdateFanin("foo_1", {"from_bar_missing", 0}, {"a", 1},
                  /*success=*/false, error_msg, &expected_node);
  // Update fanin of node where to fanin is missing.
  error_msg =
      "Can't update fanin 'a:0' to missing fanin 'to_bar_missing:1' in node "
      "'foo_1'.";
  TestUpdateFanin("foo_1", {"a", 0}, {"to_bar_missing", 1}, /*success=*/false,
                  error_msg, &expected_node);
  // Update fanin of node where from/to fanins and node are missing.
  error_msg =
      "Can't update missing fanin 'from_bar_missing:0' to missing fanin "
      "'to_bar_missing:1' in missing node 'foo_missing'.";
  TestUpdateFanin("foo_missing", {"from_bar_missing", 0}, {"to_bar_missing", 1},
                  /*success=*/false, error_msg, /*expected_node=*/nullptr);
  // Update fanin of node where from fanin is invalid.
  error_msg =
      "Can't update invalid fanin 'a:-2' to fanin 'a:0' in node 'foo_1'.";
  TestUpdateFanin("foo_1", {"a", -2}, {"a", 0},
                  /*success=*/false, error_msg, &expected_node);
  // Update fanin of node where to fanin is invalid.
  error_msg =
      "Can't update fanin 'a:0' to invalid fanin 'a:-2' in node 'foo_1'.";
  TestUpdateFanin("foo_1", {"a", 0}, {"a", -2},
                  /*success=*/false, error_msg, &expected_node);
  // Update fanin of node where from/to fanins are invalid and missing and node
  // is missing.
  error_msg =
      "Can't update invalid/missing fanin 'from_bar_missing:-2' to "
      "invalid/missing fanin 'to_bar_missing:-3' in missing node "
      "'foo_missing'.";
  TestUpdateFanin("foo_missing", {"from_bar_missing", -2},
                  {"to_bar_missing", -3},
                  /*success=*/false, error_msg, /*expected_node=*/nullptr);
}

void TestUpdateFaninFromFaninToNodeAsSwitchControl(const TensorId& fanin) {
  string tensor_id_str = TensorIdToString(fanin);
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "Switch", {}, {}),
       NDef("c", "NotImportant", {tensor_id_str})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  Status s = graph.UpdateFanin("c", fanin, {"b", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  string expected_msg = absl::Substitute(
      "Can't update fanin '$0' to fanin '^b' in node 'c', to fanin is a Switch "
      "control dependency.",
      fanin.ToString());
  EXPECT_EQ(s.error_message(), expected_msg);

  EXPECT_EQ(graph.graph()->node_size(), 3);

  NodeDef expected;
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  expected = NDef("", "", {});
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  CompareNodeInputs(graph, &expected, b);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  expected = NDef("", "", {tensor_id_str});
  CompareNodeInputs(graph, &expected, c);
}

TEST(MutableGraphViewTest, UpdateFaninToNodeAsSwitchControl) {
  TestUpdateFaninFromFaninToNodeAsSwitchControl({"a", 0});
  TestUpdateFaninFromFaninToNodeAsSwitchControl({"a", 1});
  TestUpdateFaninFromFaninToNodeAsSwitchControl({"a", Graph::kControlSlot});
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnGraphInit) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "Switch", {}, {}), NDef("d", "Identity", {"c:1"}),
       NDef("foo_1", "IdentityN", {"a", "b:1", "^b"}),
       NDef("foo_2", "IdentityN", {"a", "^b", "^b"}),
       NDef("foo_3", "IdentityN", {"a", "b:1", "^b", "^b"}),
       NDef("foo_4", "IdentityN", {"a:2", "b:1", "^b", "^b", "^a", "^a"}),
       NDef("foo_5", "NotImportant", {"a:2", "b:1", "^b", "^b", "^a", "^a"}),
       NDef("foo_6", "Identity", {"d", "^d"}),
       NDef("foo_7", "NotImportant",
            {"a:3", "b:2", "d", "^d", "^d", "^a", "^b", "^a", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_EQ(graph.graph()->node_size(), 11);
  NodeDef expected;
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  expected = NDef("", "", {});
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  CompareNodeInputs(graph, &expected, b);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  CompareNodeInputs(graph, &expected, c);

  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  expected = NDef("", "", {"c:1"});
  CompareNodeInputs(graph, &expected, d);

  NodeDef* foo_1 = graph.GetNode("foo_1");
  ASSERT_NE(foo_1, nullptr);
  expected = NDef("", "", {"a", "b:1"});
  CompareNodeInputs(graph, &expected, foo_1);

  NodeDef* foo_2 = graph.GetNode("foo_2");
  ASSERT_NE(foo_2, nullptr);
  expected = NDef("", "", {"a", "^b"});
  CompareNodeInputs(graph, &expected, foo_2);

  NodeDef* foo_3 = graph.GetNode("foo_3");
  ASSERT_NE(foo_3, nullptr);
  expected = NDef("", "", {"a", "b:1"});
  CompareNodeInputs(graph, &expected, foo_3);

  NodeDef* foo_4 = graph.GetNode("foo_4");
  ASSERT_NE(foo_4, nullptr);
  expected = NDef("", "", {"a:2", "b:1"});
  CompareNodeInputs(graph, &expected, foo_4);

  NodeDef* foo_5 = graph.GetNode("foo_5");
  ASSERT_NE(foo_5, nullptr);
  expected = NDef("", "", {"a:2", "b:1"});
  CompareNodeInputs(graph, &expected, foo_5);

  NodeDef* foo_6 = graph.GetNode("foo_6");
  ASSERT_NE(foo_6, nullptr);
  expected = NDef("", "", {"d", "^d"});
  CompareNodeInputs(graph, &expected, foo_6);

  NodeDef* foo_7 = graph.GetNode("foo_7");
  ASSERT_NE(foo_7, nullptr);
  expected = NDef("", "", {"a:3", "b:2", "d", "^d"});
  CompareNodeInputs(graph, &expected, foo_7);
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnAddFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"^a"}),
       NDef("c", "NotImportant", {"a:1"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.AddRegularFanin("b", {"a", 2}).ok());
  NodeDef expected;
  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  expected = NDef("", "", {"a:2"});
  CompareNodeInputs(graph, &expected, b);

  EXPECT_TRUE(graph.AddControllingFanin("c", {"a", Graph::kControlSlot}).ok());
  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  expected = NDef("", "", {"a:1"});
  CompareNodeInputs(graph, &expected, c);
}

TEST(MutableGraphViewTest, NoDedupControlFlowControllingFaninsOnAddFanin) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "Switch", {}, {}), NDef("b", "Identity", {"a:1"}),
       NDef("c", "", {}, {}), NDef("d", "", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.AddRegularFanin("c", {"b", 2}).ok());
  NodeDef expected;
  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  expected = NDef("", "", {"b:2"});
  CompareNodeInputs(graph, &expected, c);
  EXPECT_TRUE(graph.AddControllingFanin("c", {"b", Graph::kControlSlot}).ok());
  expected = NDef("", "", {"b:2", "^b"});
  CompareNodeInputs(graph, &expected, c);
  EXPECT_TRUE(graph.AddControllingFanin("c", {"b", Graph::kControlSlot}).ok());
  expected = NDef("", "", {"b:2", "^b"});
  CompareNodeInputs(graph, &expected, c);

  EXPECT_TRUE(graph.AddControllingFanin("d", {"b", Graph::kControlSlot}).ok());
  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  expected = NDef("", "", {"^b"});
  CompareNodeInputs(graph, &expected, d);
  EXPECT_TRUE(graph.AddControllingFanin("d", {"b", Graph::kControlSlot}).ok());
  expected = NDef("", "", {"^b"});
  CompareNodeInputs(graph, &expected, d);
  EXPECT_TRUE(graph.AddRegularFanin("d", {"b", 3}).ok());
  expected = NDef("", "", {"b:3", "^b"});
  CompareNodeInputs(graph, &expected, d);
}

TEST(MutableGraphViewTest, DedupControllingFaninsOnUpdateFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {"a:1", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.UpdateFanin("c", {"a", 1}, {"b", 2}).ok());
  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  NodeDef expected = NDef("", "", {"b:2"});
  CompareNodeInputs(graph, &expected, c);
}

TEST(MutableGraphViewTest, NoDedupControlFlowControllingFaninsOnUpdateFanin) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "Switch", {}, {}), NDef("b", "Identity", {"a:1"}),
       NDef("c", "Identity", {"a:2"}), NDef("d", "NotImportant", {"c", "^b"}),
       NDef("e", "NotImportant", {"b", "^c"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph
                  .UpdateFanin("d", {"b", Graph::kControlSlot},
                               {"c", Graph::kControlSlot})
                  .ok());
  NodeDef expected;
  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  expected = NDef("", "", {"c", "^c"});
  CompareNodeInputs(graph, &expected, d);

  EXPECT_TRUE(graph.UpdateFanin("e", {"b", 0}, {"c", 3}).ok());
  NodeDef* e = graph.GetNode("e");
  ASSERT_NE(e, nullptr);
  expected = NDef("", "", {"c:3", "^c"});
  CompareNodeInputs(graph, &expected, e);

  EXPECT_TRUE(
      graph.UpdateFanin("e", {"c", 3}, {"c", Graph::kControlSlot}).ok());
  ASSERT_NE(e, nullptr);
  expected = NDef("", "", {"^c"});
  CompareNodeInputs(graph, &expected, e);
}

TEST(MutableGraphViewTest, UpdateMaxRegularOutputPortOnAddFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.AddRegularFanin("c", {"a", 3}).ok());
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);

  auto fanouts = graph.GetFanouts(*a, /*include_controlled_nodes=*/true);
  EXPECT_EQ(fanouts.size(), 2);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(fanouts.count(MutableGraphView::InputPort(b, 0)), 1);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(fanouts.count(MutableGraphView::InputPort(c, 0)), 1);
}

TEST(MutableGraphViewTest, UpdateMaxRegularOutputPortOnRemoveFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"a:2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.RemoveRegularFanin("c", {"a", 2}).ok());
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);

  auto fanouts = graph.GetFanouts(*a, /*include_controlled_nodes=*/true);
  EXPECT_EQ(fanouts.size(), 1);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(fanouts.count(MutableGraphView::InputPort(b, 0)), 1);
}

TEST(MutableGraphViewTest, KeepMaxRegularOutputPortOnRemoveFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"a:2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.RemoveRegularFanin("b", {"a", 1}).ok());
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);

  auto fanouts = graph.GetFanouts(*a, /*include_controlled_nodes=*/true);
  EXPECT_EQ(fanouts.size(), 1);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(fanouts.count(MutableGraphView::InputPort(c, 0)), 1);
}

TEST(MutableGraphViewTest, UpdateMaxRegularOutputPortOnUpdateFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a:1"}),
       NDef("c", "NotImportant", {"a:2"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.UpdateFanin("c", {"a", 2}, {"b", 3}).ok());
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);

  auto a_fanouts = graph.GetFanouts(*a, /*include_controlled_nodes=*/true);
  EXPECT_EQ(a_fanouts.size(), 1);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  EXPECT_EQ(a_fanouts.count(MutableGraphView::InputPort(b, 0)), 1);

  auto b_fanouts = graph.GetFanouts(*b, /*include_controlled_nodes=*/true);
  EXPECT_EQ(b_fanouts.size(), 1);

  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(b_fanouts.count(MutableGraphView::InputPort(c, 0)), 1);
}

TEST(MutableGraphViewTest, AddControllingFaninMissing) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);
  // Missing fanin.
  Status s = graph.AddControllingFanin("a", {"c", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  string expected_msg = "Can't add missing controlling fanin '^c' to node 'a'.";
  EXPECT_EQ(s.error_message(), expected_msg);
  // Missing node.
  s = graph.AddControllingFanin("d", {"a", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  expected_msg = "Can't add controlling fanin '^a' to missing node 'd'.";
  EXPECT_EQ(s.error_message(), expected_msg);
  // Missing node and fanin.
  s = graph.AddControllingFanin("c", {"d", Graph::kControlSlot});
  EXPECT_FALSE(s.ok());
  expected_msg =
      "Can't add missing controlling fanin '^d' to missing node 'c'.";
  EXPECT_EQ(s.error_message(), expected_msg);

  ASSERT_EQ(graph.graph()->node_size(), 2);
  NodeDef expected;
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  expected = NDef("", "", {});
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  CompareNodeInputs(graph, &expected, b);
}

TEST(MutableGraphViewTest, AddControllingFaninExistingControl) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);
  EXPECT_TRUE(graph.AddControllingFanin("a", {"b", Graph::kControlSlot}).ok());
  EXPECT_TRUE(graph.AddControllingFanin("a", {"b", Graph::kControlSlot}).ok());

  ASSERT_EQ(graph.graph()->node_size(), 2);
  NodeDef expected;
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  expected = NDef("", "", {"^b"});
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  expected = NDef("", "", {});
  CompareNodeInputs(graph, &expected, b);
}

TEST(MutableGraphViewTest, AddControllingFaninNotSwitch) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);
  EXPECT_TRUE(graph.AddControllingFanin("a", {"b", 2}).ok());
  EXPECT_TRUE(graph.AddControllingFanin("a", {"b", 2}).ok());

  ASSERT_EQ(graph.graph()->node_size(), 2);
  NodeDef expected;
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  expected = NDef("", "", {"^b"});
  CompareNodeInputs(graph, &expected, a);

  NodeDef* b = graph.GetNode("b");
  ASSERT_NE(b, nullptr);
  expected = NDef("", "", {});
  CompareNodeInputs(graph, &expected, b);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitchWithIdentity) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("switch", "Switch", {}, {}),
       NDef("identity", "Identity", {"switch"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.AddControllingFanin("a", {"switch", 0}).ok());
  EXPECT_TRUE(graph.AddControllingFanin("a", {"switch", 0}).ok());

  ASSERT_EQ(graph.graph()->node_size(), 3);
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  NodeDef expected = NDef("", "", {"^identity"});
  CompareNodeInputs(graph, &expected, a);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitchWithNoExistingIdentity) {
  constexpr char kDevice[] = "/device:foo:0";
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}),
       NDef("switch", "Switch", {}, {{"T", DT_FLOAT}}, kDevice)},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.AddControllingFanin("a", {"switch", 0}).ok());
  EXPECT_TRUE(graph.AddControllingFanin("a", {"switch", 0}).ok());

  ASSERT_EQ(graph.graph()->node_size(), 3);
  NodeDef expected;
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  expected = NDef("", "", {"^ConstantFoldingCtrl/switch_0"});
  CompareNodeInputs(graph, &expected, a);

  NodeDef* identity = graph.GetNode("ConstantFoldingCtrl/switch_0");
  ASSERT_NE(identity, nullptr);
  expected = NDef("", "", {"switch"});
  CompareNodeInputs(graph, &expected, identity);
  EXPECT_EQ(identity->op(), "Identity");
  EXPECT_EQ(identity->device(), kDevice);
  ASSERT_TRUE(identity->attr().count("T"));
  EXPECT_EQ(identity->attr().at("T").type(), DT_FLOAT);
}

TEST(MutableGraphViewTest, AddControllingFaninSwitchWithExistingAddedIdentity) {
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("switch", "Switch", {}, {}),
       NDef("ConstantFoldingCtrl/switch_0", "Identity", {"switch"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.AddControllingFanin("a", {"switch", 0}).ok());
  EXPECT_TRUE(graph.AddControllingFanin("a", {"switch", 0}).ok());

  ASSERT_EQ(graph.graph()->node_size(), 3);
  NodeDef* a = graph.GetNode("a");
  ASSERT_NE(a, nullptr);
  NodeDef expected = NDef("", "", {"^ConstantFoldingCtrl/switch_0"});
  CompareNodeInputs(graph, &expected, a);
}

TEST(MutableGraphViewTest, RemoveControllingFaninMissing) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {}, {}),
       NDef("d", "NotImportant", {"^a", "^b"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.RemoveControllingFanin("d", "c").ok());

  ASSERT_EQ(graph.graph()->node_size(), 4);
  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  NodeDef expected = NDef("", "", {"^a", "^b"});
  CompareNodeInputs(graph, &expected, d);
}

TEST(MutableGraphViewTest, RemoveControllingFaninExisting) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {}, {}),
       NDef("c", "NotImportant", {}, {}),
       NDef("d", "NotImportant", {"^a", "^b", "^c"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.RemoveControllingFanin("d", "a").ok());
  EXPECT_TRUE(graph.RemoveControllingFanin("d", "a").ok());

  ASSERT_EQ(graph.graph()->node_size(), 4);
  NodeDef* d = graph.GetNode("d");
  ASSERT_NE(d, nullptr);
  NodeDef expected = NDef("", "", {"^c", "^b"});
  CompareNodeInputs(graph, &expected, d);
}

TEST(MutableGraphViewTest, RemoveControllingFaninOnRegularFanin) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("a", "NotImportant", {}, {}), NDef("b", "NotImportant", {"a"}),
       NDef("c", "NotImportant", {"a", "b", "^c"})},
      /*funcs=*/{});

  MutableGraphView graph(&graph_def);

  EXPECT_TRUE(graph.RemoveControllingFanin("c", "a").ok());
  EXPECT_TRUE(graph.RemoveControllingFanin("c", "b").ok());

  ASSERT_EQ(graph.graph()->node_size(), 3);
  NodeDef* c = graph.GetNode("c");
  ASSERT_NE(c, nullptr);
  NodeDef expected = NDef("", "", {"a", "b", "^c"});
  CompareNodeInputs(graph, &expected, c);
}

TEST(MutableGraphViewTest, DeleteNodes) {
  // Actual node.op() is not important in this test.
  GraphDef graph_def = test::function::GDef(
      {NDef("bar", "NotImportant", {}, {}),
       NDef("other", "NotImportant", {}, {}),
       NDef("foo_1", "NotImportant", {"bar", "other", "bar:1", "^bar"}),
       NDef("foo_2", "NotImportant", {"other:1", "bar:2", "^bar"})},
      /*funcs=*/{});

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

  EXPECT_EQ(bar_fanouts.size(), 1);
  EXPECT_EQ(bar_fanouts.count(MutableGraphView::InputPort(foo_2, 1)), 1);

  EXPECT_EQ(other_fanouts.size(), 1);
  EXPECT_EQ(other_fanouts.count(MutableGraphView::InputPort(foo_2, 0)), 1);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
