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
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/inputs/trivial_test_graph_input_yielder.h"
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
