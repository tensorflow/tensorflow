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

#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

TEST(GraphUtilsTest, AddScalarConstNodeBool) {
  GraphDef graph;
  NodeDef* bool_node;
  TF_EXPECT_OK(AddScalarConstNode<bool>(true, &graph, &bool_node));
  EXPECT_TRUE(ContainsNodeWithName(bool_node->name(), graph));
  EXPECT_EQ(bool_node->attr().at("value").tensor().bool_val(0), true);
}

TEST(GraphUtilsTest, AddScalarConstNodeDouble) {
  GraphDef graph;
  NodeDef* double_node;
  TF_EXPECT_OK(AddScalarConstNode<double>(3.14, &graph, &double_node));
  EXPECT_TRUE(ContainsNodeWithName(double_node->name(), graph));
  EXPECT_FLOAT_EQ(double_node->attr().at("value").tensor().double_val(0), 3.14);
}

TEST(GraphUtilsTest, AddScalarConstNodeFloat) {
  GraphDef graph;
  NodeDef* float_node;
  TF_EXPECT_OK(AddScalarConstNode<float>(3.14, &graph, &float_node));
  EXPECT_TRUE(ContainsNodeWithName(float_node->name(), graph));
  EXPECT_FLOAT_EQ(float_node->attr().at("value").tensor().float_val(0), 3.14);
}

TEST(GraphUtilsTest, AddScalarConstNodeInt) {
  GraphDef graph;
  NodeDef* int_node;
  TF_EXPECT_OK(AddScalarConstNode<int>(42, &graph, &int_node));
  EXPECT_TRUE(ContainsNodeWithName(int_node->name(), graph));
  EXPECT_EQ(int_node->attr().at("value").tensor().int_val(0), 42);
}

TEST(GraphUtilsTest, AddScalarConstNodeInt64) {
  GraphDef graph;
  NodeDef* int64_node;
  TF_EXPECT_OK(AddScalarConstNode<int64>(42, &graph, &int64_node));
  EXPECT_TRUE(ContainsNodeWithName(int64_node->name(), graph));
  EXPECT_EQ(int64_node->attr().at("value").tensor().int64_val(0), 42);
}

TEST(GraphUtilsTest, AddScalarConstNodeString) {
  GraphDef graph;
  NodeDef* string_node;
  TF_EXPECT_OK(AddScalarConstNode<StringPiece>("hello", &graph, &string_node));
  EXPECT_TRUE(ContainsNodeWithName(string_node->name(), graph));
  EXPECT_EQ(string_node->attr().at("value").tensor().string_val(0), "hello");
}

TEST(GraphUtilsTest, Compare) {
  GraphDef graphA;
  GraphDef graphB;
  EXPECT_TRUE(Compare(graphA, graphB));

  NodeDef* nodeA;
  TF_EXPECT_OK(AddNode("A", "OpA", {}, {}, &graphA, &nodeA));
  NodeDef* nodeB;
  TF_EXPECT_OK(AddNode("B", "OpB", {"A"}, {}, &graphA, &nodeB));
  EXPECT_FALSE(Compare(graphA, graphB));

  graphB.mutable_node()->CopyFrom(graphA.node());
  EXPECT_TRUE(Compare(graphA, graphB));
}

TEST(GraphUtilsTest, ContainsNodeWithName) {
  GraphDef graph;
  EXPECT_TRUE(!ContainsNodeWithName("A", graph));

  NodeDef* node;
  TF_EXPECT_OK(AddNode("A", "OpA", {}, {}, &graph, &node));
  EXPECT_TRUE(ContainsNodeWithName("A", graph));

  TF_EXPECT_OK(DeleteNodes({"A"}, &graph));
  EXPECT_TRUE(!ContainsNodeWithName("A", graph));
}

TEST(GraphUtilsTest, ContainsNodeWithOp) {
  GraphDef graph;
  EXPECT_TRUE(!ContainsNodeWithOp("OpA", graph));

  NodeDef* node;
  TF_EXPECT_OK(AddNode("A", "OpA", {}, {}, &graph, &node));
  EXPECT_TRUE(ContainsNodeWithOp("OpA", graph));

  TF_EXPECT_OK(DeleteNodes({"A"}, &graph));
  EXPECT_TRUE(!ContainsNodeWithOp("OpA", graph));
}

TEST(GraphUtilsTest, FindNodeWithName) {
  GraphDef graph;
  EXPECT_EQ(FindNodeWithName("A", graph), -1);

  NodeDef* node;
  TF_EXPECT_OK(AddNode("A", "OpA", {}, {}, &graph, &node));
  EXPECT_NE(FindNodeWithName("A", graph), -1);

  TF_EXPECT_OK(DeleteNodes({"A"}, &graph));
  EXPECT_EQ(FindNodeWithName("A", graph), -1);
}

TEST(GraphUtilsTest, FindNodeWithOp) {
  GraphDef graph;
  EXPECT_EQ(FindNodeWithOp("OpA", graph), -1);

  NodeDef* node;
  TF_EXPECT_OK(AddNode("A", "OpA", {}, {}, &graph, &node));
  EXPECT_NE(FindNodeWithOp("OpA", graph), -1);

  TF_EXPECT_OK(DeleteNodes({"A"}, &graph));
  EXPECT_EQ(FindNodeWithOp("OpA", graph), -1);
}

TEST(GraphUtilsTest, SetUniqueName) {
  GraphDef graph;

  NodeDef* node1;
  TF_EXPECT_OK(AddNode("", "A", {}, {}, &graph, &node1));
  NodeDef* node2;
  TF_EXPECT_OK(AddNode("", "A", {}, {}, &graph, &node2));
  EXPECT_NE(node1->name(), node2->name());

  TF_EXPECT_OK(DeleteNodes({node1->name()}, &graph));
  NodeDef* node3;
  TF_EXPECT_OK(AddNode("", "A", {}, {}, &graph, &node3));
  EXPECT_NE(node2->name(), node3->name());
}

TEST(GraphUtilsTest, ReplaceInput) {
  GraphDef graph;

  NodeDef* node1;
  TF_EXPECT_OK(AddNode("", "A", {}, {}, &graph, &node1));

  NodeDef* node2;
  TF_EXPECT_OK(AddNode("", "A", {node1->name()}, {}, &graph, &node2));

  NodeDef* node3;
  TF_EXPECT_OK(AddNode("", "A", {node2->name()}, {}, &graph, &node3));

  EXPECT_EQ(node3->input(0), node2->name());

  GraphView view(&graph);
  ReplaceInput(*node2, *node1, &view);

  EXPECT_EQ(node3->input(0), node1->name());
}

}  // namespace
}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow
