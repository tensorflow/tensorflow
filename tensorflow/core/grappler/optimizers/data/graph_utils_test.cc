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

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

TEST(GraphUtilsTest, AddScalarConstNodeBool) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* bool_node = AddScalarConstNode<bool>(true, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(bool_node->name(), *graph.GetGraph()));
  EXPECT_EQ(bool_node->attr().at("value").tensor().bool_val(0), true);
}

TEST(GraphUtilsTest, AddScalarConstNodeDouble) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* double_node = AddScalarConstNode<double>(3.14, &graph);
  EXPECT_TRUE(
      ContainsGraphNodeWithName(double_node->name(), *graph.GetGraph()));
  EXPECT_FLOAT_EQ(double_node->attr().at("value").tensor().double_val(0), 3.14);
}

TEST(GraphUtilsTest, AddScalarConstNodeFloat) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* float_node = AddScalarConstNode<float>(3.14, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(float_node->name(), *graph.GetGraph()));
  EXPECT_FLOAT_EQ(float_node->attr().at("value").tensor().float_val(0), 3.14);
}

TEST(GraphUtilsTest, AddScalarConstNodeInt) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* int_node = AddScalarConstNode<int>(42, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(int_node->name(), *graph.GetGraph()));
  EXPECT_EQ(int_node->attr().at("value").tensor().int_val(0), 42);
}

TEST(GraphUtilsTest, AddScalarConstNodeInt64) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* int64_node = AddScalarConstNode<int64>(42, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(int64_node->name(), *graph.GetGraph()));
  EXPECT_EQ(int64_node->attr().at("value").tensor().int64_val(0), 42);
}

TEST(GraphUtilsTest, AddScalarConstNodeString) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* string_node = AddScalarConstNode<StringPiece>("hello", &graph);
  EXPECT_TRUE(
      ContainsGraphNodeWithName(string_node->name(), *graph.GetGraph()));
  EXPECT_EQ(string_node->attr().at("value").tensor().string_val(0), "hello");
}

TEST(GraphUtilsTest, Compare) {
  GraphDef graph_def_a;
  MutableGraphView graph_a(&graph_def_a);
  GraphDef graph_def_b;
  MutableGraphView graph_b(&graph_def_b);

  EXPECT_TRUE(Compare(graph_def_a, graph_def_b));

  AddNode("A", "OpA", {}, {}, &graph_a);
  AddNode("B", "OpB", {"A"}, {}, &graph_a);
  EXPECT_FALSE(Compare(graph_def_a, graph_def_b));

  graph_def_b.mutable_node()->CopyFrom(graph_def_a.node());
  EXPECT_TRUE(Compare(graph_def_a, graph_def_b));
}

TEST(GraphUtilsTest, ContainsGraphNodeWithName) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_TRUE(!ContainsGraphNodeWithName("A", *graph.GetGraph()));

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName("A", *graph.GetGraph()));

  graph.DeleteNodes({"A"});
  EXPECT_TRUE(!ContainsGraphNodeWithName("A", *graph.GetGraph()));
}

TEST(GraphUtilsTest, ContainsGraphFunctionWithName) {
  FunctionDefLibrary library;
  EXPECT_FALSE(ContainsGraphFunctionWithName("new_function", library));
  FunctionDef* new_function = library.add_function();
  SetUniqueGraphFunctionName("new_function", &library, new_function);

  EXPECT_TRUE(
      ContainsGraphFunctionWithName(new_function->signature().name(), library));
}

TEST(GraphUtilsTest, ContainsFunctionNodeWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_FALSE(ContainsFunctionNodeWithName(
      "weird_name_that_should_not_be_there", function));
  EXPECT_TRUE(ContainsFunctionNodeWithName("two", function));
}

TEST(GraphUtilsTest, ContainsNodeWithOp) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_TRUE(!ContainsNodeWithOp("OpA", *graph.GetGraph()));

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_TRUE(ContainsNodeWithOp("OpA", *graph.GetGraph()));

  graph.DeleteNodes({"A"});
  EXPECT_TRUE(!ContainsNodeWithOp("OpA", *graph.GetGraph()));
}

TEST(GraphUtilsTest, FindGraphNodeWithName) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_EQ(FindGraphNodeWithName("A", *graph.GetGraph()), -1);

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_NE(FindGraphNodeWithName("A", *graph.GetGraph()), -1);

  graph.DeleteNodes({"A"});
  EXPECT_EQ(FindGraphNodeWithName("A", *graph.GetGraph()), -1);
}

TEST(GraphUtilsTest, FindFunctionWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(
      FindFunctionNodeWithName("weird_name_that_should_not_be_there", function),
      -1);
  EXPECT_NE(FindFunctionNodeWithName("two", function), -1);
}

TEST(GraphUtilsTest, FindGraphFunctionWithName) {
  FunctionDefLibrary library;
  EXPECT_EQ(FindGraphFunctionWithName("new_function", library), -1);
  FunctionDef* new_function = library.add_function();
  SetUniqueGraphFunctionName("new_function", &library, new_function);

  EXPECT_NE(
      FindGraphFunctionWithName(new_function->signature().name(), library), -1);
}

TEST(GraphUtilsTest, FindNodeWithOp) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_EQ(FindNodeWithOp("OpA", *graph.GetGraph()), -1);

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_NE(FindNodeWithOp("OpA", *graph.GetGraph()), -1);

  graph.DeleteNodes({"A"});
  EXPECT_EQ(FindNodeWithOp("OpA", *graph.GetGraph()), -1);
}

TEST(GraphUtilsTest, SetUniqueGraphNodeName) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);

  NodeDef* node1 = AddNode("", "A", {}, {}, &graph);
  NodeDef* node2 = AddNode("", "A", {}, {}, &graph);
  EXPECT_NE(node1->name(), node2->name());

  graph.DeleteNodes({node1->name()});
  NodeDef* node3 = AddNode("", "A", {}, {}, &graph);
  EXPECT_NE(node2->name(), node3->name());
}

TEST(GraphUtilsTest, SetUniqueFunctionNodeName) {
  FunctionDef function = test::function::XTimesTwo();
  NodeDef node;
  SetUniqueFunctionNodeName("abc", &function, &node);
  for (const NodeDef& function_node : function.node_def()) {
    EXPECT_NE(node.name(), function_node.name());
  }
  auto* new_node = function.add_node_def();
  *new_node = node;

  NodeDef other;
  SetUniqueFunctionNodeName("abc", &function, &other);
  EXPECT_NE(other.name(), new_node->name());
}

TEST(GraphUtilsTest, SetUniqueGraphFunctionName) {
  FunctionDefLibrary library;
  FunctionDef* new_function = library.add_function();
  SetUniqueGraphFunctionName("new_function", &library, new_function);

  FunctionDef* other_function = library.add_function();
  SetUniqueGraphFunctionName("new_function", &library, other_function);
  EXPECT_NE(new_function->signature().name(),
            other_function->signature().name());
}

}  // namespace
}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow
