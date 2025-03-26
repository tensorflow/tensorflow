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

#include "tensorflow/core/framework/dataset_metadata.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace graph_utils {
namespace {

using test::function::NDef;

constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";
constexpr char kToutputTypes[] = "Toutput_types";

TEST(GraphUtilsTest, GetFirstElementIndexWithPredicate) {
  std::vector<int> vec({1, 2, 3, 4, 5, 6});
  auto result = GetFirstElementIndexWithPredicate(
      [](int elem) { return elem % 3 == 0; }, vec);

  EXPECT_EQ(result, 2);

  result = GetFirstElementIndexWithPredicate(
      [](int elem) { return elem % 7 == 0; }, vec);
  EXPECT_EQ(result, -1);
}

TEST(GraphUtilsTest, AddScalarConstNodeBool) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* bool_node = AddScalarConstNode<bool>(true, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(bool_node->name(), *graph.graph()));
  EXPECT_EQ(bool_node->attr().at("value").tensor().bool_val(0), true);
}

TEST(GraphUtilsTest, AddScalarConstNodeDouble) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* double_node = AddScalarConstNode<double>(3.14, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(double_node->name(), *graph.graph()));
  EXPECT_FLOAT_EQ(double_node->attr().at("value").tensor().double_val(0), 3.14);
}

TEST(GraphUtilsTest, AddScalarConstNodeFloat) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* float_node = AddScalarConstNode<float>(3.14, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(float_node->name(), *graph.graph()));
  EXPECT_FLOAT_EQ(float_node->attr().at("value").tensor().float_val(0), 3.14);
}

TEST(GraphUtilsTest, AddScalarConstNodeInt) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* int_node = AddScalarConstNode<int>(42, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(int_node->name(), *graph.graph()));
  EXPECT_EQ(int_node->attr().at("value").tensor().int_val(0), 42);
}

TEST(GraphUtilsTest, AddScalarConstNodeInt64) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* int64_node = AddScalarConstNode<int64_t>(42, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(int64_node->name(), *graph.graph()));
  EXPECT_EQ(int64_node->attr().at("value").tensor().int64_val(0), 42);
}

TEST(GraphUtilsTest, AddScalarConstNodeString) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* string_node = AddScalarConstNode<absl::string_view>("hello", &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName(string_node->name(), *graph.graph()));
  EXPECT_EQ(string_node->attr().at("value").tensor().string_val(0), "hello");
}

TEST(GraphUtilsTest, GetScalarConstNodeInt64) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* int64_node = AddScalarConstNode<int64_t>(128, &graph);
  int64_t result;
  EXPECT_TRUE(GetScalarConstNodeValue<int64_t>(*int64_node, &result).ok());
  EXPECT_EQ(result, 128);
}

TEST(GraphUtilsTest, GetScalarConstNodeBool) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* bool_node = AddScalarConstNode<bool>(true, &graph);
  bool result;
  EXPECT_TRUE(GetScalarConstNodeValue<bool>(*bool_node, &result).ok());
  EXPECT_EQ(result, true);
}

TEST(GraphUtilsTest, GetScalarConstNodeErrorWithNonConst) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* non_const = AddScalarPlaceholder(DT_INT64, &graph);
  int64_t result;
  absl::Status s = GetScalarConstNodeValue<int64_t>(*non_const, &result);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.message(),
            "Node Placeholder is not a Const node. Op: Placeholder");
}

TEST(GraphUtilsTest, GetScalarConstNodeErrorWithType) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  NodeDef* int64_node = AddScalarConstNode<int64_t>(128, &graph);
  bool result;
  absl::Status s = GetScalarConstNodeValue<bool>(*int64_node, &result);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.message(),
            "Node Const should have type bool but has type: int64");
}

TEST(GraphUtilsTest, GetScalarConstNodeErrorWithVector) {
  NodeDef node;
  node.set_name("Const");
  node.set_op("Const");

  (*node.mutable_attr())["dtype"].set_type(DT_INT64);
  auto tensor = (*node.mutable_attr())["value"].mutable_tensor();
  tensor->set_dtype(DT_INT64);
  tensor->mutable_tensor_shape()->mutable_dim()->Add()->set_size(1);
  tensor->add_int64_val(128);

  int64_t result;
  absl::Status s = GetScalarConstNodeValue<int64_t>(node, &result);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(s.message(), "Node Const should be a scalar but has shape: [1]");
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
  EXPECT_TRUE(!ContainsGraphNodeWithName("A", *graph.graph()));

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_TRUE(ContainsGraphNodeWithName("A", *graph.graph()));

  EXPECT_TRUE(graph.DeleteNodes({"A"}).ok());
  EXPECT_TRUE(!ContainsGraphNodeWithName("A", *graph.graph()));
}

TEST(GraphUtilsTest, ContainsGraphFunctionWithName) {
  FunctionDefLibrary library;
  EXPECT_FALSE(ContainsGraphFunctionWithName("new_function", library));
  FunctionDef* new_function = library.add_function();
  SetUniqueGraphFunctionName("new_function", &library, new_function);

  EXPECT_TRUE(
      ContainsGraphFunctionWithName(new_function->signature().name(), library));
}

TEST(GraphUtilsTest, ContainsNodeWithOp) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_TRUE(!ContainsNodeWithOp("OpA", *graph.graph()));

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_TRUE(ContainsNodeWithOp("OpA", *graph.graph()));

  EXPECT_TRUE(graph.DeleteNodes({"A"}).ok());
  EXPECT_TRUE(!ContainsNodeWithOp("OpA", *graph.graph()));
}

TEST(GraphUtilsTest, FindGraphNodeWithName) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_EQ(FindGraphNodeWithName("A", *graph.graph()), -1);

  AddNode("A", "OpA", {}, {}, &graph);
  EXPECT_NE(FindGraphNodeWithName("A", *graph.graph()), -1);

  EXPECT_TRUE(graph.DeleteNodes({"A"}).ok());
  EXPECT_EQ(FindGraphNodeWithName("A", *graph.graph()), -1);
}

TEST(GraphUtilsTest, FindGraphFunctionWithName) {
  FunctionDefLibrary library;
  EXPECT_EQ(FindGraphFunctionWithName("new_function", library), -1);
  FunctionDef* new_function = library.add_function();
  SetUniqueGraphFunctionName("new_function", &library, new_function);

  EXPECT_NE(
      FindGraphFunctionWithName(new_function->signature().name(), library), -1);
}

TEST(GraphUtilsTest, FindGraphNodeWithOp) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_EQ(FindGraphNodeWithOp("OpA", *graph.graph()), -1);

  AddNode("A", "OpA", {}, {}, &graph);
  AddNode("B", "OpB", {"A"}, {}, &graph);
  AddNode("A2", "OpA", {"A"}, {}, &graph);
  EXPECT_EQ(FindGraphNodeWithOp("OpA", *graph.graph()), 0);

  EXPECT_TRUE(graph.DeleteNodes({"B"}).ok());
  EXPECT_EQ(FindGraphNodeWithOp("OpB", *graph.graph()), -1);
  EXPECT_EQ(FindGraphNodeWithName("A2", *graph.graph()), 1);
}

TEST(GraphUtilsTest, FindAllGraphNodesWithOp) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);
  EXPECT_EQ(FindGraphNodeWithOp("OpA", *graph.graph()), -1);

  AddNode("A", "OpA", {}, {}, &graph);
  AddNode("B", "OpB", {"A"}, {}, &graph);
  AddNode("A2", "OpA", {"B"}, {}, &graph);
  std::vector<int> result_indices =
      FindAllGraphNodesWithOp("OpA", *graph.graph());
  EXPECT_EQ(result_indices.size(), 2);
  EXPECT_EQ(result_indices.at(0), 0);
  EXPECT_EQ(result_indices.at(1), 2);

  EXPECT_TRUE(graph.DeleteNodes({"A2"}).ok());
  std::vector<int> result_indices_new =
      FindAllGraphNodesWithOp("OpA", *graph.graph());
  EXPECT_EQ(result_indices_new.size(), 1);
  EXPECT_EQ(result_indices_new.at(0), 0);
}

TEST(GraphUtilsTest, SetUniqueGraphNodeName) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);

  NodeDef* node1 = AddNode("", "A", {}, {}, &graph);
  NodeDef* node2 = AddNode("", "A", {}, {}, &graph);
  EXPECT_NE(node1->name(), node2->name());

  EXPECT_TRUE(graph.DeleteNodes({node1->name()}).ok());
  NodeDef* node3 = AddNode("", "A", {}, {}, &graph);
  EXPECT_NE(node2->name(), node3->name());
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

TEST(GraphUtilsTest, GetInputNode) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);

  NodeDef* node1 = AddNode("", "A", {}, {}, &graph);
  NodeDef* node2 = AddNode("", "A", {node1->name()}, {}, &graph);

  EXPECT_EQ(GetInputNode(*node2, graph), node1);
  EXPECT_EQ(GetInputNode(*node1, graph), nullptr);
}

TEST(GraphUtilsTest, GetIthInputNode) {
  GraphDef graph_def;
  MutableGraphView graph(&graph_def);

  NodeDef* node1 = AddNode("", "A", {}, {}, &graph);
  NodeDef* node2 = AddNode("", "A", {}, {}, &graph);
  NodeDef* node3 = AddNode("", "A", {node1->name(), node2->name()}, {}, &graph);

  EXPECT_EQ(GetInputNode(*node3, graph), node1);
  EXPECT_EQ(GetInputNode(*node3, graph, 1), node2);
  EXPECT_EQ(GetInputNode(*node3, graph, 0), node1);
  EXPECT_EQ(GetInputNode(*node3, graph, 2), nullptr);
  EXPECT_EQ(GetInputNode(*node1, graph), nullptr);
}

TEST(GraphUtilsTest, EnsureNodeNamesUnique) {
  Graph g(OpRegistry::Global());

  Node *const_0, *const_1, *const_2;

  // Arbitrary const
  Tensor tensor(DT_INT32, {});
  tensor.scalar<int32>()() = 5;

  for (auto node : {&const_0, &const_1}) {
    TF_EXPECT_OK(NodeBuilder("Const", "Const")
                     .Attr("value", tensor)
                     .Attr("dtype", DT_INT32)
                     .Finalize(&g, node));
  }
  // Make sure generated name doesn't clash with existing name either
  TF_EXPECT_OK(NodeBuilder("Const_1", "Const")
                   .Attr("value", tensor)
                   .Attr("dtype", DT_INT32)
                   .Finalize(&g, &const_2));

  TF_EXPECT_OK(EnsureNodeNamesUnique(&g));
  EXPECT_NE(const_0->name(), const_1->name());
  EXPECT_NE(const_1->name(), const_2->name());
  EXPECT_NE(const_0->name(), const_2->name());
}

TEST(GraphUtilsTest, TestGetFetchNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef* node1 = AddNode("node1", "Identity", {}, {}, &graph);
  NodeDef* node2 = AddNode("node2", "Identity", {node1->name()}, {}, &graph);
  NodeDef* node3 = AddNode("node3", "Identity", {node2->name()}, {}, &graph);
  item.fetch.push_back(node3->name());

  NodeDef* sink_node;
  TF_EXPECT_OK(GetFetchNode(graph, item, &sink_node));
  EXPECT_EQ(sink_node->name(), node3->name());
}

TEST(GraphUtilsTest, TestFindSinkNodeMultipleFetches) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef* node1 = AddNode("node1", "Identity", {}, {}, &graph);
  NodeDef* node2 = AddNode("node2", "Identity", {node1->name()}, {}, &graph);
  NodeDef* node3 = AddNode("node3", "Identity", {node2->name()}, {}, &graph);
  item.fetch.push_back(node2->name());
  item.fetch.push_back(node3->name());

  NodeDef* sink_node;
  absl::Status s = GetFetchNode(graph, item, &sink_node);
  EXPECT_FALSE(s.ok());
}

TEST(GraphUtilsTest, TestFindSinkNodeNoFetches) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef* node1 = AddNode("node1", "Identity", {}, {}, &graph);
  NodeDef* node2 = AddNode("node2", "Identity", {node1->name()}, {}, &graph);
  AddNode("node3", "Identity", {node2->name()}, {}, &graph);

  NodeDef* sink_node;
  absl::Status s = GetFetchNode(graph, item, &sink_node);
  EXPECT_FALSE(s.ok());
}

TEST(GraphUtilsTest, TestCopyShapesAndTypesAttrsNoShapes) {
  NodeDef from = NDef("range", "RangeDataset", {},
                      {{kOutputTypes, absl::Span<const DataType>{}}});
  NodeDef to_node;
  EXPECT_FALSE(CopyShapesAndTypesAttrs(from, &to_node));
}

TEST(GraphUtilsTest, TestCopyShapesAndTypesAttrsNoTypes) {
  NodeDef from = NDef("range", "RangeDataset", {},
                      {{kOutputShapes, absl::Span<const TensorShape>{}}});
  NodeDef to_node;
  EXPECT_FALSE(CopyShapesAndTypesAttrs(from, &to_node));
}

TEST(GraphUtilsTest, TestCopyShapesAndTypesAttrsOutputTypes) {
  NodeDef from = NDef("range", "RangeDataset", {},
                      {{kOutputShapes, 666}, {kOutputTypes, 888}});
  NodeDef to_node;
  EXPECT_TRUE(CopyShapesAndTypesAttrs(from, &to_node));
  EXPECT_EQ(to_node.attr().at(kOutputShapes).i(), 666);
  EXPECT_EQ(to_node.attr().at(kOutputTypes).i(), 888);
}

TEST(GraphUtilsTest, TestCopyShapesAndTypesAttrsToutputTypes) {
  NodeDef from = NDef("tensor", "TensorDataset", {},
                      {{kOutputShapes, 666}, {kToutputTypes, 888}});
  NodeDef to_node;
  EXPECT_TRUE(CopyShapesAndTypesAttrs(from, &to_node));
  EXPECT_EQ(to_node.attr().at(kOutputShapes).i(), 666);
  EXPECT_EQ(to_node.attr().at(kOutputTypes).i(), 888);
}

TEST(GraphUtilsTest, TestSetMetadataName) {
  NodeDef node = NDef("range", "RangeDataset", {},
                      {{kOutputShapes, 666}, {kOutputTypes, 888}});
  EXPECT_TRUE(SetMetadataName("metadata_name", &node).ok());
  EXPECT_TRUE(node.attr().contains("metadata"));
  data::Metadata metadata;
  metadata.ParseFromString(node.attr().at("metadata").s());
  EXPECT_EQ("metadata_name", metadata.name());
  EXPECT_FALSE(SetMetadataName("new_metadata_name", &node).ok());
}

}  // namespace
}  // namespace graph_utils
}  // namespace grappler
}  // namespace tensorflow
