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

#include "tensorflow/core/grappler/optimizers/data/noop_elimination.h"
#include <tuple>
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

std::vector<std::pair<string, AttrValue>> GetCommonAttributes() {
  AttrValue shapes_attr, types_attr;
  SetAttrValue("output_shapes", &shapes_attr);
  SetAttrValue("output_types", &types_attr);
  std::vector<std::pair<string, AttrValue>> commonAttributes = {
      {"output_shapes", shapes_attr}, {"output_types", types_attr}};

  return commonAttributes;
}

NodeDef *MakeUnaryNode(StringPiece node_type, int count, string input_node,
                       MutableGraphView *graph) {
  NodeDef *node_count = graph_utils::AddScalarConstNode<int64>(count, graph);
  return graph_utils::AddNode("", node_type,
                              {std::move(input_node), node_count->name()},
                              GetCommonAttributes(), graph);
}

NodeDef *MakeUnaryNonConstNode(StringPiece node_type, string input_node,
                               MutableGraphView *graph) {
  NodeDef *node_count = graph_utils::AddScalarPlaceholder(DT_INT32, graph);
  return graph_utils::AddNode("", node_type,
                              {std::move(input_node), node_count->name()},
                              GetCommonAttributes(), graph);
}

NodeDef *MakeCacheNode(string input_node, MutableGraphView *graph) {
  NodeDef *node_filename =
      graph_utils::AddScalarConstNode<StringPiece>("", graph);
  return graph_utils::AddNode("", "CacheDataset",
                              {std::move(input_node), node_filename->name()},
                              GetCommonAttributes(), graph);
}

NodeDef *MakeRangeNode(MutableGraphView *graph) {
  auto *start_node = graph_utils::AddScalarConstNode<int64>(0, graph);
  auto *stop_node = graph_utils::AddScalarConstNode<int64>(10, graph);
  auto *step_node = graph_utils::AddScalarConstNode<int64>(1, graph);

  std::vector<string> range_inputs = {start_node->name(), stop_node->name(),
                                      step_node->name()};

  return graph_utils::AddNode("", "RangeDataset", range_inputs,
                              GetCommonAttributes(), graph);
}

struct NoOpLastEliminationTest
    : ::testing::TestWithParam<std::tuple<string, int, bool>> {};

// This test checks whether the no-op elimination correctly handles
// transformations at the end of the pipeline.
TEST_P(NoOpLastEliminationTest, EliminateLastNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  const string &node_type = std::get<0>(GetParam());
  const int node_count = std::get<1>(GetParam());
  const bool should_keep_node = std::get<2>(GetParam());

  NodeDef *range_node = MakeRangeNode(&graph);

  NodeDef *node =
      MakeUnaryNode(node_type, node_count, range_node->name(), &graph);

  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName(node->name(), output),
            should_keep_node);
}

INSTANTIATE_TEST_CASE_P(
    BasicRemovalTest, NoOpLastEliminationTest,
    ::testing::Values(std::make_tuple("TakeDataset", -3, false),
                      std::make_tuple("TakeDataset", -1, false),
                      std::make_tuple("TakeDataset", 0, true),
                      std::make_tuple("TakeDataset", 3, true),
                      std::make_tuple("SkipDataset", -1, true),
                      std::make_tuple("SkipDataset", 0, false),
                      std::make_tuple("SkipDataset", 3, true),
                      std::make_tuple("RepeatDataset", 1, false),
                      std::make_tuple("RepeatDataset", 2, true)));

struct NoOpMiddleEliminationTest
    : ::testing::TestWithParam<std::tuple<string, int, bool>> {};

// This test checks whether the no-op elimination correctly handles
// transformations int the middle of the pipeline.
TEST_P(NoOpMiddleEliminationTest, EliminateMiddleNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  const string &node_type = std::get<0>(GetParam());
  const int node_count = std::get<1>(GetParam());
  const bool should_keep_node = std::get<2>(GetParam());

  NodeDef *range_node = MakeRangeNode(&graph);

  NodeDef *node =
      MakeUnaryNode(node_type, node_count, range_node->name(), &graph);

  NodeDef *cache_node = MakeCacheNode(node->name(), &graph);
  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName(node->name(), output),
            should_keep_node);
  EXPECT_TRUE(
      graph_utils::ContainsGraphNodeWithName(cache_node->name(), output));

  NodeDef cache_node_out = output.node(
      graph_utils::FindGraphNodeWithName(cache_node->name(), output));

  EXPECT_EQ(cache_node_out.input_size(), 2);
  auto last_node_input = (should_keep_node ? node : range_node)->name();
  EXPECT_EQ(cache_node_out.input(0), last_node_input);
}

INSTANTIATE_TEST_CASE_P(
    BasicRemovalTest, NoOpMiddleEliminationTest,
    ::testing::Values(std::make_tuple("TakeDataset", -1, false),
                      std::make_tuple("TakeDataset", -3, false),
                      std::make_tuple("TakeDataset", 0, true),
                      std::make_tuple("TakeDataset", 3, true),
                      std::make_tuple("SkipDataset", -1, true),
                      std::make_tuple("SkipDataset", 0, false),
                      std::make_tuple("SkipDataset", 3, true),
                      std::make_tuple("RepeatDataset", 1, false),
                      std::make_tuple("RepeatDataset", 2, true)));

using NodesTypes = std::tuple<std::pair<string, int>, std::pair<string, int>>;
struct NoOpMultipleEliminationTest : ::testing::TestWithParam<NodesTypes> {};

// This test checks whether the no-op elimination correctly removes
// multiple noop nodes.
TEST_P(NoOpMultipleEliminationTest, EliminateMultipleNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  static_assert(std::tuple_size<NodesTypes>::value == 2,
                "Make sure to include everything in the test");
  const std::vector<std::pair<string, int>> noop_nodes = {
      std::get<0>(GetParam()), std::get<1>(GetParam())};

  NodeDef *range_node = MakeRangeNode(&graph);

  NodeDef *previous = range_node;
  std::vector<string> nodes_to_remove;
  nodes_to_remove.reserve(noop_nodes.size());

  for (const auto &noop_node : noop_nodes) {
    NodeDef *node = MakeUnaryNode(noop_node.first, noop_node.second,
                                  previous->name(), &graph);
    nodes_to_remove.push_back(node->name());
    previous = node;
  }

  NodeDef *cache_node = MakeCacheNode(previous->name(), &graph);
  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const auto &noop_node_name : nodes_to_remove)
    EXPECT_FALSE(
        graph_utils::ContainsGraphNodeWithName(noop_node_name, output));

  EXPECT_TRUE(
      graph_utils::ContainsGraphNodeWithName(cache_node->name(), output));

  NodeDef cache_node_out = output.node(
      graph_utils::FindGraphNodeWithName(cache_node->name(), output));

  EXPECT_EQ(cache_node_out.input_size(), 2);
  EXPECT_EQ(cache_node_out.input(0), range_node->name());
}

const auto *const kTakeNode = new std::pair<string, int>{"TakeDataset", -1};
const auto *const kSkipNode = new std::pair<string, int>{"SkipDataset", 0};
const auto *const kRepeatNode = new std::pair<string, int>{"RepeatDataset", 1};

INSTANTIATE_TEST_CASE_P(
    BasicRemovalTest, NoOpMultipleEliminationTest,
    ::testing::Combine(::testing::Values(*kTakeNode, *kSkipNode, *kRepeatNode),
                       ::testing::Values(*kTakeNode, *kSkipNode,
                                         *kRepeatNode)));

struct NoOpPlaceholdersTest
    : ::testing::TestWithParam<std::tuple<string, string>> {};

TEST_P(NoOpPlaceholdersTest, NonConstNoOpNode) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  static_assert(std::tuple_size<NodesTypes>::value == 2,
                "Make sure to include everything in the test");
  const std::vector<string> noop_nodes = {std::get<0>(GetParam()),
                                          std::get<1>(GetParam())};
  NodeDef *range_node = MakeRangeNode(&graph);
  std::vector<string> nodes_to_keep;
  nodes_to_keep.reserve(noop_nodes.size());
  NodeDef *previous = range_node;

  for (const auto &noop_node : noop_nodes) {
    NodeDef *node = MakeUnaryNonConstNode(noop_node, previous->name(), &graph);
    nodes_to_keep.push_back(node->name());
    previous = node;
  }

  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  for (const auto &noop_node_name : nodes_to_keep)
    EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName(noop_node_name, output));
}

INSTANTIATE_TEST_CASE_P(
    DoNotRemovePlaceholders, NoOpPlaceholdersTest,
    ::testing::Combine(
        ::testing::Values("TakeDataset", "SkipDataset", "RepeatDataset"),
        ::testing::Values("TakeDataset", "SkipDataset", "RepeatDataset")));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
