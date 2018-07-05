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

void MakeUnaryNode(GraphDef *graph, const std::string &node_type, int count,
                   string input_node, NodeDef **return_node) {
  NodeDef *node_count;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<int64>(count, graph, &node_count));
  TF_ASSERT_OK(graph_utils::AddNode("", node_type,
                                    {std::move(input_node), node_count->name()},
                                    GetCommonAttributes(), graph, return_node));
}

void MakeCacheNode(GraphDef *graph, string input_node, NodeDef **return_node) {
  NodeDef *node_filename;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<StringPiece>("", graph, &node_filename));
  TF_ASSERT_OK(graph_utils::AddNode(
      "", "CacheDataset", {std::move(input_node), node_filename->name()},
      GetCommonAttributes(), graph, return_node));
}

void MakeRangeNode(GraphDef *graph, NodeDef **range_node) {
  NodeDef *start_node, *stop_node, *step_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(0, graph, &start_node));
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(10, graph, &stop_node));
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(1, graph, &step_node));

  std::vector<string> range_inputs = {start_node->name(), stop_node->name(),
                                      step_node->name()};

  TF_ASSERT_OK(graph_utils::AddNode("", "RangeDataset", range_inputs,
                                    GetCommonAttributes(), graph, range_node));
}

struct NoOpLastEliminationTest
    : ::testing::TestWithParam<std::tuple<std::string, int, bool>> {};

// This test checks whether the no-op elimination correctly handles
// transformations at the end of the pipeline.
TEST_P(NoOpLastEliminationTest, EliminateLastNoOpNode) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;

  const std::string &node_type = std::get<0>(GetParam());
  const int node_count = std::get<1>(GetParam());
  const bool should_keep_node = std::get<2>(GetParam());

  NodeDef *range_node;
  MakeRangeNode(graph, &range_node);

  NodeDef *node;
  MakeUnaryNode(graph, node_type, node_count, range_node->name(), &node);

  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsNodeWithName(node->name(), output),
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
    : ::testing::TestWithParam<std::tuple<std::string, int, bool>> {};

// This test checks whether the no-op elimination correctly handles
// transformations int the middle of the pipeline.
TEST_P(NoOpMiddleEliminationTest, EliminateMiddleNoOpNode) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;

  const std::string &node_type = std::get<0>(GetParam());
  const int node_count = std::get<1>(GetParam());
  const bool should_keep_node = std::get<2>(GetParam());

  NodeDef *range_node;
  MakeRangeNode(graph, &range_node);

  NodeDef *node;
  MakeUnaryNode(graph, node_type, node_count, range_node->name(), &node);

  NodeDef *cache_node;
  MakeCacheNode(graph, node->name(), &cache_node);
  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsNodeWithName(node->name(), output),
            should_keep_node);
  EXPECT_TRUE(graph_utils::ContainsNodeWithName(cache_node->name(), output));

  NodeDef cache_node_out =
      output.node(graph_utils::FindNodeWithName(cache_node->name(), output));

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
  GraphDef *graph = &item.graph;

  static_assert(std::tuple_size<NodesTypes>::value == 2,
                "Make sure to include everything in the test");
  const std::vector<std::pair<string, int>> noop_nodes = {
      std::get<0>(GetParam()), std::get<1>(GetParam())};

  NodeDef *range_node;
  MakeRangeNode(graph, &range_node);

  NodeDef *previous = range_node;
  std::vector<string> nodes_to_remove;
  nodes_to_remove.reserve(noop_nodes.size());

  for (const auto &noop_node : noop_nodes) {
    NodeDef *node;
    MakeUnaryNode(graph, noop_node.first, noop_node.second, previous->name(),
                  &node);
    nodes_to_remove.push_back(node->name());
    previous = node;
  }

  NodeDef *cache_node;
  MakeCacheNode(graph, previous->name(), &cache_node);
  NoOpElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  for (const auto &noop_node_name : nodes_to_remove)
    EXPECT_FALSE(graph_utils::ContainsNodeWithName(noop_node_name, output));

  EXPECT_TRUE(graph_utils::ContainsNodeWithName(cache_node->name(), output));

  NodeDef cache_node_out =
      output.node(graph_utils::FindNodeWithName(cache_node->name(), output));

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

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
