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

#include "tensorflow/core/grappler/optimizers/data/take_all_elimination.h"

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

void MakeTakeNode(GraphDef *graph, int count, string input_node,
                  NodeDef **return_node) {
  NodeDef *take_count;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<int64>(count, graph, &take_count));
  TF_ASSERT_OK(graph_utils::AddNode("", "TakeDataset",
                                    {std::move(input_node), take_count->name()},
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

struct TakeLastEliminationTest
    : ::testing::TestWithParam<std::pair<int, bool>> {};

// This test checks if given pipeline:
// range(10) -> take(X)
// is transformed into:
// range(10)  if X < 0.
TEST_P(TakeLastEliminationTest, EliminateLastTakeNode) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;

  const int node_count = GetParam().first;
  const bool should_keep_node = GetParam().second;

  NodeDef *range_node;
  MakeRangeNode(graph, &range_node);

  NodeDef *take_all_node;
  MakeTakeNode(graph, node_count, range_node->name(), &take_all_node);

  TakeAllElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsNodeWithName(take_all_node->name(), output),
            should_keep_node);
}

INSTANTIATE_TEST_CASE_P(BasicRemovalTest, TakeLastEliminationTest,
                        ::testing::Values(std::make_pair(-1, false),
                                          std::make_pair(-3, false),
                                          std::make_pair(0, true),
                                          std::make_pair(3, true)));

struct TakeMiddleEliminationTest
    : ::testing::TestWithParam<std::pair<int, bool>> {};

// This test checks if given pipeline:
// range(10) -> take(X) -> take(3)
// is transformed into:
// range(10) -> take(3) if X < 0.
TEST_P(TakeMiddleEliminationTest, EliminateMiddleTakeNode) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;

  const int node_count = GetParam().first;
  const bool should_keep_node = GetParam().second;

  NodeDef *range_node;
  MakeRangeNode(graph, &range_node);

  NodeDef *take_all_node;
  MakeTakeNode(graph, node_count, range_node->name(), &take_all_node);

  NodeDef *take_three_node;
  MakeTakeNode(graph, 3, take_all_node->name(), &take_three_node);

  TakeAllElimination optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(graph_utils::ContainsNodeWithName(take_all_node->name(), output),
            should_keep_node);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithName(take_three_node->name(), output));

  NodeDef take_three_out = output.node(
      graph_utils::FindNodeWithName(take_three_node->name(), output));

  EXPECT_EQ(take_three_out.input_size(), 2);
  auto last_node_input =
      (should_keep_node ? take_all_node : range_node)->name();
  EXPECT_EQ(take_three_out.input(0), last_node_input);
}

INSTANTIATE_TEST_CASE_P(BasicRemovalTest, TakeMiddleEliminationTest,
                        ::testing::Values(std::make_pair(-1, false),
                                          std::make_pair(-3, false),
                                          std::make_pair(0, true),
                                          std::make_pair(3, true)));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
