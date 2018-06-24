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

#include "tensorflow/core/grappler/optimizers/data/shuffle_and_repeat_fusion.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(ShuffleAndRepeatFusionTest, FuseShuffleAndRepeatNodesIntoOne) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue("output_shapes", &shapes_attr);
  common_attrs[0] = std::make_pair("output_shapes", shapes_attr);
  AttrValue types_attr;
  SetAttrValue("output_types", &types_attr);
  common_attrs[1] = std::make_pair("output_types", types_attr);

  NodeDef *start_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(0, graph, &start_node));
  NodeDef *stop_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(10, graph, &stop_node));
  NodeDef *step_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(1, graph, &step_node));

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RangeDataset", range_inputs,
                                    common_attrs, graph, &range_node));

  NodeDef *buffer_size_node;
  TF_ASSERT_OK(
      graph_utils::AddScalarConstNode<int64>(128, graph, &buffer_size_node));
  NodeDef *seed_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(-1, graph, &seed_node));
  NodeDef *seed2_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(-1, graph, &seed2_node));
  std::vector<string> shuffle_inputs(4);
  shuffle_inputs[0] = range_node->name();
  shuffle_inputs[1] = buffer_size_node->name();
  shuffle_inputs[2] = seed_node->name();
  shuffle_inputs[3] = seed2_node->name();
  NodeDef *shuffle_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "ShuffleDataset", shuffle_inputs,
                                    common_attrs, graph, &shuffle_node));

  NodeDef *count_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(-1, graph, &count_node));
  std::vector<string> repeat_inputs(2);
  repeat_inputs[0] = shuffle_node->name();
  repeat_inputs[1] = count_node->name();
  NodeDef *repeat_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RepeatDataset", repeat_inputs,
                                    common_attrs, graph, &repeat_node));

  ShuffleAndRepeatFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsNodeWithName(shuffle_node->name(), output));
  EXPECT_FALSE(graph_utils::ContainsNodeWithName(repeat_node->name(), output));
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("ShuffleAndRepeatDataset", output));
  NodeDef shuffle_and_repeat_node = output.node(
      graph_utils::FindNodeWithOp("ShuffleAndRepeatDataset", output));
  EXPECT_EQ(shuffle_and_repeat_node.input_size(), 5);
  EXPECT_EQ(shuffle_and_repeat_node.input(0), shuffle_node->input(0));
  EXPECT_EQ(shuffle_and_repeat_node.input(1), shuffle_node->input(1));
  EXPECT_EQ(shuffle_and_repeat_node.input(2), shuffle_node->input(2));
  EXPECT_EQ(shuffle_and_repeat_node.input(3), shuffle_node->input(3));
  EXPECT_EQ(shuffle_and_repeat_node.input(4), repeat_node->input(1));
  EXPECT_TRUE(
      AreAttrValuesEqual(shuffle_and_repeat_node.attr().at("output_shapes"),
                         repeat_node->attr().at("output_shapes")));
  EXPECT_TRUE(
      AreAttrValuesEqual(shuffle_and_repeat_node.attr().at("output_types"),
                         repeat_node->attr().at("output_types")));
}

TEST(ShuffleAndRepeatFusionTest, NoChange) {
  GrapplerItem item;
  GraphDef *graph = &item.graph;

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue("output_shapes", &shapes_attr);
  common_attrs[0] = std::make_pair("output_shapes", shapes_attr);
  AttrValue types_attr;
  SetAttrValue("output_types", &types_attr);
  common_attrs[1] = std::make_pair("output_types", types_attr);

  NodeDef *start_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(0, graph, &start_node));
  NodeDef *stop_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(10, graph, &stop_node));
  NodeDef *step_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(1, graph, &step_node));

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RangeDataset", range_inputs,
                                    common_attrs, graph, &range_node));

  NodeDef *count_node;
  TF_ASSERT_OK(graph_utils::AddScalarConstNode<int64>(-1, graph, &count_node));
  std::vector<string> repeat_inputs(2);
  repeat_inputs[0] = range_node->name();
  repeat_inputs[1] = count_node->name();
  NodeDef *repeat_node;
  TF_ASSERT_OK(graph_utils::AddNode("", "RepeatDataset", repeat_inputs,
                                    common_attrs, graph, &repeat_node));

  ShuffleAndRepeatFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_TRUE(graph_utils::Compare(*graph, output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
