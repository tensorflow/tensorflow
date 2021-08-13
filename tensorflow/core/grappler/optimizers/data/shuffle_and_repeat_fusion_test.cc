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

constexpr char kOutputShapes[] = "output_shapes";
constexpr char kOutputTypes[] = "output_types";
constexpr char kReshuffleEachIteration[] = "reshuffle_each_iteration";

TEST(ShuffleAndRepeatFusionTest, FuseShuffleV1AndRepeat) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue(kOutputShapes, &shapes_attr);
  common_attrs[0] = std::make_pair(kOutputShapes, shapes_attr);
  AttrValue types_attr;
  SetAttrValue(kOutputTypes, &types_attr);
  common_attrs[1] = std::make_pair(kOutputTypes, types_attr);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             common_attrs, &graph);

  NodeDef *buffer_size_node =
      graph_utils::AddScalarConstNode<int64_t>(128, &graph);
  NodeDef *seed_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  NodeDef *seed2_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  std::vector<string> shuffle_inputs(4);
  shuffle_inputs[0] = range_node->name();
  shuffle_inputs[1] = buffer_size_node->name();
  shuffle_inputs[2] = seed_node->name();
  shuffle_inputs[3] = seed2_node->name();
  NodeDef *shuffle_node = graph_utils::AddNode(
      "", "ShuffleDataset", shuffle_inputs, common_attrs, &graph);
  (*shuffle_node->mutable_attr())[kReshuffleEachIteration].set_b(true);

  NodeDef *count_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  std::vector<string> repeat_inputs(2);
  repeat_inputs[0] = shuffle_node->name();
  repeat_inputs[1] = count_node->name();
  NodeDef *repeat_node = graph_utils::AddNode(
      "", "RepeatDataset", repeat_inputs, common_attrs, &graph);

  ShuffleAndRepeatFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(shuffle_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(repeat_node->name(), output));
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("ShuffleAndRepeatDataset", output));
  NodeDef shuffle_and_repeat_node = output.node(
      graph_utils::FindGraphNodeWithOp("ShuffleAndRepeatDataset", output));
  EXPECT_EQ(shuffle_and_repeat_node.input_size(), 5);
  EXPECT_EQ(shuffle_and_repeat_node.input(0), shuffle_node->input(0));
  EXPECT_EQ(shuffle_and_repeat_node.input(1), shuffle_node->input(1));
  EXPECT_EQ(shuffle_and_repeat_node.input(2), shuffle_node->input(2));
  EXPECT_EQ(shuffle_and_repeat_node.input(3), shuffle_node->input(3));
  EXPECT_EQ(shuffle_and_repeat_node.input(4), repeat_node->input(1));
  for (const auto &attr :
       {kOutputShapes, kOutputTypes, kReshuffleEachIteration}) {
    EXPECT_TRUE(AreAttrValuesEqual(shuffle_and_repeat_node.attr().at(attr),
                                   shuffle_node->attr().at(attr)));
  }
}

TEST(ShuffleAndRepeatFusionTest, FuseShuffleV2AndRepeat) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue(kOutputShapes, &shapes_attr);
  common_attrs[0] = std::make_pair(kOutputShapes, shapes_attr);
  AttrValue types_attr;
  SetAttrValue(kOutputTypes, &types_attr);
  common_attrs[1] = std::make_pair(kOutputTypes, types_attr);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             common_attrs, &graph);

  NodeDef *buffer_size_node =
      graph_utils::AddScalarConstNode<int64_t>(128, &graph);
  NodeDef *seed_generator_node =
      graph_utils::AddScalarConstNode<StringPiece>("dummy_resource", &graph);
  std::vector<string> shuffle_inputs(3);
  shuffle_inputs[0] = range_node->name();
  shuffle_inputs[1] = buffer_size_node->name();
  shuffle_inputs[2] = seed_generator_node->name();
  NodeDef *shuffle_node = graph_utils::AddNode(
      "", "ShuffleDatasetV2", shuffle_inputs, common_attrs, &graph);

  NodeDef *count_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  std::vector<string> repeat_inputs(2);
  repeat_inputs[0] = shuffle_node->name();
  repeat_inputs[1] = count_node->name();
  NodeDef *repeat_node = graph_utils::AddNode(
      "", "RepeatDataset", repeat_inputs, common_attrs, &graph);

  ShuffleAndRepeatFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(shuffle_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(repeat_node->name(), output));
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("ShuffleAndRepeatDatasetV2", output));
  NodeDef shuffle_and_repeat_node = output.node(
      graph_utils::FindGraphNodeWithOp("ShuffleAndRepeatDatasetV2", output));
  EXPECT_EQ(shuffle_and_repeat_node.input_size(), 6);
  EXPECT_EQ(shuffle_and_repeat_node.input(0), shuffle_node->input(0));
  EXPECT_EQ(shuffle_and_repeat_node.input(1), shuffle_node->input(1));
  EXPECT_EQ(shuffle_and_repeat_node.input(4), repeat_node->input(1));
  EXPECT_EQ(shuffle_and_repeat_node.input(5), shuffle_node->input(2));
  for (const auto &attr : {kOutputShapes, kOutputTypes}) {
    EXPECT_TRUE(AreAttrValuesEqual(shuffle_and_repeat_node.attr().at(attr),
                                   shuffle_node->attr().at(attr)));
  }
  EXPECT_TRUE(shuffle_and_repeat_node.attr().at(kReshuffleEachIteration).b());
}

TEST(ShuffleAndRepeatFusionTest, FuseShuffleV3AndRepeat) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue(kOutputShapes, &shapes_attr);
  common_attrs[0] = std::make_pair(kOutputShapes, shapes_attr);
  AttrValue types_attr;
  SetAttrValue(kOutputTypes, &types_attr);
  common_attrs[1] = std::make_pair(kOutputTypes, types_attr);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             common_attrs, &graph);

  NodeDef *buffer_size_node =
      graph_utils::AddScalarConstNode<int64_t>(128, &graph);
  NodeDef *seed_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  NodeDef *seed2_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  NodeDef *seed_generator_node =
      graph_utils::AddScalarConstNode<StringPiece>("dummy_resource", &graph);
  std::vector<string> shuffle_inputs(5);
  shuffle_inputs[0] = range_node->name();
  shuffle_inputs[1] = buffer_size_node->name();
  shuffle_inputs[2] = seed_node->name();
  shuffle_inputs[3] = seed2_node->name();
  shuffle_inputs[4] = seed_generator_node->name();
  NodeDef *shuffle_node = graph_utils::AddNode(
      "", "ShuffleDatasetV3", shuffle_inputs, common_attrs, &graph);
  (*shuffle_node->mutable_attr())[kReshuffleEachIteration].set_b(true);

  NodeDef *count_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  std::vector<string> repeat_inputs(2);
  repeat_inputs[0] = shuffle_node->name();
  repeat_inputs[1] = count_node->name();
  NodeDef *repeat_node = graph_utils::AddNode(
      "", "RepeatDataset", repeat_inputs, common_attrs, &graph);

  ShuffleAndRepeatFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(shuffle_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(repeat_node->name(), output));
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("ShuffleAndRepeatDatasetV2", output));
  NodeDef shuffle_and_repeat_node = output.node(
      graph_utils::FindGraphNodeWithOp("ShuffleAndRepeatDatasetV2", output));
  EXPECT_EQ(shuffle_and_repeat_node.input_size(), 6);
  EXPECT_EQ(shuffle_and_repeat_node.input(0), shuffle_node->input(0));
  EXPECT_EQ(shuffle_and_repeat_node.input(1), shuffle_node->input(1));
  EXPECT_EQ(shuffle_and_repeat_node.input(2), shuffle_node->input(2));
  EXPECT_EQ(shuffle_and_repeat_node.input(3), shuffle_node->input(3));
  EXPECT_EQ(shuffle_and_repeat_node.input(4), repeat_node->input(1));
  EXPECT_EQ(shuffle_and_repeat_node.input(5), shuffle_node->input(4));
  for (const auto &attr :
       {kOutputShapes, kOutputTypes, kReshuffleEachIteration}) {
    EXPECT_TRUE(AreAttrValuesEqual(shuffle_and_repeat_node.attr().at(attr),
                                   shuffle_node->attr().at(attr)));
  }
}

TEST(ShuffleAndRepeatFusionTest, NoChange) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  std::vector<std::pair<string, AttrValue>> common_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue(kOutputShapes, &shapes_attr);
  common_attrs[0] = std::make_pair(kOutputShapes, shapes_attr);
  AttrValue types_attr;
  SetAttrValue(kOutputTypes, &types_attr);
  common_attrs[1] = std::make_pair(kOutputTypes, types_attr);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             common_attrs, &graph);

  NodeDef *count_node = graph_utils::AddScalarConstNode<int64_t>(-1, &graph);
  std::vector<string> repeat_inputs(2);
  repeat_inputs[0] = range_node->name();
  repeat_inputs[1] = count_node->name();
  graph_utils::AddNode("", "RepeatDataset", repeat_inputs, common_attrs,
                       &graph);

  ShuffleAndRepeatFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_TRUE(graph_utils::Compare(*graph.graph(), output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
