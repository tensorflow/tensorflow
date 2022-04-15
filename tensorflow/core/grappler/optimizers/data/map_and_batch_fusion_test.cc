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

#include "tensorflow/core/grappler/optimizers/data/map_and_batch_fusion.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(MapAndBatchFusionTest, FuseMapAndBatchNodesIntoOne) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);
  NodeDef *captured_input_node =
      graph_utils::AddScalarConstNode<StringPiece>("hello", &graph);

  NodeDef *map_node;
  {
    std::vector<string> map_inputs(2);
    map_inputs[0] = range_node->name();
    map_inputs[1] = captured_input_node->name();
    std::vector<std::pair<string, AttrValue>> map_attrs(2);
    AttrValue f_attr;
    SetAttrValue("f", &f_attr);
    map_attrs[0] = std::make_pair("f", f_attr);
    AttrValue args_attr;
    SetAttrValue("Targuments", &args_attr);
    map_attrs[1] = std::make_pair("Targuments", args_attr);
    map_node =
        graph_utils::AddNode("", "MapDataset", map_inputs, map_attrs, &graph);
  }

  NodeDef *batch_size_node =
      graph_utils::AddScalarConstNode<int64_t>(5, &graph);
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(2);
    batch_inputs[0] = map_node->name();
    batch_inputs[1] = batch_size_node->name();
    std::vector<std::pair<string, AttrValue>> batch_attrs(2);
    AttrValue shapes_attr;
    SetAttrValue("output_shapes", &shapes_attr);
    batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
    AttrValue types_attr;
    SetAttrValue("output_types", &types_attr);
    batch_attrs[1] = std::make_pair("output_types", types_attr);
    batch_node = graph_utils::AddNode("", "BatchDataset", batch_inputs,
                                      batch_attrs, &graph);
  }

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(map_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapAndBatchDataset", output));
  NodeDef map_and_batch_node = output.node(
      graph_utils::FindGraphNodeWithOp("MapAndBatchDataset", output));
  EXPECT_EQ(map_and_batch_node.input_size(), 5);
  EXPECT_EQ(map_and_batch_node.input(0), map_node->input(0));
  EXPECT_EQ(map_and_batch_node.input(1), map_node->input(1));
  EXPECT_EQ(map_and_batch_node.input(2), batch_node->input(1));
  NodeDef num_parallel_calls_node = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(3), output));
  EXPECT_EQ(num_parallel_calls_node.attr().at("value").tensor().int64_val(0),
            1);
  NodeDef drop_remainder_node = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(4), output));
  EXPECT_EQ(drop_remainder_node.attr().at("value").tensor().bool_val(0), false);
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("f"),
                                 map_node->attr().at("f")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("Targuments"),
                                 map_node->attr().at("Targuments")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(MapAndBatchFusionTest, FuseMapAndBatchV2NodesIntoOne) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);
  NodeDef *captured_input_node =
      graph_utils::AddScalarConstNode<StringPiece>("hello", &graph);

  NodeDef *map_node;
  {
    std::vector<string> map_inputs(2);
    map_inputs[0] = range_node->name();
    map_inputs[1] = captured_input_node->name();
    std::vector<std::pair<string, AttrValue>> map_attrs(2);
    AttrValue f_attr;
    SetAttrValue("f", &f_attr);
    map_attrs[0] = std::make_pair("f", f_attr);
    AttrValue args_attr;
    SetAttrValue("Targuments", &args_attr);
    map_attrs[1] = std::make_pair("Targuments", args_attr);
    map_node =
        graph_utils::AddNode("", "MapDataset", map_inputs, map_attrs, &graph);
  }

  NodeDef *batch_size_node =
      graph_utils::AddScalarConstNode<int64_t>(5, &graph);
  NodeDef *drop_remainder_node =
      graph_utils::AddScalarConstNode<bool>(true, &graph);
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(3);
    batch_inputs[0] = map_node->name();
    batch_inputs[1] = batch_size_node->name();
    batch_inputs[2] = drop_remainder_node->name();
    std::vector<std::pair<string, AttrValue>> batch_attrs(2);
    AttrValue shapes_attr;
    SetAttrValue("output_shapes", &shapes_attr);
    batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
    AttrValue types_attr;
    SetAttrValue("output_types", &types_attr);
    batch_attrs[1] = std::make_pair("output_types", types_attr);
    batch_node = graph_utils::AddNode("", "BatchDatasetV2", batch_inputs,
                                      batch_attrs, &graph);
  }

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(map_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapAndBatchDataset", output));
  NodeDef map_and_batch_node = output.node(
      graph_utils::FindGraphNodeWithOp("MapAndBatchDataset", output));
  EXPECT_EQ(map_and_batch_node.input_size(), 5);
  EXPECT_EQ(map_and_batch_node.input(0), map_node->input(0));
  EXPECT_EQ(map_and_batch_node.input(1), map_node->input(1));
  EXPECT_EQ(map_and_batch_node.input(2), batch_node->input(1));
  NodeDef num_parallel_calls_node = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(3), output));
  EXPECT_EQ(num_parallel_calls_node.attr().at("value").tensor().int64_val(0),
            1);
  EXPECT_EQ(map_and_batch_node.input(4), batch_node->input(2));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("f"),
                                 map_node->attr().at("f")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("Targuments"),
                                 map_node->attr().at("Targuments")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(MapAndBatchFusionTest, FuseParallelMapAndBatchNodesIntoOne) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);
  NodeDef *captured_input_node =
      graph_utils::AddScalarConstNode<StringPiece>("hello", &graph);
  NodeDef *num_parallel_calls_node =
      graph_utils::AddScalarConstNode<int>(2, &graph);

  NodeDef *map_node;
  {
    std::vector<string> map_inputs(3);
    map_inputs[0] = range_node->name();
    map_inputs[1] = captured_input_node->name();
    map_inputs[2] = num_parallel_calls_node->name();
    std::vector<std::pair<string, AttrValue>> map_attrs(2);
    AttrValue f_attr;
    SetAttrValue("f", &f_attr);
    map_attrs[0] = std::make_pair("f", f_attr);
    AttrValue args_attr;
    SetAttrValue("Targuments", &args_attr);
    map_attrs[1] = std::make_pair("Targuments", args_attr);
    map_node = graph_utils::AddNode("", "ParallelMapDataset", map_inputs,
                                    map_attrs, &graph);
  }

  NodeDef *batch_size_node =
      graph_utils::AddScalarConstNode<int64_t>(5, &graph);
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(2);
    batch_inputs[0] = map_node->name();
    batch_inputs[1] = batch_size_node->name();
    std::vector<std::pair<string, AttrValue>> batch_attrs(2);
    AttrValue shapes_attr;
    SetAttrValue("output_shapes", &shapes_attr);
    batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
    AttrValue types_attr;
    SetAttrValue("output_types", &types_attr);
    batch_attrs[1] = std::make_pair("output_types", types_attr);
    batch_node = graph_utils::AddNode("", "BatchDataset", batch_inputs,
                                      batch_attrs, &graph);
  }

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(map_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapAndBatchDataset", output));
  NodeDef map_and_batch_node = output.node(
      graph_utils::FindGraphNodeWithOp("MapAndBatchDataset", output));
  EXPECT_EQ(map_and_batch_node.input_size(), 5);
  EXPECT_EQ(map_and_batch_node.input(0), map_node->input(0));
  EXPECT_EQ(map_and_batch_node.input(1), map_node->input(1));
  EXPECT_EQ(map_and_batch_node.input(2), batch_node->input(1));
  NodeDef num_parallel_calls_node2 = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(3), output));
  EXPECT_EQ(num_parallel_calls_node2.attr().at("value").tensor().int64_val(0),
            2);
  NodeDef drop_remainder_node = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(4), output));
  EXPECT_EQ(drop_remainder_node.attr().at("value").tensor().bool_val(0), false);
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("f"),
                                 map_node->attr().at("f")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("Targuments"),
                                 map_node->attr().at("Targuments")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(MapAndBatchFusionTest, FuseParallelMapV2AndBatchNodesIntoOne) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);
  NodeDef *captured_input_node =
      graph_utils::AddScalarConstNode<StringPiece>("hello", &graph);
  NodeDef *num_parallel_calls_node =
      graph_utils::AddScalarConstNode<int64_t>(2, &graph);

  NodeDef *map_node;
  {
    std::vector<string> map_inputs(3);
    map_inputs[0] = range_node->name();
    map_inputs[1] = captured_input_node->name();
    map_inputs[2] = num_parallel_calls_node->name();
    std::vector<std::pair<string, AttrValue>> map_attrs(2);
    AttrValue f_attr;
    SetAttrValue("f", &f_attr);
    map_attrs[0] = std::make_pair("f", f_attr);
    AttrValue args_attr;
    SetAttrValue("Targuments", &args_attr);
    map_attrs[1] = std::make_pair("Targuments", args_attr);
    map_node = graph_utils::AddNode("", "ParallelMapDatasetV2", map_inputs,
                                    map_attrs, &graph);
  }

  NodeDef *batch_size_node =
      graph_utils::AddScalarConstNode<int64_t>(5, &graph);
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(2);
    batch_inputs[0] = map_node->name();
    batch_inputs[1] = batch_size_node->name();
    std::vector<std::pair<string, AttrValue>> batch_attrs(2);
    AttrValue shapes_attr;
    SetAttrValue("output_shapes", &shapes_attr);
    batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
    AttrValue types_attr;
    SetAttrValue("output_types", &types_attr);
    batch_attrs[1] = std::make_pair("output_types", types_attr);
    batch_node = graph_utils::AddNode("", "BatchDataset", batch_inputs,
                                      batch_attrs, &graph);
  }

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(map_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapAndBatchDataset", output));
  NodeDef map_and_batch_node = output.node(
      graph_utils::FindGraphNodeWithOp("MapAndBatchDataset", output));
  EXPECT_EQ(map_and_batch_node.input_size(), 5);
  EXPECT_EQ(map_and_batch_node.input(0), map_node->input(0));
  EXPECT_EQ(map_and_batch_node.input(1), map_node->input(1));
  EXPECT_EQ(map_and_batch_node.input(2), batch_node->input(1));
  NodeDef num_parallel_calls_node2 = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(3), output));
  EXPECT_EQ(num_parallel_calls_node2.attr().at("value").tensor().int64_val(0),
            2);
  NodeDef drop_remainder_node = output.node(
      graph_utils::FindGraphNodeWithName(map_and_batch_node.input(4), output));
  EXPECT_EQ(drop_remainder_node.attr().at("value").tensor().bool_val(0), false);
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("f"),
                                 map_node->attr().at("f")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("Targuments"),
                                 map_node->attr().at("Targuments")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(map_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(MapAndBatchFusionTest, NoChange) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);

  NodeDef *batch_size_node =
      graph_utils::AddScalarConstNode<int64_t>(5, &graph);
  std::vector<string> batch_inputs(2);
  batch_inputs[0] = range_node->name();
  batch_inputs[1] = batch_size_node->name();
  std::vector<std::pair<string, AttrValue>> batch_attrs(2);
  AttrValue shapes_attr;
  SetAttrValue("output_shapes", &shapes_attr);
  batch_attrs[0] = std::make_pair("output_shapes", shapes_attr);
  AttrValue types_attr;
  SetAttrValue("output_types", &types_attr);
  batch_attrs[1] = std::make_pair("output_types", types_attr);
  graph_utils::AddNode("", "BatchDataset", batch_inputs, batch_attrs, &graph);

  MapAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_TRUE(graph_utils::Compare(*graph.graph(), output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
