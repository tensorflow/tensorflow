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

#include "tensorflow/core/grappler/optimizers/data/unbatch_and_batch_fusion.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

TEST(UnbatchAndBatchFusionTest, FuseUnbatchAndBatchNodesIntoOne) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);

  NodeDef *unbatch_node;
  {
    std::vector<string> unbatch_inputs(1);
    unbatch_inputs[0] = range_node->name();
    std::vector<std::pair<string, AttrValue>> unbatch_attrs;
    unbatch_node =
        graph_utils::AddNode("", "UnbatchDataset", unbatch_inputs, unbatch_attrs, &graph);
  }

  NodeDef *batch_size_node = graph_utils::AddScalarConstNode<int64>(5, &graph);
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(2);
    batch_inputs[0] = unbatch_node->name();
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

  UnbatchAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(unbatch_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ExperimentalUnbatchAndBatchDataset", output));
  NodeDef unbatch_and_batch_node = output.node(
      graph_utils::FindGraphNodeWithOp("ExperimentalUnbatchAndBatchDataset", output));
  EXPECT_EQ(unbatch_and_batch_node.input_size(), 3);

  EXPECT_EQ(unbatch_and_batch_node.input(0), unbatch_node->input(0));
  EXPECT_EQ(unbatch_and_batch_node.input(1), batch_node->input(1));
  NodeDef drop_remainder_node = output.node(
      graph_utils::FindGraphNodeWithName(unbatch_and_batch_node.input(2), output));
  EXPECT_EQ(drop_remainder_node.attr().at("value").tensor().bool_val(0), false);
  EXPECT_TRUE(AreAttrValuesEqual(unbatch_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(unbatch_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(UnbatchAndBatchFusionTest, FuseUnbatchAndBatchV2NodesIntoOne) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  NodeDef *start_node = graph_utils::AddScalarConstNode<int64>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);

  NodeDef *unbatch_node;
  {
    std::vector<string> unbatch_inputs(1);
    unbatch_inputs[0] = range_node->name();
    std::vector<std::pair<string, AttrValue>> unbatch_attrs;
    unbatch_node =
        graph_utils::AddNode("", "UnbatchDataset", unbatch_inputs, unbatch_attrs, &graph);
  }

  NodeDef *batch_size_node = graph_utils::AddScalarConstNode<int64>(5, &graph);
  NodeDef *drop_remainder_node =
      graph_utils::AddScalarConstNode<bool>(true, &graph);
  NodeDef *batch_node;
  {
    std::vector<string> batch_inputs(3);
    batch_inputs[0] = unbatch_node->name();
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

  UnbatchAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(unbatch_node->name(), output));
  EXPECT_FALSE(
      graph_utils::ContainsGraphNodeWithName(batch_node->name(), output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ExperimentalUnbatchAndBatchDataset", output));
  NodeDef unbatch_and_batch_node = output.node(
      graph_utils::FindGraphNodeWithOp("ExperimentalUnbatchAndBatchDataset", output));
  EXPECT_EQ(unbatch_and_batch_node.input_size(), 3);
  EXPECT_EQ(unbatch_and_batch_node.input(0), unbatch_node->input(0));
  EXPECT_EQ(unbatch_and_batch_node.input(1), batch_node->input(1));
  EXPECT_EQ(unbatch_and_batch_node.input(2), batch_node->input(2));
  EXPECT_TRUE(AreAttrValuesEqual(unbatch_and_batch_node.attr().at("output_shapes"),
                                 batch_node->attr().at("output_shapes")));
  EXPECT_TRUE(AreAttrValuesEqual(unbatch_and_batch_node.attr().at("output_types"),
                                 batch_node->attr().at("output_types")));
}

TEST(UnbatchAndBatchFusionTest, NoChange) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef *start_node = graph_utils::AddScalarConstNode<int64>(0, &graph);
  NodeDef *stop_node = graph_utils::AddScalarConstNode<int64>(10, &graph);
  NodeDef *step_node = graph_utils::AddScalarConstNode<int64>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_node->name();
  range_inputs[1] = stop_node->name();
  range_inputs[2] = step_node->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);

  NodeDef *batch_size_node = graph_utils::AddScalarConstNode<int64>(5, &graph);
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

  UnbatchAndBatchFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_TRUE(graph_utils::Compare(*graph.graph(), output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
