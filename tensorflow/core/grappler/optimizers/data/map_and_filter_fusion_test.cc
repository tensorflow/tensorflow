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

#include "tensorflow/core/grappler/optimizers/data/map_and_filter_fusion.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {
using graph_tests_utils::MakeFilterNode;
using graph_tests_utils::MakeMapNode;
using graph_tests_utils::MakeParallelMapNode;

TEST(MapAndFilterFusionTest, FuseMapAndFilter) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeMapNode("map", "range"), MakeFilterNode("filter", "map")},
      // FunctionLib
      {
          test::function::XTimesTwo(),
          test::function::IsZero(),
      });

  MapAndFilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter", output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapDataset", output));

  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("FilterByLastComponentDataset", output));
}

TEST(MapAndFilterFusionTest, FuseParallelMapAndFilter) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 3}, {"dtype", "DT_INT32"}}),
       MakeParallelMapNode("map", "range", "num_parallel_calls", "XTimesTwo",
                           /*sloppy=*/false),
       MakeFilterNode("filter", "map")},
      // FunctionLib
      {
          test::function::XTimesTwo(),
          test::function::IsZero(),
      });

  MapAndFilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter", output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ParallelMapDataset", output))
      << output.DebugString();
  auto& map_node = output.node(
      graph_utils::FindGraphNodeWithOp("ParallelMapDataset", output));
  EXPECT_FALSE(map_node.attr().at("sloppy").b()) << map_node.DebugString();
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("FilterByLastComponentDataset", output))
      << output.DebugString();
}

TEST(MapAndFilterFusionTest, FuseMapAndFilterWithExtraChild) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeMapNode("map", "range"), MakeFilterNode("filter", "map"),
       NDef("cache", "CacheDataset", {"filter", "filename"}, {})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
          test::function::IsZero(),
      });

  MapAndFilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter", output));
  ASSERT_TRUE(graph_utils::ContainsNodeWithOp("MapDataset", output));
  ASSERT_TRUE(
      graph_utils::ContainsNodeWithOp("FilterByLastComponentDataset", output));
  ASSERT_TRUE(graph_utils::ContainsNodeWithOp("CacheDataset", output));

  int map_id = graph_utils::FindGraphNodeWithOp("MapDataset", output);
  auto& map_node = output.node(map_id);
  ASSERT_EQ(map_node.input_size(), 1);
  EXPECT_EQ(map_node.input(0), "range");

  int filter_by_component_id =
      graph_utils::FindGraphNodeWithOp("FilterByLastComponentDataset", output);
  auto& filter_by_component = output.node(filter_by_component_id);
  ASSERT_EQ(filter_by_component.input_size(), 1);
  EXPECT_EQ(filter_by_component.input(0), map_node.name());

  int cache_id = graph_utils::FindGraphNodeWithOp("CacheDataset", output);
  auto& cache_node = output.node(cache_id);
  ASSERT_EQ(cache_node.input_size(), 2);
  EXPECT_EQ(cache_node.input(0), filter_by_component.name());
}

TEST(MapAndFilterFusionTest, FuseParallelMapAndFilterWithExtraChild) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 3}, {"dtype", "DT_INT32"}}),
       MakeParallelMapNode("map", "range", "num_parallel_calls", "XTimesTwo",
                           /*sloppy=*/true),
       MakeFilterNode("filter", "map"),
       NDef("cache", "CacheDataset", {"filter", "filename"}, {})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
          test::function::IsZero(),
      });

  MapAndFilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter", output));
  ASSERT_TRUE(graph_utils::ContainsNodeWithOp("ParallelMapDataset", output));
  ASSERT_TRUE(
      graph_utils::ContainsNodeWithOp("FilterByLastComponentDataset", output));
  ASSERT_TRUE(graph_utils::ContainsNodeWithOp("CacheDataset", output));

  int map_id = graph_utils::FindGraphNodeWithOp("ParallelMapDataset", output);
  auto& map_node = output.node(map_id);
  ASSERT_EQ(map_node.input_size(), 2);
  EXPECT_EQ(map_node.input(0), "range");
  EXPECT_EQ(map_node.input(1), "num_parallel_calls");

  int filter_by_component_id =
      graph_utils::FindGraphNodeWithOp("FilterByLastComponentDataset", output);
  auto& filter_by_component = output.node(filter_by_component_id);
  ASSERT_EQ(filter_by_component.input_size(), 1);
  EXPECT_EQ(filter_by_component.input(0), map_node.name());

  int cache_id = graph_utils::FindGraphNodeWithOp("CacheDataset", output);
  auto& cache_node = output.node(cache_id);
  ASSERT_EQ(cache_node.input_size(), 2);
  EXPECT_EQ(cache_node.input(0), filter_by_component.name());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
