/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/autotune_buffer_sizes.h"

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

absl::Status OptimizeWithAutotuneBufferSizes(const GrapplerItem &item,
                                             GraphDef *output, bool autotune) {
  AutotuneBufferSizes optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

class SimpleInject : public ::testing::TestWithParam<string> {};

TEST_P(SimpleInject, AutotuneBufferSizesTest) {
  const string async_dataset = GetParam();
  using test::function::NDef;
  GrapplerItem item;
  if (async_dataset == "map") {
    item.graph = test::function::GDef(
        {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
         NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
         NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
         NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
         NDef("num_parallel_calls", "Const", {},
              {{"value", 1}, {"dtype", DT_INT32}}),
         graph_tests_utils::MakeParallelMapNode(
             "map", "range", "num_parallel_calls", "XTimesTwo",
             /*sloppy=*/false)},
        // FunctionLib
        {
            test::function::XTimesTwo(),
        });
  } else if (async_dataset == "interleave") {
    item.graph = test::function::GDef(
        {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
         NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
         NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
         NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
         NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
         NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
         NDef("num_parallel_calls", "Const", {},
              {{"value", 1}, {"dtype", DT_INT32}}),
         graph_tests_utils::MakeParallelInterleaveV2Node(
             "interleave", "range", "cycle_length", "block_length",
             "num_parallel_calls", "XTimesTwo", /*sloppy=*/false)},
        // FunctionLib
        {
            test::function::XTimesTwo(),
        });
  } else if (async_dataset == "map_and_batch") {
    item.graph = test::function::GDef(
        {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
         NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
         NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
         NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
         NDef("batch_size", "Const", {}, {{"value", 32}, {"dtype", DT_INT64}}),
         NDef("num_parallel_calls", "Const", {},
              {{"value", 1}, {"dtype", DT_INT64}}),
         NDef("drop_remainder", "Const", {},
              {{"value", false}, {"dtype", DT_BOOL}}),
         graph_tests_utils::MakeMapAndBatchNode(
             "map_and_batch", "range", "batch_size", "num_parallel_calls",
             "drop_remainder", "XTimesTwo")},
        // FunctionLib
        {
            test::function::XTimesTwo(),
        });
  }

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithAutotuneBufferSizes(item, &output, true));

  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("PrefetchDataset", output));
  int index = graph_utils::FindGraphNodeWithOp("PrefetchDataset", output);
  const NodeDef prefetch_node = output.node(index);
  EXPECT_TRUE(prefetch_node.attr().find("legacy_autotune") ==
              prefetch_node.attr().end());
  EXPECT_EQ(prefetch_node.input_size(), 2);
  NodeDef async_node = output.node(
      graph_utils::FindGraphNodeWithName(prefetch_node.input(0), output));
  EXPECT_EQ(async_node.name(), async_dataset);
  NodeDef buffer_size_val = output.node(
      graph_utils::FindGraphNodeWithName(prefetch_node.input(1), output));
  EXPECT_EQ(buffer_size_val.attr().at("value").tensor().int64_val(0), -1);
}

INSTANTIATE_TEST_SUITE_P(Test, SimpleInject,
                         ::testing::Values("map", "interleave",
                                           "map_and_batch"));

class AutotuneSetting : public ::testing::TestWithParam<bool> {};

TEST_P(AutotuneSetting, AutotuneBufferSizesTest) {
  const bool autotune = GetParam();

  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelMapNode("map", "range",
                                              "num_parallel_calls", "XTimesTwo",
                                              /*sloppy=*/false)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithAutotuneBufferSizes(item, &output, autotune));
  EXPECT_EQ(graph_utils::ContainsNodeWithOp("PrefetchDataset", output),
            autotune);
}

class MultipleNodes
    : public ::testing::TestWithParam<std::tuple<bool, int64_t>> {};

TEST_P(MultipleNodes, AutotuneBufferSizesTest) {
  const bool legacy_autotune = std::get<0>(GetParam());
  const int64_t initial_buffer_size = std::get<1>(GetParam());

  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef *start_val = graph_utils::AddScalarConstNode<int64_t>(0, &graph);
  NodeDef *stop_val = graph_utils::AddScalarConstNode<int64_t>(10, &graph);
  NodeDef *step_val = graph_utils::AddScalarConstNode<int64_t>(1, &graph);

  std::vector<string> range_inputs(3);
  range_inputs[0] = start_val->name();
  range_inputs[1] = stop_val->name();
  range_inputs[2] = step_val->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("range", "RangeDataset",
                                             range_inputs, range_attrs, &graph);

  NodeDef *parallelism_val =
      graph_utils::AddScalarConstNode<int64_t>(1, &graph);
  std::vector<string> map_inputs1(2);
  map_inputs1[0] = range_node->name();
  map_inputs1[1] = parallelism_val->name();
  std::vector<std::pair<string, AttrValue>> map_attrs(4);
  AttrValue attr_val;
  SetAttrValue("value", &attr_val);
  map_attrs[0] = std::make_pair("f", attr_val);
  map_attrs[1] = std::make_pair("Targuments", attr_val);
  map_attrs[2] = std::make_pair("output_types", attr_val);
  map_attrs[3] = std::make_pair("output_shapes", attr_val);
  NodeDef *map_node1 = graph_utils::AddNode("map1", "ParallelMapDatasetV2",
                                            map_inputs1, map_attrs, &graph);

  NodeDef *buffer_size_val =
      graph_utils::AddScalarConstNode<int64_t>(initial_buffer_size, &graph);
  std::vector<string> prefetch_inputs(2);
  prefetch_inputs[0] = map_node1->name();
  prefetch_inputs[1] = buffer_size_val->name();
  std::vector<std::pair<string, AttrValue>> prefetch_attrs(4);
  AttrValue legacy_autotune_attr;
  SetAttrValue(legacy_autotune, &legacy_autotune_attr);
  AttrValue buffer_size_min_attr;
  SetAttrValue(0, &buffer_size_min_attr);
  prefetch_attrs[0] = std::make_pair("legacy_autotune", legacy_autotune_attr);
  prefetch_attrs[1] = std::make_pair("buffer_size_min", buffer_size_min_attr);
  prefetch_attrs[2] = std::make_pair("output_types", attr_val);
  prefetch_attrs[3] = std::make_pair("output_shapes", attr_val);
  NodeDef *prefetch_node = graph_utils::AddNode(
      "prefetch", "PrefetchDataset", prefetch_inputs, prefetch_attrs, &graph);

  std::vector<string> map_inputs2(2);
  map_inputs2[0] = prefetch_node->name();
  map_inputs2[1] = parallelism_val->name();
  NodeDef *map_node2 = graph_utils::AddNode("map2", "ParallelMapDatasetV2",
                                            map_inputs2, map_attrs, &graph);

  std::vector<string> map_inputs3(1);
  map_inputs3[0] = map_node2->name();
  graph_utils::AddNode("map3", "MapDataset", map_inputs3, map_attrs, &graph);

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithAutotuneBufferSizes(item, &output, true));

  std::vector<int> prefetch_indices =
      graph_utils::FindAllGraphNodesWithOp("PrefetchDataset", output);
  EXPECT_EQ(prefetch_indices.size(), 2);

  NodeDef new_map_node3 =
      output.node(graph_utils::FindGraphNodeWithName("map3", output));

  NodeDef new_prefetch_node2 = output.node(
      graph_utils::FindGraphNodeWithName(new_map_node3.input(0), output));
  EXPECT_EQ(new_prefetch_node2.op(), "PrefetchDataset");
  EXPECT_EQ(new_prefetch_node2.input_size(), 2);
  EXPECT_TRUE(new_prefetch_node2.attr().find("legacy_autotune") ==
              new_prefetch_node2.attr().end());
  EXPECT_TRUE(new_prefetch_node2.attr().find("buffer_size_min") ==
              new_prefetch_node2.attr().end());
  NodeDef new_buffer_size_val2 = output.node(
      graph_utils::FindGraphNodeWithName(new_prefetch_node2.input(1), output));
  EXPECT_EQ(new_buffer_size_val2.attr().at("value").tensor().int64_val(0), -1);

  NodeDef new_map_node2 = output.node(
      graph_utils::FindGraphNodeWithName(new_prefetch_node2.input(0), output));
  EXPECT_EQ(new_map_node2.name(), "map2");

  NodeDef new_prefetch_node1 = output.node(
      graph_utils::FindGraphNodeWithName(new_map_node2.input(0), output));
  EXPECT_EQ(new_prefetch_node1.op(), "PrefetchDataset");
  EXPECT_EQ(new_prefetch_node1.input_size(), 2);
  EXPECT_EQ(new_prefetch_node1.attr().at("legacy_autotune").b(),
            legacy_autotune);
  EXPECT_EQ(new_prefetch_node1.attr().at("buffer_size_min").i(),
            (initial_buffer_size == -1 ? 0 : initial_buffer_size));
  NodeDef new_buffer_size_val1 = output.node(
      graph_utils::FindGraphNodeWithName(new_prefetch_node1.input(1), output));
  EXPECT_EQ(new_buffer_size_val1.attr().at("value").tensor().int64_val(0), -1);

  NodeDef new_map_node1 = output.node(
      graph_utils::FindGraphNodeWithName(new_prefetch_node1.input(0), output));
  EXPECT_EQ(new_map_node1.name(), "map1");
}

INSTANTIATE_TEST_SUITE_P(Test, MultipleNodes,
                         ::testing::Combine(::testing::Values(true, false),
                                            ::testing::Values(-1, 3)));

INSTANTIATE_TEST_SUITE_P(Test, AutotuneSetting, ::testing::Values(false, true));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
