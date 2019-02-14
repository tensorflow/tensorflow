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

#include "tensorflow/core/grappler/optimizers/data/make_numa_aware.h"

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

TEST(MakeNumaAwareTest, ReplaceSimple) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {
          NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
          NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
          NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
          NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
          NDef("batch_size", "Const", {}, {{"value", 3}, {"dtype", DT_INT32}}),
          NDef("num_parallel_calls", "Const", {},
               {{"value", 5}, {"dtype", DT_INT32}}),
          NDef("drop_remainder", "Const", {},
               {{"value", 0}, {"dtype", DT_BOOL}}),
          graph_tests_utils::MakeMapAndBatchNode(
              "map_and_batch", "range", "batch_size", "num_parallel_calls",
              "drop_remainder"),
      },
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MakeNumaAware optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map_and_batch", output));
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp("ExperimentalMapAndBatchDataset",
                                               output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(
      "ExperimentalNumaMapAndBatchDataset", output));
}

TEST(MapAndBatchNumaAawareReplacementTest, ReplaceWithExtraChild) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {
          NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
          NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
          NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
          NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
          NDef("batch_size", "Const", {}, {{"value", 3}, {"dtype", DT_INT32}}),
          NDef("num_parallel_calls", "Const", {},
               {{"value", 5}, {"dtype", DT_INT32}}),
          NDef("drop_remainder", "Const", {},
               {{"value", 0}, {"dtype", DT_BOOL}}),
          graph_tests_utils::MakeMapAndBatchNode(
              "map_and_batch", "range", "batch_size", "num_parallel_calls",
              "drop_remainder"),
          NDef("cache", "CacheDataset", {"map_and_batch"}, {}),
      },
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MakeNumaAware optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map_and_batch", output));
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp("ExperimentalMapAndBatchDataset",
                                               output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(
      "ExperimentalNumaMapAndBatchDataset", output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("CacheDataset", output));

  int numa_map_and_batch_component_id = graph_utils::FindGraphNodeWithOp(
      "ExperimentalNumaMapAndBatchDataset", output);
  auto& numa_map_and_batch_component =
      output.node(numa_map_and_batch_component_id);
  EXPECT_EQ(numa_map_and_batch_component.input(0), "range");

  int cache_id = graph_utils::FindGraphNodeWithOp("CacheDataset", output);
  auto& cache_node = output.node(cache_id);
  EXPECT_EQ(cache_node.input(0), numa_map_and_batch_component.name());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
