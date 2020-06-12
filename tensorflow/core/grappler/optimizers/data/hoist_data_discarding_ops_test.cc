/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/hoist_data_discarding_ops.h"

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

TEST(HoistDataDiscardingOpsTest, ExampleOps) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {
          {"output_shapes", gtl::ArraySlice<TensorShape>{}},
          {"output_types", gtl::ArraySlice<DataType>{}}, 
      }),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelMapNode("map", "range",
                                              "num_parallel_calls", "XTimesTwo",
                                              /*sloppy=*/false),
       NDef("dummy_memory_cache", "DummyMemoryCache", {}, {}),
       graph_tests_utils::MakeCacheV2Node("cache", "map", "", "dummy_memory_cache"),
       NDef("take_count", "Const", {},
            {{"value", 5}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeTakeNode("take", "cache", "take_count"),
       NDef("skip_count", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeSkipNode("skip", "take", "skip_count"),
       NDef("batch_size", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", true}, {"dtype", DT_BOOL}}),
       graph_tests_utils::MakeMapAndBatchNode("map_and_batch", "skip",
                                              "batch_size", "drop_remainder",
                                              "XTimesTwo"),
       NDef("num_shards", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("index", "Const", {},
            {{"value", 0}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeShardNode("shard", "map_and_batch",
                                        "num_shards", "index")},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  HoistDataDiscardingOps optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("hoist_data_discarding_ops/take", output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("hoist_data_discarding_ops/skip", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("hoist_data_discarding_ops/shard", output));

  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("take", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("skip", output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("shard", output));

  MutableGraphView graph(&output);
  EXPECT_TRUE(graph_utils::GetInputNode(*graph.GetNode("hoist_data_discarding_ops/take"),
                                        graph)->name() == "range");
  EXPECT_TRUE(graph_utils::GetInputNode(*graph.GetNode("hoist_data_discarding_ops/skip"),
                                        graph)->name() == "hoist_data_discarding_ops/take");
  EXPECT_TRUE(graph_utils::GetInputNode(*graph.GetNode("map_and_batch"), graph)->name() == "cache");
}

}  // namespace
}  // namsepace grappler
}  // namespace tensorflow
