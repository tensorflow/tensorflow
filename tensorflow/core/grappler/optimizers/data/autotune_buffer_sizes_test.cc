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

Status OptimizeWithAutotuneBufferSizes(const GrapplerItem &item,
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

INSTANTIATE_TEST_SUITE_P(Test, AutotuneSetting, ::testing::Values(false, true));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
