/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/inject_prefetch.h"

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

using test::function::NDef;

constexpr char kOptionsDataset[] = "OptionsDataset";
constexpr char kParallelMapDataset[] = "ParallelMapDatasetV2";
constexpr char kPrefetchDataset[] = "PrefetchDataset";

Status Optimize(InjectPrefetch &optimizer, const GrapplerItem &item,
                GraphDef *output, bool autotune) {
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

Status OptimizeWithInjectPrefetch(const GrapplerItem &item, GraphDef *output,
                                  bool autotune) {
  InjectPrefetch optimizer;
  return Optimize(optimizer, item, output, autotune);
}

Status OptimizeWithInjectPrefetchEligible(const GrapplerItem &item,
                                          GraphDef *output, bool autotune) {
  InjectPrefetchEligible optimizer;
  return Optimize(optimizer, item, output, autotune);
}

class InjectPrefetchParameterizedTest : public ::testing::TestWithParam<bool> {
};

TEST_P(InjectPrefetchParameterizedTest, TestAutotuneSetting) {
  const bool autotune = GetParam();

  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("Sink", "Identity", {"range"}, {})});

  item.fetch.push_back("Sink");

  // Test inject_prefetch
  GraphDef inject_prefetch_output;
  TF_ASSERT_OK(
      OptimizeWithInjectPrefetch(item, &inject_prefetch_output, autotune));
  EXPECT_EQ(autotune, graph_utils::ContainsNodeWithOp(kPrefetchDataset,
                                                      inject_prefetch_output));
  EXPECT_EQ(autotune, graph_utils::ContainsGraphNodeWithName(
                          "inject/prefetch_range", inject_prefetch_output));

  // Test inject_prefetch_eligible
  GraphDef inject_prefetch_eligible_output;
  TF_ASSERT_OK(OptimizeWithInjectPrefetchEligible(
      item, &inject_prefetch_eligible_output, autotune));
  EXPECT_EQ(false, graph_utils::ContainsNodeWithOp(
                       kPrefetchDataset, inject_prefetch_eligible_output));
  EXPECT_EQ(false,
            graph_utils::ContainsGraphNodeWithName(
                "inject/prefetch_range", inject_prefetch_eligible_output));
  EXPECT_EQ(item.graph.DebugString(),
            inject_prefetch_eligible_output.DebugString());
}

INSTANTIATE_TEST_SUITE_P(AutotuneSetting, InjectPrefetchParameterizedTest,
                         ::testing::Values(false, true));

TEST(InjectPrefetchTest, FromFunctionDef) {
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("Sink", "_Retval", {"range"}, {})});

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectPrefetch(item, &output, true));
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp(kPrefetchDataset, output));
}

TEST(InjectPrefetchTest, AlreadyPrefetched) {
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("prefetch", kPrefetchDataset, {"range"}, {}),
       NDef("Sink", "Identity", {"prefetch"}, {})});

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectPrefetch(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(kPrefetchDataset, output));
  EXPECT_EQ(6, output.node_size());
}

TEST(InjectPrefetchTest, AlreadyParallelMap) {
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("parallel_map", kParallelMapDataset, {"range"},
            {{"f", "__inference_Dataset_map_normalize_8232"},
             {"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("Sink", "Identity", {"parallel_map"}, {})});

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectPrefetch(item, &output, true));
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp(kPrefetchDataset, output));
  EXPECT_EQ(6, output.node_size());
}

TEST(InjectPrefetchTest, OptionsFollowedByPrefetched) {
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("prefetch", kPrefetchDataset, {"range"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("options", kOptionsDataset, {"prefetch"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("Sink", "Identity", {"options"}, {})});

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithInjectPrefetch(item, &output, true));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("inject/prefetch_options",
                                                      output));
  EXPECT_EQ(7, output.node_size());
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
