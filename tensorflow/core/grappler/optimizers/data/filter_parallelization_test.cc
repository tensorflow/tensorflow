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

#include "tensorflow/core/grappler/optimizers/data/filter_parallelization.h"

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

Status OptimizeWithFilterParallelization(const GrapplerItem& item,
                                         GraphDef* output, bool autotune) {
  FilterParallelization optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

using graph_tests_utils::MakeFilterNode;
const char stateless_fun_name[] = "NonZero";
const char stateful_fun_name[] = "RandomUniformLess";

class AutotuneSetting : public ::testing::TestWithParam<bool> {};

TEST_P(AutotuneSetting, FilterParallelizationTest) {
  const bool autotune = GetParam();

  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter", "range", stateless_fun_name),
       NDef("Sink", "Identity", {"filter"}, {})},
      // FunctionLib
      {
          test::function::NonZero(),
      });

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithFilterParallelization(item, &output, autotune));
  EXPECT_EQ(graph_utils::ContainsNodeWithOp("ParallelFilterDataset", output),
            autotune);
  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName("filter", output),
            !autotune);
}

INSTANTIATE_TEST_SUITE_P(Test, AutotuneSetting, ::testing::Values(false, true));

class FromFunctionDef : public ::testing::TestWithParam<string> {};

TEST_P(FromFunctionDef, FilterParallelizationTest) {
  const string op = GetParam();
  bool from_function_def = (op == "_Retval");

  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter", "range", stateless_fun_name),
       NDef("Sink", op, {"filter"}, {})},
      // FunctionLib
      {
          test::function::NonZero(),
      });

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithFilterParallelization(item, &output, true));
  EXPECT_EQ(graph_utils::ContainsNodeWithOp("ParallelFilterDataset", output),
            !from_function_def);
  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName("filter", output),
            from_function_def);
}

INSTANTIATE_TEST_SUITE_P(Test, FromFunctionDef,
                         ::testing::Values("Identity", "_Retval"));

TEST(ParallelizeAssert, FilterParallelizationTest) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter1", "range", stateful_fun_name),
       MakeFilterNode("filter2", "filter1", stateless_fun_name),
       NDef("cache", "CacheDataset", {"filter2", "filename"}, {}),
       NDef("Sink", "Identity", {"cache"}, {})},
      // FunctionLib
      {
          test::function::NonZero(),
          test::function::RandomUniformLess(),
      });

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithFilterParallelization(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ParallelFilterDataset", output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("FilterDataset", output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("filter1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter2", output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
