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

#include "tensorflow/core/grappler/optimizers/data/batch_parallelization.h"

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

Status OptimizeWithBatchParallelization(const GrapplerItem& item,
                                        GraphDef* output, bool autotune) {
  BatchParallelization optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

using graph_tests_utils::MakeBatchV2Node;

class AutotuneSetting : public ::testing::TestWithParam<bool> {};

TEST_P(AutotuneSetting, BatchParallelizationTest) {
  const bool autotune = GetParam();

  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       MakeBatchV2Node("batch", "range", "batch_size", "drop_remainder",
                       /*parallel_copy=*/false),
       NDef("Sink", "Identity", {"batch"}, {})},
      // FunctionLib
      {});

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithBatchParallelization(item, &output, autotune));
  EXPECT_EQ(graph_utils::ContainsNodeWithOp("ParallelBatchDataset", output),
            autotune);
  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName("batch", output), !autotune);
}

INSTANTIATE_TEST_SUITE_P(Test, AutotuneSetting, ::testing::Values(false, true));

class FromFunctionDef : public ::testing::TestWithParam<string> {};

TEST_P(FromFunctionDef, BatchParallelizationTest) {
  const string op = GetParam();
  bool from_function_def = (op == "_Retval");

  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       MakeBatchV2Node("batch", "range", "batch_size", "drop_remainder",
                       /*parallel_copy=*/false),
       NDef("Sink", op, {"batch"}, {})},
      // FunctionLib
      {});

  item.fetch.push_back("Sink");

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithBatchParallelization(item, &output, true));
  EXPECT_EQ(graph_utils::ContainsNodeWithOp("ParallelBatchDataset", output),
            !from_function_def);
  EXPECT_EQ(graph_utils::ContainsGraphNodeWithName("batch", output),
            from_function_def);
}

INSTANTIATE_TEST_SUITE_P(Test, FromFunctionDef,
                         ::testing::Values("Identity", "_Retval"));

// Test the input and attr values after applying the optimization.
class ValueRewrites : public ::testing::TestWithParam<bool> {};

TEST_P(ValueRewrites, BatchParallelizationTest) {
  const bool parallel_copy = GetParam();
  using test::function::NDef;
  GrapplerItem item;

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       MakeBatchV2Node("batch", "range", "batch_size", "drop_remainder",
                       parallel_copy),
       NDef("Sink", "Identity", {"batch"}, {})},
      // FunctionLib
      {});

  item.fetch.push_back("Sink");

  NodeDef batch =
      item.graph.node(graph_utils::FindGraphNodeWithName("batch", item.graph));
  EXPECT_TRUE(batch.attr().find("parallel_copy") != batch.attr().end());

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithBatchParallelization(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ParallelBatchDataset", output));

  NodeDef parallel_batch = output.node(
      graph_utils::FindGraphNodeWithOp("ParallelBatchDataset", output));
  EXPECT_EQ(parallel_batch.input_size(), 4);
  EXPECT_EQ(parallel_batch.input(0), "range");
  EXPECT_EQ(parallel_batch.input(1), "batch_size");
  EXPECT_EQ(parallel_batch.input(3), "drop_remainder");
  EXPECT_EQ(parallel_batch.attr().at("parallel_copy").b(), parallel_copy);

  NodeDef parallelism_val = output.node(
      graph_utils::FindGraphNodeWithName(parallel_batch.input(2), output));
  EXPECT_EQ(parallelism_val.attr().at("value").tensor().int64_val(0), -1);
}

INSTANTIATE_TEST_SUITE_P(Test, ValueRewrites, ::testing::Values(false, true));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
