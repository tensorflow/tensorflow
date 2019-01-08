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

#include "tensorflow/core/grappler/optimizers/data/map_parallelization.h"

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

using graph_tests_utils::MakeMapNode;
const char stateless_fun_name[] = "XTimesTwo";
const char stateful_fun_name[] = "RandomUniform";

TEST(MapParallelizationTest, ParallelizeSimpleMap) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeMapNode("map1", "range", stateless_fun_name)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MapParallelization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ParallelMapDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map2", output));
}

TEST(MapParallelization, ParallelizeAssert) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeMapNode("map1", "range", stateful_fun_name),
       MakeMapNode("map2", "map1", stateless_fun_name),
       NDef("cache", "CacheDataset", {"map2", "filename"}, {})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
          test::function::RandomUniform(),
      });

  MapParallelization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ParallelMapDataset", output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("map1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map2", output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
