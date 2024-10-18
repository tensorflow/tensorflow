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

#include "tensorflow/core/grappler/optimizers/data/map_fusion.h"

#include <functional>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/errors.h"

namespace tensorflow {
namespace grappler {
namespace {

using graph_tests_utils::MakeMapNode;
using graph_tests_utils::MakeParallelMapV2Node;

constexpr char kConstOpName[] = "Const";

NodeDef CreateScalarConstNodeHelper(
    const std::string& node_name, DataType dtype,
    const std::function<void(TensorProto*)>& add_value) {
  NodeDef node;
  node.set_op(kConstOpName);
  node.set_name(node_name);

  (*node.mutable_attr())["dtype"].set_type(dtype);
  auto tensor = std::make_unique<tensorflow::TensorProto>();
  auto tensor_shape = std::make_unique<tensorflow::TensorShapeProto>();
  tensor->set_allocated_tensor_shape(tensor_shape.release());
  tensor->set_dtype(dtype);
  add_value(tensor.get());
  (*node.mutable_attr())["value"].set_allocated_tensor(tensor.release());

  return node;
}

absl::Status OptimizeWithMapFusion(const GrapplerItem& item, GraphDef* output,
                                   bool autotune) {
  MapFusion optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (autotune) {
    (*config.mutable_parameter_map())["autotune"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["autotune"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

class AutotuneSetting : public ::testing::TestWithParam<bool> {};

TEST_P(AutotuneSetting, MapFusionTest) {
  const bool autotune = GetParam();

  using test::function::NDef;
  GrapplerItem item;
  NodeDef num_parallel_calls_node = CreateScalarConstNodeHelper(
      "num_parallel_calls", DT_INT64,
      [](TensorProto* proto) { proto->add_int64_val(-1); });
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       num_parallel_calls_node,
       MakeParallelMapV2Node("map1", "range", num_parallel_calls_node.name(),
                             "XTimesTwo", "default",
                             /*use_unbounded_threadpool=*/false),
       MakeParallelMapV2Node("map2", "map1", num_parallel_calls_node.name(),
                             "XTimesTwo", "default",
                             /*use_unbounded_threadpool=*/false)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MapFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapFusion(item, &output, autotune));
  if (autotune) {
    EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map1", output));
    EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map2", output));
  } else {
    EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("map1", output));
    EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("map2", output));
  }
}

INSTANTIATE_TEST_SUITE_P(Test, AutotuneSetting, ::testing::Values(false, true));

TEST(MapFusionTest, FuseTwoMapNodesIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeMapNode("map1", "range"), MakeMapNode("map2", "map1")},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MapFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapFusion(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map2", output));
}

TEST(MapFusionTest, FuseThreeNodesIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeMapNode("map1", "range"), MakeMapNode("map2", "map1"),
       MakeMapNode("map3", "map2"),
       NDef("cache", "CacheDataset", {"map3", "filename"}, {})},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MapFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapFusion(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("MapDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map2", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map3", output));
}

TEST(MapFusionTest, FuseTwoParallelMapNodesIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  NodeDef num_parallel_calls_node = CreateScalarConstNodeHelper(
      "num_parallel_calls", DT_INT64,
      [](TensorProto* proto) { proto->add_int64_val(-1); });
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       num_parallel_calls_node,
       MakeParallelMapV2Node("map1", "range", num_parallel_calls_node.name(),
                             "XTimesTwo", "default",
                             /*use_unbounded_threadpool=*/false),
       MakeParallelMapV2Node("map2", "map1", num_parallel_calls_node.name(),
                             "XTimesTwo", "default",
                             /*use_unbounded_threadpool=*/false)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MapFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapFusion(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("ParallelMapDatasetV2", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("map2", output));
}

TEST(MapFusionTest, NoChange_UnboundedThreadpoolParallelMap) {
  using test::function::NDef;
  GrapplerItem item;
  NodeDef num_parallel_calls_node = CreateScalarConstNodeHelper(
      "num_parallel_calls", DT_INT64,
      [](TensorProto* proto) { proto->add_int64_val(-1); });
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       num_parallel_calls_node,
       MakeParallelMapV2Node("map1", "range", num_parallel_calls_node.name(),
                             "XTimesTwo", "default",
                             /*use_unbounded_threadpool=*/true),
       MakeParallelMapV2Node("map2", "map1", num_parallel_calls_node.name(),
                             "XTimesTwo", "default",
                             /*use_unbounded_threadpool=*/false)},
      // FunctionLib
      {
          test::function::XTimesTwo(),
      });

  MapFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapFusion(item, &output, true));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("map1", output));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName("map2", output));
}

TEST(MapFusionTest, FusedNodesAndFunctionsAreNamedAfterOldNodesAndFunctions) {
  using test::function::NDef;
  NodeDef num_parallel_calls_node = CreateScalarConstNodeHelper(
      "num_parallel_calls", DT_INT64,
      [](TensorProto* proto) { proto->add_int64_val(-1); });
  auto graph = [&num_parallel_calls_node](
                   const std::string& parent_map_node_name,
                   const std::string& map_node_name,
                   const std::string& parent_function_name,
                   const std::string& function_name) {
    FunctionDef parent_fn = test::function::XTimesTwo();
    FunctionDef fn = test::function::XTimesTwo();
    parent_fn.mutable_signature()->set_name(parent_function_name);
    fn.mutable_signature()->set_name(function_name);
    return test::function::GDef(
        {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
         NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
         NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
         NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
         num_parallel_calls_node,
         MakeParallelMapV2Node(parent_map_node_name, "range",
                               num_parallel_calls_node.name(),
                               parent_function_name, "default",
                               /*use_unbounded_threadpool=*/false),
         MakeParallelMapV2Node(map_node_name, parent_map_node_name,
                               num_parallel_calls_node.name(), function_name,
                               "default", /*use_unbounded_threadpool=*/false)},
        // FunctionLib
        {parent_fn, fn});
  };

  GrapplerItem item_1;
  item_1.graph = graph("map1", "map2", "fnA", "fnB");
  GraphDef output_1;
  TF_ASSERT_OK(OptimizeWithMapFusion(item_1, &output_1, true));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName(
      "map_fusion_nodes/map1/map2", output_1));
  EXPECT_TRUE(graph_utils::ContainsGraphFunctionWithName(
      "map_fusion_funcs/fnA/fnB", output_1.library()));

  GrapplerItem item_2;
  item_2.graph = graph("map3", "map4", "fnC", "fnD");
  GraphDef output_2;
  TF_ASSERT_OK(OptimizeWithMapFusion(item_2, &output_2, true));
  EXPECT_TRUE(graph_utils::ContainsGraphNodeWithName(
      "map_fusion_nodes/map3/map4", output_2));
  EXPECT_TRUE(graph_utils::ContainsGraphFunctionWithName(
      "map_fusion_funcs/fnC/fnD", output_2.library()));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
