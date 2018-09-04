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

#include "tensorflow/core/grappler/optimizers/data/filter_fusion.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"

#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

NodeDef MakeFilterNode(StringPiece name, StringPiece input_node_name) {
  return test::function::NDef(
      name, "FilterDataset", {string(input_node_name)},
      {{"predicate", FunctionDefHelper::FunctionRef("IsZero")},
       {"Targuments", {}},
       {"output_shapes", {}},
       {"output_types", {}}});
}

TEST(FilterFusionTest, FuseTwoFilterIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter1", "range"),
       MakeFilterNode("filter2", "filter1")},
      // FunctionLib
      {
          test::function::IsZero(),
      });

  FilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("FilterDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter2", output));
}

TEST(FilterFusionTest, FuseThreeNodesIntoOne) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("filename", "Const", {}, {{"value", ""}, {"dtype", DT_STRING}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       MakeFilterNode("filter1", "range"), MakeFilterNode("filter2", "filter1"),
       MakeFilterNode("filter3", "filter2"),
       NDef("cache", "CacheDataset", {"filter3", "filename"}, {})},
      // FunctionLib
      {
          test::function::IsZero(),
      });

  FilterFusion optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp("FilterDataset", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter1", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter2", output));
  EXPECT_FALSE(graph_utils::ContainsGraphNodeWithName("filter3", output));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
