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

#include "tensorflow/core/grappler/optimizers/data/disable_intra_op_parallelism.h"

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

// If the user manually sets intra op parallelism, we don't insert the op.
class IntraOpAlreadySetTest
    : public ::testing::TestWithParam<std::tuple<string, int64>> {};

TEST_P(IntraOpAlreadySetTest, IntraOpParallelism) {
  const string op = std::get<0>(GetParam());
  const int64 value = std::get<1>(GetParam());

  GrapplerItem item;
  MutableGraphView graph(&item.graph);

  NodeDef *start_val = graph_utils::AddScalarConstNode<int64>(0, &graph);
  NodeDef *stop_val = graph_utils::AddScalarConstNode<int64>(10, &graph);
  NodeDef *step_val = graph_utils::AddScalarConstNode<int64>(1, &graph);
  std::vector<string> range_inputs(3);
  range_inputs[0] = start_val->name();
  range_inputs[1] = stop_val->name();
  range_inputs[2] = step_val->name();
  std::vector<std::pair<string, AttrValue>> range_attrs;
  NodeDef *range_node = graph_utils::AddNode("", "RangeDataset", range_inputs,
                                             range_attrs, &graph);

  NodeDef *max_parallelism_val =
      graph_utils::AddScalarConstNode<int64>(value, &graph);
  std::vector<string> parallelism_inputs(2);
  parallelism_inputs[0] = range_node->name();
  parallelism_inputs[1] = max_parallelism_val->name();
  std::vector<std::pair<string, AttrValue>> parallelism_attrs;
  graph_utils::AddNode("", op, parallelism_inputs, parallelism_attrs, &graph);

  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(op, item.graph));
  EXPECT_EQ(item.graph.node_size(), 6);
  EXPECT_EQ(max_parallelism_val->attr().at("value").tensor().int64_val(0),
            value);

  DisableIntraOpParallelism optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_EQ(output.node_size(), 6);
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(op, output));
  NodeDef test_node = output.node(graph_utils::FindGraphNodeWithOp(op, output));
  NodeDef test_val = output.node(
      graph_utils::FindGraphNodeWithName(test_node.input(1), output));
  EXPECT_EQ(test_val.attr().at("value").tensor().int64_val(0), value);
}

INSTANTIATE_TEST_SUITE_P(
    Test, IntraOpAlreadySetTest,
    ::testing::Combine(
        ::testing::Values("MaxIntraOpParallelismDataset",
                          "ExperimentalMaxIntraOpParallelismDataset"),
        ::testing::Values(1, 5)));

// If the user hasn't set intra op parallelism, we insert the op to disable it.
TEST(IntraOpNotSetTest, IntraOpParallelism) {
  GrapplerItem item;

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("Sink", "Identity", {"range"}, {})});
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset",
                                               item.graph));
  EXPECT_EQ(item.graph.node_size(), 5);

  DisableIntraOpParallelism optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_EQ(output.node_size(), 7);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset", output));
  NodeDef test_node = output.node(
      graph_utils::FindGraphNodeWithOp("MaxIntraOpParallelismDataset", output));
  NodeDef test_val = output.node(
      graph_utils::FindGraphNodeWithName(test_node.input(1), output));
  EXPECT_EQ(test_val.attr().at("value").tensor().int64_val(0), 1);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
