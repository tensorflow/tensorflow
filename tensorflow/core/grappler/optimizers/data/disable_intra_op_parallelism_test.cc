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
  NodeDef *range_node = graph_utils::AddNode("range", "RangeDataset",
                                             range_inputs, range_attrs, &graph);

  NodeDef *parallelism_val =
      graph_utils::AddScalarConstNode<int64>(value, &graph);
  std::vector<string> parallelism_inputs(2);
  parallelism_inputs[0] = range_node->name();
  parallelism_inputs[1] = parallelism_val->name();
  std::vector<std::pair<string, AttrValue>> parallelism_attrs;
  NodeDef *parallelism_node = graph_utils::AddNode(
      "max_parallelism", op, parallelism_inputs, parallelism_attrs, &graph);

  std::vector<string> sink_inputs(1);
  sink_inputs[0] = parallelism_node->name();
  std::vector<std::pair<string, AttrValue>> sink_attrs;
  NodeDef *sink_node =
      graph_utils::AddNode("Sink", "Identity", sink_inputs, sink_attrs, &graph);
  item.fetch.push_back(sink_node->name());

  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(op, item.graph));
  EXPECT_EQ(item.graph.node_size(), 7);
  EXPECT_EQ(parallelism_val->attr().at("value").tensor().int64_val(0), value);

  DisableIntraOpParallelism optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_EQ(output.node_size(), 7);
  EXPECT_TRUE(graph_utils::ContainsNodeWithOp(op, output));
  NodeDef new_parallelism_node =
      output.node(graph_utils::FindGraphNodeWithOp(op, output));
  NodeDef new_parallelism_val = output.node(graph_utils::FindGraphNodeWithName(
      new_parallelism_node.input(1), output));
  EXPECT_EQ(new_parallelism_val.attr().at("value").tensor().int64_val(0),
            value);
}

INSTANTIATE_TEST_SUITE_P(
    Test, IntraOpAlreadySetTest,
    ::testing::Combine(
        ::testing::Values("MaxIntraOpParallelismDataset",
                          "ExperimentalMaxIntraOpParallelismDataset"),
        ::testing::Values(1, 5)));

// Test the case if the user hasn't set intra op parallelism.
//
// If we can not find the sink node or sink node op is "_Retval", we don't apply
// the optimization; otherwise, we insert the op to disable intra op
// parallelism.
class IntraOpNotSetTest : public ::testing::TestWithParam<string> {};

TEST_P(IntraOpNotSetTest, IntraOpParallelism) {
  const string op = GetParam();
  GrapplerItem item;

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("Sink", op, {"range"}, {})});
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset",
                                               item.graph));
  EXPECT_EQ(item.graph.node_size(), 5);
  item.fetch.push_back("Sink_fake");

  DisableIntraOpParallelism optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_FALSE(
      graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset", output));
  EXPECT_EQ(output.node_size(), 5);

  item.fetch[0] = "Sink";
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  if (op == "_Retval") {
    EXPECT_FALSE(graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset",
                                                 output));
    EXPECT_EQ(output.node_size(), 5);
    return;
  }

  EXPECT_EQ(output.node_size(), 7);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset", output));
  NodeDef sink_node =
      output.node(graph_utils::FindGraphNodeWithName("Sink", output));
  EXPECT_EQ(sink_node.input_size(), 1);
  NodeDef parallelism_node = output.node(
      graph_utils::FindGraphNodeWithName(sink_node.input(0), output));
  EXPECT_EQ(parallelism_node.op(), "MaxIntraOpParallelismDataset");
  EXPECT_EQ(parallelism_node.input_size(), 2);
  NodeDef range_node = output.node(
      graph_utils::FindGraphNodeWithName(parallelism_node.input(0), output));
  EXPECT_EQ(range_node.name(), "range");
  NodeDef parallelism_val = output.node(
      graph_utils::FindGraphNodeWithName(parallelism_node.input(1), output));
  EXPECT_EQ(parallelism_val.attr().at("value").tensor().int64_val(0), 1);
}

INSTANTIATE_TEST_SUITE_P(Test, IntraOpNotSetTest,
                         ::testing::Values("Identity", "_Retval"));

// Test the autotune case with ModelDataset in the pipeline. We will insert
// MaxIntraOpParallelismDataset before ModelDataset.
TEST(AutotuneWithModelTest, IntraOpParallelism) {
  GrapplerItem item;

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{}},
             {"output_types", gtl::ArraySlice<DataType>{}}}),
       NDef("model", "ModelDataset", {"range"}, {}),
       NDef("Sink", "Identity", {"model"}, {})});
  EXPECT_FALSE(graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset",
                                               item.graph));
  EXPECT_EQ(item.graph.node_size(), 6);
  item.fetch.push_back("Sink");

  DisableIntraOpParallelism optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(output.node_size(), 8);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("MaxIntraOpParallelismDataset", output));
  NodeDef sink_node =
      output.node(graph_utils::FindGraphNodeWithName("Sink", output));
  EXPECT_EQ(sink_node.input_size(), 1);
  NodeDef model_node = output.node(
      graph_utils::FindGraphNodeWithName(sink_node.input(0), output));
  EXPECT_EQ(model_node.op(), "ModelDataset");
  EXPECT_EQ(model_node.input_size(), 1);
  NodeDef parallelism_node = output.node(
      graph_utils::FindGraphNodeWithName(model_node.input(0), output));
  EXPECT_EQ(parallelism_node.op(), "MaxIntraOpParallelismDataset");
  EXPECT_EQ(parallelism_node.input_size(), 2);
  NodeDef range_node = output.node(
      graph_utils::FindGraphNodeWithName(parallelism_node.input(0), output));
  EXPECT_EQ(range_node.name(), "range");
  NodeDef parallelism_val = output.node(
      graph_utils::FindGraphNodeWithName(parallelism_node.input(1), output));
  EXPECT_EQ(parallelism_val.attr().at("value").tensor().int64_val(0), 1);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
