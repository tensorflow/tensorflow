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

#include "tensorflow/core/grappler/optimizers/data/use_private_thread_pool.h"

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

// If the user manually sets private thread pool, we don't insert the op.
class ThreadPoolOpAlreadySetTest : public ::testing::TestWithParam<int64> {};

TEST_P(ThreadPoolOpAlreadySetTest, PrivateThreadPool) {
  const int64 num_of_threads = GetParam();

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
  NodeDef *num_of_threads_val =
      graph_utils::AddScalarConstNode<int64>(num_of_threads, &graph);
  std::vector<string> private_threads_inputs(2);
  private_threads_inputs[0] = range_node->name();
  private_threads_inputs[1] = num_of_threads_val->name();
  std::vector<std::pair<string, AttrValue>> private_threads_attrs;
  NodeDef *private_threads_node = graph_utils::AddNode(
      "private_thread_pool", "PrivateThreadPoolDataset", private_threads_inputs,
      private_threads_attrs, &graph);
  std::vector<string> sink_inputs(1);
  sink_inputs[0] = private_threads_node->name();
  std::vector<std::pair<string, AttrValue>> sink_attrs;
  NodeDef *sink_node =
      graph_utils::AddNode("Sink", "Identity", sink_inputs, sink_attrs, &graph);
  item.fetch.push_back(sink_node->name());

  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", item.graph));
  EXPECT_EQ(item.graph.node_size(), 7);
  EXPECT_EQ(num_of_threads_val->attr().at("value").tensor().int64_val(0),
            num_of_threads);

  UsePrivateThreadPool optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_EQ(output.node_size(), 7);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", output));
  NodeDef new_private_threads_node = output.node(
      graph_utils::FindGraphNodeWithOp("PrivateThreadPoolDataset", output));
  NodeDef new_num_of_threads_val =
      output.node(graph_utils::FindGraphNodeWithName(
          new_private_threads_node.input(1), output));
  EXPECT_EQ(new_num_of_threads_val.attr().at("value").tensor().int64_val(0),
            num_of_threads);
}

INSTANTIATE_TEST_SUITE_P(Test, ThreadPoolOpAlreadySetTest,
                         ::testing::Values(1, 2, 4));

// Test the case if the user hasn't set private thread pool.
//
// If we can not find the sink node or sink node op is "_Retval", we don't apply
// the optimization; otherwise, we insert the op to use private thread pool.
class ThreadPoolOpNotSetTest : public ::testing::TestWithParam<string> {};

TEST_P(ThreadPoolOpNotSetTest, PrivateThreadPool) {
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
  EXPECT_FALSE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", item.graph));
  EXPECT_EQ(item.graph.node_size(), 5);
  item.fetch.push_back("Sink_fake");

  UsePrivateThreadPool optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  EXPECT_FALSE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", output));
  EXPECT_EQ(output.node_size(), 5);

  item.fetch[0] = "Sink";
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  if (op == "_Retval") {
    EXPECT_FALSE(
        graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", output));
    EXPECT_EQ(output.node_size(), 5);
    return;
  }

  EXPECT_EQ(output.node_size(), 7);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", output));
  NodeDef sink_node =
      output.node(graph_utils::FindGraphNodeWithName("Sink", output));
  EXPECT_EQ(sink_node.input_size(), 1);
  NodeDef private_threads_node = output.node(
      graph_utils::FindGraphNodeWithName(sink_node.input(0), output));
  EXPECT_EQ(private_threads_node.op(), "PrivateThreadPoolDataset");
  EXPECT_EQ(private_threads_node.input_size(), 2);
  NodeDef range_node = output.node(graph_utils::FindGraphNodeWithName(
      private_threads_node.input(0), output));
  EXPECT_EQ(range_node.name(), "range");
  NodeDef num_of_threads_val = output.node(graph_utils::FindGraphNodeWithName(
      private_threads_node.input(1), output));
  EXPECT_EQ(num_of_threads_val.attr().at("value").tensor().int64_val(0), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, ThreadPoolOpNotSetTest,
                         ::testing::Values("Identity", "_Retval"));

// Test the autotune case with ModelDataset in the pipeline. We will insert
// PrivateThreadPoolDataset before ModelDataset.
TEST(AutotuneWithModelTest, PrivateThreadPool) {
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
  EXPECT_FALSE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", item.graph));
  EXPECT_EQ(item.graph.node_size(), 6);
  item.fetch.push_back("Sink");

  UsePrivateThreadPool optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));

  EXPECT_EQ(output.node_size(), 8);
  EXPECT_TRUE(
      graph_utils::ContainsNodeWithOp("PrivateThreadPoolDataset", output));
  NodeDef sink_node =
      output.node(graph_utils::FindGraphNodeWithName("Sink", output));
  EXPECT_EQ(sink_node.input_size(), 1);
  NodeDef model_node = output.node(
      graph_utils::FindGraphNodeWithName(sink_node.input(0), output));
  EXPECT_EQ(model_node.op(), "ModelDataset");
  EXPECT_EQ(model_node.input_size(), 1);
  NodeDef private_threads_node = output.node(
      graph_utils::FindGraphNodeWithName(model_node.input(0), output));
  EXPECT_EQ(private_threads_node.op(), "PrivateThreadPoolDataset");
  EXPECT_EQ(private_threads_node.input_size(), 2);
  NodeDef range_node = output.node(graph_utils::FindGraphNodeWithName(
      private_threads_node.input(0), output));
  EXPECT_EQ(range_node.name(), "range");
  NodeDef num_of_threads_val = output.node(graph_utils::FindGraphNodeWithName(
      private_threads_node.input(1), output));
  EXPECT_EQ(num_of_threads_val.attr().at("value").tensor().int64_val(0), 0);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
