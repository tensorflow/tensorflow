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

#include "tensorflow/core/grappler/optimizers/data/map_vectorization.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using test::function::NDef;

NodeDef* AddConstBoolNode(GraphDef* graph, StringPiece name, bool value) {
  NodeDef* node = graph->add_node();
  *node = NDef(/*name=*/name, /*op=*/"Const", /*inputs=*/{},
               /*attrs=*/{{"value", value}, {"dtype", DT_BOOL}});

  return node;
}

NodeDef* AddConstInt32Node(GraphDef* graph, StringPiece name, int value) {
  NodeDef* node = graph->add_node();
  *node = NDef(/*name=*/name, /*op=*/"Const", /*inputs=*/{},
               /*attrs=*/{{"value", value}, {"dtype", DT_INT32}});

  return node;
}

NodeDef* AddConstInt64Node(GraphDef* graph, StringPiece name, int64 value) {
  NodeDef* node = graph->add_node();
  *node = NDef(/*name=*/name, /*op=*/"Const", /*inputs=*/{},
               /*attrs=*/{{"value", value}, {"dtype", DT_INT64}});

  return node;
}

// Adds a simple vectorizable map function that is akin to
// dataset.map(lambda x: tf.identity(x))
FunctionDef* AddMapFn(GraphDef* graph) {
  FunctionDef* map_fn = graph->mutable_library()->add_function();
  *map_fn = FunctionDefHelper::Create(
      /*function_name=*/"map_fn",
      /*in_def=*/{"x: int64"},
      /*out_def=*/{"res: int64"},
      /*attr_def=*/{},
      /*node_def=*/{{{"node"}, "Identity", {"x"}, {{"T", DT_INT64}}}},
      /*ret_def=*/{{"res", "node:output"}});

  return map_fn;
}

NodeDef* AddMapNode(GraphDef* graph, const string& input_dataset,
                    bool parallel = false) {
  NodeDef* map_node = graph->add_node();
  auto map_fn = AddMapFn(graph);

  if (parallel) {
    auto num_parallel_calls = AddConstInt32Node(graph, "num_parallel_calls", 8);
    *map_node =
        NDef(/*name=*/"map", /*op=*/"ParallelMapDataset",
             /*inputs=*/{input_dataset, num_parallel_calls->name()},
             /*attrs=*/
             {{"f", FunctionDefHelper::FunctionRef(map_fn->signature().name())},
              {"Targuments", gtl::ArraySlice<DataType>({})},
              {"output_types", gtl::ArraySlice<DataType>({DT_INT64})},
              {"output_shapes", gtl::ArraySlice<TensorShape>({{}})},
              {"use_inter_op_parallelism", false},
              {"sloppy", true},
              {"preserve_cardinality", true}});
  } else {
    *map_node =
        NDef(/*name=*/"map", /*op=*/"MapDataset",
             /*inputs=*/{input_dataset},
             /*attrs=*/
             {{"f", FunctionDefHelper::FunctionRef(map_fn->signature().name())},
              {"Targuments", gtl::ArraySlice<DataType>({})},
              {"output_types", gtl::ArraySlice<DataType>({DT_INT64})},
              {"output_shapes", gtl::ArraySlice<TensorShape>({{}})},
              {"use_inter_op_parallelism", false},
              {"preserve_cardinality", true}});
  }

  return map_node;
}

NodeDef* AddBatchNode(GraphDef* graph, const string& input_dataset,
                      bool v2 = false) {
  NodeDef* batch_node = graph->add_node();

  auto batch_size = AddConstInt64Node(graph, "batch_size", 10);

  if (v2) {
    auto drop_remainder = AddConstBoolNode(graph, "drop_remainder", true);
    *batch_node = NDef(
        /*name=*/"batch", /*op=*/"BatchDatasetV2",
        /*inputs=*/{input_dataset, batch_size->name(), drop_remainder->name()},
        /*attrs=*/
        {{"output_types", gtl::ArraySlice<DataType>({DT_INT64})},
         {"output_shapes", gtl::ArraySlice<TensorShape>({{10, 1}})}});
  } else {
    *batch_node = NDef(
        /*name=*/"batch", /*op=*/"BatchDataset",
        /*inputs=*/{input_dataset, batch_size->name()},
        /*attrs=*/
        {{"output_types", gtl::ArraySlice<DataType>({DT_INT64})},
         {"output_shapes", gtl::ArraySlice<PartialTensorShape>({{-1, 1}})}});
  }
  return batch_node;
}

NodeDef* AddRangeNode(GraphDef* graph) {
  auto start = AddConstInt64Node(graph, "start", 0);
  auto stop = AddConstInt64Node(graph, "stop", 10);
  auto step = AddConstInt64Node(graph, "step", 1);
  NodeDef* range_node = graph->add_node();
  *range_node = NDef(/*name=*/"range", /*op=*/"RangeDataset",
                     /*inputs=*/{start->name(), stop->name(), step->name()},
                     /*attrs=*/
                     {{"output_shapes", gtl::ArraySlice<TensorShape>({{}})},
                      {"output_types", gtl::ArraySlice<DataType>({DT_INT64})}});

  return range_node;
}

void CheckNotVectorized(const GraphDef& output, const string& map_op,
                        const string& batch_op, const string& map_input_name) {
  ASSERT_EQ(graph_utils::FindAllGraphNodesWithOp(map_op, output).size(), 1);
  ASSERT_EQ(graph_utils::FindAllGraphNodesWithOp(batch_op, output).size(), 1);
  const NodeDef& map_node =
      output.node(graph_utils::FindGraphNodeWithOp(map_op, output));
  const NodeDef& batch_node =
      output.node(graph_utils::FindGraphNodeWithOp(batch_op, output));
  EXPECT_EQ(map_node.input(0), map_input_name);
  EXPECT_EQ(batch_node.input(0), map_node.name());
}

void CheckVectorized(const GraphDef& output, const string& map_op,
                     const string& batch_op, const string& map_input_name) {
  ASSERT_EQ(graph_utils::FindAllGraphNodesWithOp(map_op, output).size(), 1);
  ASSERT_EQ(graph_utils::FindAllGraphNodesWithOp(batch_op, output).size(), 1);
  const NodeDef& map_node =
      output.node(graph_utils::FindGraphNodeWithOp(map_op, output));
  const NodeDef& batch_node =
      output.node(graph_utils::FindGraphNodeWithOp(batch_op, output));
  EXPECT_EQ(map_node.input(0), batch_node.name());
  EXPECT_EQ(batch_node.input(0), map_input_name);

  // Check that the function is actually vectorized.
  // The vectorization of the identity function is itself.
  string function_name = map_node.attr().at("f").func().name();
  int found =
      graph_utils::FindGraphFunctionWithName(function_name, output.library());
  ASSERT_NE(found, -1);
  const auto& function = output.library().function(found);
  EXPECT_EQ(function.node_def(0).op(), "Identity");
}

class MapThenBatchTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(MapThenBatchTest, IsVectorized) {
  bool use_parallel_map = std::get<0>(GetParam());
  bool use_batch_v2 = std::get<1>(GetParam());
  GrapplerItem item;
  auto range_node = AddRangeNode(&item.graph);
  auto map_node = AddMapNode(&item.graph, range_node->name(), use_parallel_map);
  auto batch_node = AddBatchNode(&item.graph, map_node->name(), use_batch_v2);
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, map_node->op(), batch_node->op(), range_node->name());
}

INSTANTIATE_TEST_SUITE_P(MapThenBatchTest, MapThenBatchTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

// Not all dataset types have "output_shapes" and "output_types" attrs defined.
// Add a generic input node which may not have these attrs defined.
NodeDef* AddArbitraryInputNode(GraphDef* graph,
                               std::vector<PartialTensorShape>* output_shapes,
                               std::vector<DataType>* output_types) {
  auto input_node = graph->add_node();
  std::vector<std::pair<string, FunctionDefHelper::AttrValueWrapper>> attrs;
  if (output_shapes) {
    attrs.push_back({"output_shapes", *output_shapes});
  }
  if (output_types) {
    attrs.push_back({"output_types", *output_types});
  }

  *input_node = NDef(/*name=*/"input", /*op=*/"InputDataset", /*inputs=*/{},
                     /*attrs=*/attrs);

  return input_node;
}

TEST(MapVectorizationTest, VectorizeWithUndefinedOutputShapes) {
  // Tests that the optimization doesn't break when the input to MapDataset
  // doesn't have an output_shapes attr defined. In this case, the map and
  // batch swap does not occur.
  GrapplerItem item;
  std::vector<DataType> input_types({DT_INT64});
  auto input_node = AddArbitraryInputNode(&item.graph, nullptr, &input_types);
  auto map_node = AddMapNode(&item.graph, input_node->name());
  auto batch_node = AddBatchNode(&item.graph, map_node->name());
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckNotVectorized(output, map_node->op(), batch_node->op(),
                     input_node->name());
}

TEST(MapVectorizationTest, VectorizeWithUnknownRank) {
  // Tests that the optimization doesn't break when the input to MapDataset
  // has components with unknown rank. In this case, the optimization does not
  // occur.
  GrapplerItem item;
  std::vector<PartialTensorShape> input_shapes({{}});
  std::vector<DataType> input_types({DT_INT64});
  auto input_node =
      AddArbitraryInputNode(&item.graph, &input_shapes, &input_types);
  auto map_node = AddMapNode(&item.graph, input_node->name());
  auto batch_node = AddBatchNode(&item.graph, map_node->name());
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckNotVectorized(output, map_node->op(), batch_node->op(),
                     input_node->name());
}

TEST(MapVectorizationTest, VectorizeWithUnknownDim) {
  // Tests that the optimization doesn't break when the input to MapDataset
  // has components with unknown dimensions. In this case, the optimization does
  // not occur.
  GrapplerItem item;
  std::vector<PartialTensorShape> input_shapes({{-1, 2}});
  std::vector<DataType> input_types({DT_INT64});
  auto input_node =
      AddArbitraryInputNode(&item.graph, &input_shapes, &input_types);
  auto map_node = AddMapNode(&item.graph, input_node->name());
  auto batch_node = AddBatchNode(&item.graph, map_node->name());
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckNotVectorized(output, map_node->op(), batch_node->op(),
                     input_node->name());
}

TEST(MapVectorizationTest, VectorizeWithUndefinedOutputTypes) {
  // Tests that the optimization doesn't break when the input doesn't have
  // an output_types attr defined. The output_types of the input node, even
  // if not present, can be inferred from the map function input signature.
  GrapplerItem item;
  std::vector<PartialTensorShape> input_shapes({{1}});
  auto input_node = AddArbitraryInputNode(&item.graph, &input_shapes, nullptr);
  auto map_node = AddMapNode(&item.graph, input_node->name());
  auto batch_node = AddBatchNode(&item.graph, map_node->name());
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, map_node->op(), batch_node->op(), input_node->name());
}

// TODO(rachelim): Add test that has a polymorphic function.

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
