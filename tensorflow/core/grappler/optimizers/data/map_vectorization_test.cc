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
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kConstOp[] = "Const";
constexpr char kRangeOp[] = "RangeDataset";
constexpr char kBatchOp[] = "BatchDataset";
constexpr char kBatchV2Op[] = "BatchDatasetV2";
constexpr char kExperimentalMapAndBatchOp[] = "ExperimentalMapAndBatchDataset";
constexpr char kMapOp[] = "MapDataset";
constexpr char kParallelMapOp[] = "ParallelMapDataset";
constexpr char kAttrNameF[] = "f";
constexpr char kAttrNameTarguments[] = "Targuments";
constexpr char kAttrNameOutputTypes[] = "output_types";
constexpr char kAttrNameOutputShapes[] = "output_shapes";
constexpr char kAttrNameInterOpParallelism[] = "use_inter_op_parallelism";
constexpr char kAttrNamePreserveCardinality[] = "preserve_cardinality";
constexpr char kAttrNameSloppy[] = "sloppy";
constexpr char kAttrNameValue[] = "value";
constexpr char kAttrNameDtype[] = "dtype";

using test::function::NDef;

NodeDef* AddConstBoolNode(GraphDef* graph, StringPiece name, bool value) {
  NodeDef* node = graph->add_node();
  *node = NDef(/*name=*/name, /*op=*/kConstOp, /*inputs=*/{},
               /*attrs=*/{{kAttrNameValue, value}, {kAttrNameDtype, DT_BOOL}});

  return node;
}

NodeDef* AddConstInt32Node(GraphDef* graph, StringPiece name, int value) {
  NodeDef* node = graph->add_node();
  *node = NDef(/*name=*/name, /*op=*/kConstOp, /*inputs=*/{},
               /*attrs=*/{{kAttrNameValue, value}, {kAttrNameDtype, DT_INT32}});

  return node;
}

NodeDef* AddConstInt64Node(GraphDef* graph, StringPiece name, int64 value) {
  NodeDef* node = graph->add_node();
  *node = NDef(/*name=*/name, /*op=*/kConstOp, /*inputs=*/{},
               /*attrs=*/{{kAttrNameValue, value}, {kAttrNameDtype, DT_INT64}});

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
                    const string& map_fn, bool parallel = false,
                    int suffix = 0) {
  NodeDef* map_node = graph->add_node();

  if (parallel) {
    auto num_parallel_calls = AddConstInt32Node(
        graph, strings::StrCat("num_parallel_calls", suffix), 8);
    *map_node = NDef(
        /*name=*/strings::StrCat("map", suffix), /*op=*/kParallelMapOp,
        /*inputs=*/{input_dataset, num_parallel_calls->name()},
        /*attrs=*/
        {{kAttrNameF, FunctionDefHelper::FunctionRef(map_fn)},
         {kAttrNameTarguments, gtl::ArraySlice<DataType>({})},
         {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
         {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})},
         {kAttrNameInterOpParallelism, false},
         {kAttrNameSloppy, true},
         {kAttrNamePreserveCardinality, true}});
  } else {
    *map_node =
        NDef(/*name=*/strings::StrCat("map", suffix), /*op=*/kMapOp,
             /*inputs=*/{input_dataset},
             /*attrs=*/
             {{kAttrNameF, FunctionDefHelper::FunctionRef(map_fn)},
              {kAttrNameTarguments, gtl::ArraySlice<DataType>({})},
              {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
              {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})},
              {kAttrNameInterOpParallelism, false},
              {kAttrNamePreserveCardinality, true}});
  }

  return map_node;
}

NodeDef* AddBatchNode(GraphDef* graph, const string& input_dataset,
                      bool v2 = false, int suffix = 0) {
  NodeDef* batch_node = graph->add_node();

  auto batch_size =
      AddConstInt64Node(graph, strings::StrCat("batch_size", suffix), 10);

  if (v2) {
    auto drop_remainder = AddConstBoolNode(
        graph, strings::StrCat("drop_remainder", suffix), true);
    *batch_node = NDef(
        /*name=*/strings::StrCat("batch", suffix), /*op=*/kBatchV2Op,
        /*inputs=*/{input_dataset, batch_size->name(), drop_remainder->name()},
        /*attrs=*/
        {{kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
         {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{10, 1}})}});
  } else {
    *batch_node = NDef(
        /*name=*/strings::StrCat("batch", suffix), /*op=*/kBatchOp,
        /*inputs=*/{input_dataset, batch_size->name()},
        /*attrs=*/
        {{kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
         {kAttrNameOutputShapes,
          gtl::ArraySlice<PartialTensorShape>({{v2 ? 10 : -1, 1}})}});
  }
  return batch_node;
}

NodeDef* AddRangeNode(GraphDef* graph) {
  auto start = AddConstInt64Node(graph, "start", 0);
  auto stop = AddConstInt64Node(graph, "stop", 10);
  auto step = AddConstInt64Node(graph, "step", 1);
  NodeDef* range_node = graph->add_node();
  *range_node =
      NDef(/*name=*/"range", /*op=*/kRangeOp,
           /*inputs=*/{start->name(), stop->name(), step->name()},
           /*attrs=*/
           {{kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})},
            {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})}});

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
  string function_name = map_node.attr().at(kAttrNameF).func().name();
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
  auto map_fn = AddMapFn(&item.graph);
  auto map_node = AddMapNode(&item.graph, range_node->name(),
                             map_fn->signature().name(), use_parallel_map);
  auto batch_node = AddBatchNode(&item.graph, map_node->name(), use_batch_v2);
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, map_node->op(), batch_node->op(), range_node->name());
}

INSTANTIATE_TEST_SUITE_P(MapThenBatchTest, MapThenBatchTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

NodeDef* AddMapAndBatchNode(GraphDef* graph, const string& input_dataset,
                            const string& map_fn, int suffix = 0) {
  auto batch_size =
      AddConstInt64Node(graph, strings::StrCat("batch_size", suffix), 10);
  auto num_parallel_calls = AddConstInt64Node(
      graph, strings::StrCat("num_parallel_calls", suffix), 8);
  auto drop_remainder =
      AddConstBoolNode(graph, strings::StrCat("drop_remainder", suffix), true);

  auto map_and_batch_node = graph->add_node();
  *map_and_batch_node =
      NDef(/*name=*/strings::StrCat("map_and_batch", suffix),
           /*op=*/kExperimentalMapAndBatchOp,
           /*inputs=*/
           {input_dataset, batch_size->name(), num_parallel_calls->name(),
            drop_remainder->name()},
           /*attrs=*/
           {{kAttrNameF, FunctionDefHelper::FunctionRef(map_fn)},
            {kAttrNameTarguments, gtl::ArraySlice<DataType>({})},
            {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
            {kAttrNameOutputShapes,
             gtl::ArraySlice<PartialTensorShape>({{10, 1}})}});

  return map_and_batch_node;
}

TEST(MapVectorizationTest, VectorizeExperimentalMapAndBatch) {
  GrapplerItem item;
  auto range_node = AddRangeNode(&item.graph);
  auto map_fn = AddMapFn(&item.graph);
  auto map_and_batch_node = AddMapAndBatchNode(&item.graph, range_node->name(),
                                               map_fn->signature().name());
  ASSERT_NE(map_and_batch_node, nullptr);

  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, kParallelMapOp, kBatchV2Op, "range");
}

class ChainedMapAndBatchTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

// Tests:
// 1) map.batch.map.batch
// 2) map.batch.map_and_batch
// 3) map_and_batch.map.batch
// 4) map_and_batch.map_and_batch
TEST_P(ChainedMapAndBatchTest, IsVectorized) {
  GrapplerItem item;
  auto input_node = AddRangeNode(&item.graph);

  auto map_fn = AddMapFn(&item.graph);

  auto make_map_and_batch = [&item, map_fn](NodeDef* input, bool fuse,
                                            int suffix) {
    if (fuse) {
      return AddMapAndBatchNode(&item.graph, input->name(),
                                map_fn->signature().name(), suffix);
    }
    auto map_node = AddMapNode(&item.graph, input->name(),
                               map_fn->signature().name(), true, suffix);
    auto batch_node = AddBatchNode(&item.graph, map_node->name(), true, suffix);
    return batch_node;
  };

  auto map_and_batch_0 =
      make_map_and_batch(input_node, std::get<0>(GetParam()), 0);
  auto map_and_batch_1 =
      make_map_and_batch(map_and_batch_0, std::get<1>(GetParam()), 1);
  ASSERT_NE(map_and_batch_1, nullptr);

  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  TF_ASSERT_OK(TopologicalSort(&output));
  std::vector<int> map_nodes =
      graph_utils::FindAllGraphNodesWithOp(kParallelMapOp, output);
  std::vector<int> batch_nodes =
      graph_utils::FindAllGraphNodesWithOp(kBatchV2Op, output);
  ASSERT_EQ(map_nodes.size(), 2);
  ASSERT_EQ(batch_nodes.size(), 2);
  const NodeDef& range_node =
      output.node(graph_utils::FindGraphNodeWithOp(kRangeOp, output));

  const NodeDef& batch_node_0 = output.node(batch_nodes[0]);
  EXPECT_EQ(batch_node_0.input(0), range_node.name());
  const NodeDef& map_node_0 = output.node(map_nodes[0]);
  EXPECT_EQ(map_node_0.input(0), batch_node_0.name());
  const NodeDef& batch_node_1 = output.node(batch_nodes[1]);
  EXPECT_EQ(batch_node_1.input(0), map_node_0.name());
  const NodeDef& map_node_1 = output.node(map_nodes[1]);
  EXPECT_EQ(map_node_1.input(0), batch_node_1.name());
}

INSTANTIATE_TEST_SUITE_P(ChainedMapAndBatchTest, ChainedMapAndBatchTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

// Not all dataset types have kAttrNameOutputShapes and kAttrNameOutputTypes
// attrs defined. Add a generic input node which may not have these attrs
// defined.
NodeDef* AddArbitraryInputNode(GraphDef* graph,
                               std::vector<PartialTensorShape>* output_shapes,
                               std::vector<DataType>* output_types) {
  auto input_node = graph->add_node();
  std::vector<std::pair<string, FunctionDefHelper::AttrValueWrapper>> attrs;
  if (output_shapes) {
    attrs.push_back({kAttrNameOutputShapes, *output_shapes});
  }
  if (output_types) {
    attrs.push_back({kAttrNameOutputTypes, *output_types});
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
  auto map_fn = AddMapFn(&item.graph);
  auto map_node =
      AddMapNode(&item.graph, input_node->name(), map_fn->signature().name());
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
  auto map_fn = AddMapFn(&item.graph);
  auto map_node =
      AddMapNode(&item.graph, input_node->name(), map_fn->signature().name());
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
  auto map_fn = AddMapFn(&item.graph);
  auto map_node =
      AddMapNode(&item.graph, input_node->name(), map_fn->signature().name());
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
  auto map_fn = AddMapFn(&item.graph);
  auto map_node =
      AddMapNode(&item.graph, input_node->name(), map_fn->signature().name());
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
