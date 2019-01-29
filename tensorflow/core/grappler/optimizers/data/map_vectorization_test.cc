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
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/grappler/utils/topological_sort.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"

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

// Adds a simple vectorizable map function that is akin to
// dataset.map(lambda x: tf.identity(x))
FunctionDef* AddMapFn(MutableGraphView* graph) {
  FunctionDef* map_fn = graph->graph()->mutable_library()->add_function();
  *map_fn = FunctionDefHelper::Create(
      /*function_name=*/"map_fn",
      /*in_def=*/{"x: int64"},
      /*out_def=*/{"res: int64"},
      /*attr_def=*/{},
      /*node_def=*/{{{"node"}, "Identity", {"x"}, {{"T", DT_INT64}}}},
      /*ret_def=*/{{"res", "node:output"}});

  return map_fn;
}

NodeDef* AddMapNode(MutableGraphView* graph, const string& input_dataset,
                    const string& map_fn, int num_parallel_calls = 0) {
  NodeDef result;
  if (num_parallel_calls) {
    auto num_parallel_calls_node =
        graph_utils::AddScalarConstNode(num_parallel_calls, graph);
    result =
        NDef(/*name=*/"map", /*op=*/kParallelMapOp,
             /*inputs=*/{input_dataset, num_parallel_calls_node->name()},
             /*attrs=*/
             {{kAttrNameF, FunctionDefHelper::FunctionRef(map_fn)},
              {kAttrNameTarguments, gtl::ArraySlice<DataType>({})},
              {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
              {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})},
              {kAttrNameInterOpParallelism, false},
              {kAttrNameSloppy, true},
              {kAttrNamePreserveCardinality, true}});
  } else {
    result =
        NDef(/*name=*/"map", /*op=*/kMapOp,
             /*inputs=*/{input_dataset},
             /*attrs=*/
             {{kAttrNameF, FunctionDefHelper::FunctionRef(map_fn)},
              {kAttrNameTarguments, gtl::ArraySlice<DataType>({})},
              {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
              {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})},
              {kAttrNameInterOpParallelism, false},
              {kAttrNamePreserveCardinality, true}});
  }

  graph_utils::SetUniqueGraphNodeName(result.name(), graph->graph(), &result);
  return graph->AddNode(std::move(result));
}

NodeDef* AddBatchNode(MutableGraphView* graph, const string& input_dataset,
                      bool v2 = false, int64 batch_size = 10) {
  NodeDef result;
  auto batch_size_node = graph_utils::AddScalarConstNode(batch_size, graph);

  if (v2) {
    // BatchDatasetV2
    auto drop_remainder = graph_utils::AddScalarConstNode(true, graph);
    result = NDef(
        /*name=*/"batch", /*op=*/kBatchV2Op,
        /*inputs=*/
        {input_dataset, batch_size_node->name(), drop_remainder->name()},
        /*attrs=*/
        {{kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
         {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{10, 1}})}});
  } else {
    result =
        NDef(/*name=*/"batch", /*op=*/kBatchOp,
             /*inputs=*/{input_dataset, batch_size_node->name()},
             /*attrs=*/
             {{kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
              {kAttrNameOutputShapes,
               gtl::ArraySlice<PartialTensorShape>({{v2 ? 10 : -1, 1}})}});
  }

  graph_utils::SetUniqueGraphNodeName(result.name(), graph->graph(), &result);
  return graph->AddNode(std::move(result));
}

NodeDef* AddRangeNode(MutableGraphView* graph) {
  auto start = graph_utils::AddScalarConstNode(static_cast<int64>(0), graph);
  auto stop = graph_utils::AddScalarConstNode(static_cast<int64>(10), graph);
  auto step = graph_utils::AddScalarConstNode(static_cast<int64>(1), graph);

  NodeDef result =
      NDef(/*name=*/"range", /*op=*/kRangeOp,
           /*inputs=*/{start->name(), stop->name(), step->name()},
           /*attrs=*/
           {{kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})},
            {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})}});

  graph_utils::SetUniqueGraphNodeName(result.name(), graph->graph(), &result);
  return graph->AddNode(std::move(result));
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
    : public ::testing::TestWithParam<std::tuple<int, bool>> {};

TEST_P(MapThenBatchTest, IsVectorized) {
  int num_parallel_calls = std::get<0>(GetParam());
  bool use_batch_v2 = std::get<1>(GetParam());
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto range_node = AddRangeNode(&graph);
  auto map_fn = AddMapFn(&graph);
  auto map_node = AddMapNode(&graph, range_node->name(),
                             map_fn->signature().name(), num_parallel_calls);
  auto batch_node = AddBatchNode(&graph, map_node->name(), use_batch_v2);
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, map_node->op(), batch_node->op(), range_node->name());
}

INSTANTIATE_TEST_SUITE_P(MapThenBatchTest, MapThenBatchTest,
                         ::testing::Combine(::testing::Values(0, 12),
                                            ::testing::Bool()));

NodeDef* AddMapAndBatchNode(MutableGraphView* graph,
                            const string& input_dataset, const string& map_fn,
                            int64 batch_size = 10,
                            int64 num_parallel_calls = 12) {
  auto batch_size_node = graph_utils::AddScalarConstNode(batch_size, graph);
  auto num_parallel_calls_node =
      graph_utils::AddScalarConstNode(num_parallel_calls, graph);
  auto drop_remainder = graph_utils::AddScalarConstNode(true, graph);

  NodeDef result =
      NDef(/*name=*/"map_and_batch",
           /*op=*/kExperimentalMapAndBatchOp,
           /*inputs=*/
           {input_dataset, batch_size_node->name(),
            num_parallel_calls_node->name(), drop_remainder->name()},
           /*attrs=*/
           {{kAttrNameF, FunctionDefHelper::FunctionRef(map_fn)},
            {kAttrNameTarguments, gtl::ArraySlice<DataType>({})},
            {kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
            {kAttrNameOutputShapes,
             gtl::ArraySlice<PartialTensorShape>({{10, 1}})}});

  graph_utils::SetUniqueGraphNodeName(result.name(), graph->graph(), &result);
  return graph->AddNode(std::move(result));
}

TEST(MapVectorizationTest, VectorizeExperimentalMapAndBatch) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto range_node = AddRangeNode(&graph);
  auto map_fn = AddMapFn(&graph);
  auto map_and_batch_node = AddMapAndBatchNode(&graph, range_node->name(),
                                               map_fn->signature().name());
  ASSERT_NE(map_and_batch_node, nullptr);

  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, kParallelMapOp, kBatchV2Op, "range");
}

void EvaluateNodes(const GraphDef& graph,
                   const std::vector<string>& output_tensor_names,
                   std::vector<Tensor>* output_tensors) {
  std::unique_ptr<Session> session(NewSession(SessionOptions()));
  TF_CHECK_OK(session->Create(graph));
  TF_CHECK_OK(session->Run({}, output_tensor_names, {}, output_tensors));
}

void CheckNumParallelCalls(const GraphDef& output,
                           int expected_num_parallel_calls) {
  // Run the graph to see that the new num_parallel_calls is computed correctly.
  const NodeDef& map_node =
      output.node(graph_utils::FindGraphNodeWithOp(kParallelMapOp, output));
  const string& num_parallel_calls = map_node.input(1);
  std::vector<Tensor> output_tensors;
  EvaluateNodes(output, {num_parallel_calls}, &output_tensors);

  test::ExpectTensorEqual<int>(
      output_tensors.at(0),
      Tensor(static_cast<int32>(expected_num_parallel_calls)));
}

struct TestStruct {
  int original_num_parallel_calls;
  int batch_size;
  int expected_num_parallel_calls;
};

class NumParallelCallsTest : public ::testing::TestWithParam<TestStruct> {};

TEST_P(NumParallelCallsTest, TestCorrectNumParallelCalls) {
  auto params = GetParam();

  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto range_node = AddRangeNode(&graph);
  auto map_fn = AddMapFn(&graph);
  auto map_node =
      AddMapNode(&graph, range_node->name(), map_fn->signature().name(),
                 params.original_num_parallel_calls);
  auto batch_node = AddBatchNode(&graph, map_node->name(), /*v2=*/true,
                                 /*batch_size=*/params.batch_size);
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, map_node->op(), batch_node->op(), range_node->name());

  CheckNumParallelCalls(output, params.expected_num_parallel_calls);
}

TEST_P(NumParallelCallsTest, TestCorrectNumParallelCallsFused) {
  auto params = GetParam();

  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto range_node = AddRangeNode(&graph);
  auto map_fn = AddMapFn(&graph);
  auto map_and_batch_node =
      AddMapAndBatchNode(&graph, range_node->name(), map_fn->signature().name(),
                         params.batch_size, params.original_num_parallel_calls);
  ASSERT_NE(map_and_batch_node, nullptr);

  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, kParallelMapOp, kBatchV2Op, range_node->name());

  CheckNumParallelCalls(output, params.expected_num_parallel_calls);
}

INSTANTIATE_TEST_SUITE_P(
    NumParallelCalls, NumParallelCallsTest,
    ::testing::Values(TestStruct({1, 1, 1}), TestStruct({2, 10, 1}),
                      TestStruct({4, 3, 2}), TestStruct({10, 1, 10}),
                      TestStruct({-1, 1, -1}), TestStruct({-1, 10, -1})));

class ChainedMapAndBatchTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

// Tests:
// 1) map.batch.map.batch
// 2) map.batch.map_and_batch
// 3) map_and_batch.map.batch
// 4) map_and_batch.map_and_batch
TEST_P(ChainedMapAndBatchTest, IsVectorized) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto input_node = AddRangeNode(&graph);

  auto map_fn = AddMapFn(&graph);

  auto make_map_and_batch = [&graph, map_fn](NodeDef* input, bool fuse) {
    if (fuse) {
      return AddMapAndBatchNode(&graph, input->name(),
                                map_fn->signature().name());
    }
    auto map_node =
        AddMapNode(&graph, input->name(), map_fn->signature().name(), true);
    auto batch_node = AddBatchNode(&graph, map_node->name(), true);
    return batch_node;
  };

  auto map_and_batch_0 =
      make_map_and_batch(input_node, std::get<0>(GetParam()));
  auto map_and_batch_1 =
      make_map_and_batch(map_and_batch_0, std::get<1>(GetParam()));
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

// Not all dataset types have "output_shapes" and "output_types"
// attrs defined. Add a generic input node which may not have these attrs
// defined.
NodeDef* AddArbitraryInputNode(MutableGraphView* graph,
                               std::vector<PartialTensorShape>* output_shapes,
                               std::vector<DataType>* output_types) {
  std::vector<std::pair<string, FunctionDefHelper::AttrValueWrapper>> attrs;
  if (output_shapes) {
    attrs.push_back({kAttrNameOutputShapes, *output_shapes});
  }
  if (output_types) {
    attrs.push_back({kAttrNameOutputTypes, *output_types});
  }

  NodeDef result = NDef(/*name=*/"input", /*op=*/"InputDataset",
                        /*inputs=*/{},
                        /*attrs=*/attrs);

  graph_utils::SetUniqueGraphNodeName(result.name(), graph->graph(), &result);
  return graph->AddNode(std::move(result));
}

TEST(MapVectorizationTest, VectorizeWithUndefinedOutputShapes) {
  // Tests that the optimization doesn't break when the input to MapDataset
  // doesn't have an output_shapes attr defined. In this case, the map and
  // batch swap does not occur.
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  std::vector<DataType> input_types({DT_INT64});
  auto input_node = AddArbitraryInputNode(&graph, nullptr, &input_types);
  auto map_fn = AddMapFn(&graph);
  auto map_node =
      AddMapNode(&graph, input_node->name(), map_fn->signature().name());
  auto batch_node = AddBatchNode(&graph, map_node->name());
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
  MutableGraphView graph(&item.graph);
  std::vector<PartialTensorShape> input_shapes({{}});
  std::vector<DataType> input_types({DT_INT64});
  auto input_node = AddArbitraryInputNode(&graph, &input_shapes, &input_types);
  auto map_fn = AddMapFn(&graph);
  auto map_node =
      AddMapNode(&graph, input_node->name(), map_fn->signature().name());
  auto batch_node = AddBatchNode(&graph, map_node->name());
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
  MutableGraphView graph(&item.graph);
  std::vector<PartialTensorShape> input_shapes({{-1, 2}});
  std::vector<DataType> input_types({DT_INT64});
  auto input_node = AddArbitraryInputNode(&graph, &input_shapes, &input_types);
  auto map_fn = AddMapFn(&graph);
  auto map_node =
      AddMapNode(&graph, input_node->name(), map_fn->signature().name());
  auto batch_node = AddBatchNode(&graph, map_node->name());
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
  MutableGraphView graph(&item.graph);
  std::vector<PartialTensorShape> input_shapes({{1}});
  auto input_node = AddArbitraryInputNode(&graph, &input_shapes, nullptr);
  auto map_fn = AddMapFn(&graph);
  auto map_node =
      AddMapNode(&graph, input_node->name(), map_fn->signature().name());
  auto batch_node = AddBatchNode(&graph, map_node->name());
  MapVectorization optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  CheckVectorized(output, map_node->op(), batch_node->op(), input_node->name());
}

// TODO(rachelim): Add test that has a polymorphic function.

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
