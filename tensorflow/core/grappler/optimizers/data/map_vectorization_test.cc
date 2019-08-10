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
#include "tensorflow/core/grappler/optimizers/data/function_utils.h"
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
constexpr char kMapAndBatchOp[] = "MapAndBatchDataset";
constexpr char kMapOp[] = "MapDataset";
constexpr char kParallelMapOp[] = "ParallelMapDataset";
constexpr char kChooseFastestOp[] = "ChooseFastestBranchDataset";
constexpr char kPrefetchOp[] = "PrefetchDataset";
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

Status OptimizeWithMapVectorization(const GrapplerItem& item, GraphDef* output,
                                    bool use_choose_fastest) {
  MapVectorization optimizer;
  RewriterConfig_CustomGraphOptimizer config;
  if (use_choose_fastest) {
    (*config.mutable_parameter_map())["use_choose_fastest"].set_s("true");
  } else {
    (*config.mutable_parameter_map())["use_choose_fastest"].set_s("false");
  }
  TF_RETURN_IF_ERROR(optimizer.Init(&config));
  return optimizer.Optimize(nullptr, item, output);
}

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

NodeDef* AddPrefetchNode(MutableGraphView* graph, const string& input_dataset,
                         int64 buffer_size) {
  auto buffer_size_node = graph_utils::AddScalarConstNode(buffer_size, graph);
  NodeDef result =
      NDef(/*name=*/"prefetch", /*op=*/kPrefetchOp,
           /*inputs=*/{input_dataset, buffer_size_node->name()},
           /*attrs=*/
           {{kAttrNameOutputTypes, gtl::ArraySlice<DataType>({DT_INT64})},
            {kAttrNameOutputShapes, gtl::ArraySlice<TensorShape>({{}})}});

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

void CheckBranch(const FunctionDef& function, gtl::ArraySlice<string> ops) {
  for (int i = 0, size = ops.size(); i < size; ++i) {
    EXPECT_EQ(function.node_def(i).op(), ops[i]);
  }
}

const FunctionDef* GetFunction(const GraphDef& graph,
                               const string& function_name) {
  int found =
      graph_utils::FindGraphFunctionWithName(function_name, graph.library());
  if (found == -1) {
    return nullptr;
  }
  return &graph.library().function(found);
}

void CheckVectorizedWithoutChooseFastest(
    const GraphDef& output, gtl::ArraySlice<string> expected_vectorized_branch,
    const string& input_name) {
  std::vector<const NodeDef*> vectorized_branch;
  for (const auto& op : expected_vectorized_branch) {
    // This assumes that vectorized op is the only one that exists in the graph.
    // For our test cases, this is true (we don't have superfluous map/batch
    // nodes in other parts of the pipeline).
    ASSERT_EQ(graph_utils::FindAllGraphNodesWithOp(op, output).size(), 1);
    vectorized_branch.push_back(
        &output.node(graph_utils::FindGraphNodeWithOp(op, output)));
  }

  for (int i = 1; i < vectorized_branch.size() - 1; ++i) {
    const NodeDef* node = vectorized_branch[i];
    const NodeDef* next_node = vectorized_branch[i + 1];
    ASSERT_EQ(next_node->input(0), node->name());
  }
  ASSERT_EQ(vectorized_branch[0]->input(0), input_name);

  const NodeDef* vectorized_map_node = vectorized_branch[1];
  string function_name =
      vectorized_map_node->attr().at(kAttrNameF).func().name();

  const FunctionDef* function = GetFunction(output, function_name);
  ASSERT_NE(function, nullptr);
  EXPECT_EQ(function->node_def(0).op(), "Identity");
}

// Checks that a graph has undergone the map_vectorization transformation
// successfully, whereby the new graph has the shape:
//
//    input_node -------------> choose_fastest --> ...
//                               |f0    |f1
//                               |      |
//                               |      +---> new batch --> new map
//                               |
//                               +--> old map --> old batch
//
void CheckVectorizedWithChooseFastest(
    const GraphDef& output, gtl::ArraySlice<string> expected_vectorized_branch,
    gtl::ArraySlice<string> expected_original_branch,
    const string& input_name) {
  for (const auto& op :
       {kBatchOp, kBatchV2Op, kMapOp, kParallelMapOp, kMapAndBatchOp}) {
    // Check that the dataset nodes have been removed from the main graph.
    ASSERT_EQ(graph_utils::FindAllGraphNodesWithOp(op, output).size(), 0);
  }
  ASSERT_EQ(
      graph_utils::FindAllGraphNodesWithOp(kChooseFastestOp, output).size(), 1);
  const NodeDef& choose_fastest_node =
      output.node(graph_utils::FindGraphNodeWithOp(kChooseFastestOp, output));
  ASSERT_EQ(choose_fastest_node.input(0), input_name);

  const auto& functions_list = choose_fastest_node.attr().at("branches").list();

  // Branch 0: vectorized
  const FunctionDef* branch_0 =
      GetFunction(output, functions_list.func(0).name());
  ASSERT_NE(branch_0, nullptr);
  CheckBranch(*branch_0, expected_vectorized_branch);

  // Branch 1: original
  const FunctionDef* branch_1 =
      GetFunction(output, functions_list.func(1).name());
  ASSERT_NE(branch_1, nullptr);
  CheckBranch(*branch_1, expected_original_branch);

  const NodeDef& vectorized_map_node =
      branch_0->node_def(function_utils::FindFunctionNodeWithOp(
          expected_vectorized_branch[1], *branch_0));
  string function_name =
      vectorized_map_node.attr().at(kAttrNameF).func().name();

  const FunctionDef* function = GetFunction(output, function_name);
  ASSERT_NE(function, nullptr);
  EXPECT_EQ(function->node_def(0).op(), "Identity");
}

class MapThenBatchTest
    : public ::testing::TestWithParam<std::tuple<int, bool, int, bool>> {};

TEST_P(MapThenBatchTest, IsVectorized) {
  int num_parallel_calls = std::get<0>(GetParam());
  bool use_batch_v2 = std::get<1>(GetParam());
  int prefetch = std::get<2>(GetParam());
  bool use_choose_fastest = std::get<3>(GetParam());
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto range_dataset = AddRangeNode(&graph);
  auto map_fn = AddMapFn(&graph);
  auto dataset = AddMapNode(&graph, range_dataset->name(),
                            map_fn->signature().name(), num_parallel_calls);

  if (prefetch) {
    dataset = AddPrefetchNode(&graph, dataset->name(), prefetch);
  }
  dataset = AddBatchNode(&graph, dataset->name(), use_batch_v2);
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, use_choose_fastest));

  std::vector<string> expected_original_branch;
  expected_original_branch.push_back(num_parallel_calls > 0 ? kParallelMapOp
                                                            : kMapOp);
  if (prefetch) {
    expected_original_branch.push_back(kPrefetchOp);
  }
  expected_original_branch.push_back(use_batch_v2 > 0 ? kBatchV2Op : kBatchOp);

  std::vector<string> expected_vectorized_branch;
  expected_vectorized_branch.push_back(use_batch_v2 > 0 ? kBatchV2Op
                                                        : kBatchOp);
  expected_vectorized_branch.push_back(num_parallel_calls > 0 ? kParallelMapOp
                                                              : kMapOp);
  if (prefetch) {
    expected_vectorized_branch.push_back(kPrefetchOp);
  }

  if (use_choose_fastest) {
    CheckVectorizedWithChooseFastest(output, expected_vectorized_branch,
                                     expected_original_branch,
                                     range_dataset->name());

  } else {
    CheckVectorizedWithoutChooseFastest(output, expected_vectorized_branch,
                                        range_dataset->name());
  }
}

INSTANTIATE_TEST_SUITE_P(MapThenBatchTest, MapThenBatchTest,
                         ::testing::Combine(::testing::Values(0, 12),
                                            ::testing::Bool(),
                                            ::testing::Values(0, 20),
                                            ::testing::Bool()));

class MapAndBatchTest : public ::testing::TestWithParam<bool> {};

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
           /*op=*/kMapAndBatchOp,
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

TEST_P(MapAndBatchTest, VectorizeMapAndBatch) {
  GrapplerItem item;
  MutableGraphView graph(&item.graph);
  auto range_node = AddRangeNode(&graph);
  auto map_fn = AddMapFn(&graph);
  auto map_and_batch_node = AddMapAndBatchNode(&graph, range_node->name(),
                                               map_fn->signature().name());
  ASSERT_NE(map_and_batch_node, nullptr);

  GraphDef output;
  bool use_choose_fastest = GetParam();

  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, use_choose_fastest));
  if (use_choose_fastest) {
    CheckVectorizedWithChooseFastest(output, {kBatchV2Op, kParallelMapOp},
                                     {kMapAndBatchOp}, range_node->name());
  } else {
    CheckVectorizedWithoutChooseFastest(output, {kBatchV2Op, kParallelMapOp},
                                        range_node->name());
  }
}

INSTANTIATE_TEST_SUITE_P(MapAndBatchTest, MapAndBatchTest, ::testing::Bool());

class ChainedMapAndBatchTest
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

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

  bool fuse_0 = std::get<0>(GetParam());
  bool fuse_1 = std::get<1>(GetParam());
  bool use_choose_fastest = std::get<2>(GetParam());
  auto map_and_batch_0 = make_map_and_batch(input_node, fuse_0);
  auto map_and_batch_1 = make_map_and_batch(map_and_batch_0, fuse_1);
  ASSERT_NE(map_and_batch_1, nullptr);

  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, use_choose_fastest));
  TF_ASSERT_OK(TopologicalSort(&output));

  if (use_choose_fastest) {
    std::vector<int> choose_fastest_nodes =
        graph_utils::FindAllGraphNodesWithOp(kChooseFastestOp, output);
    ASSERT_EQ(choose_fastest_nodes.size(), 2);

    std::vector<string> fused_sequence({kMapAndBatchOp});
    std::vector<string> unfused_sequence({kParallelMapOp, kBatchV2Op});
    const NodeDef& range_node =
        output.node(graph_utils::FindGraphNodeWithOp(kRangeOp, output));
    const NodeDef& choose_fastest_0 = output.node(choose_fastest_nodes[0]);
    ASSERT_EQ(choose_fastest_0.input(0), range_node.name());
    const NodeDef& choose_fastest_1 = output.node(choose_fastest_nodes[1]);
    ASSERT_EQ(choose_fastest_1.input(0), choose_fastest_0.name());

    auto check_branches = [&output](const NodeDef& choose_fastest_node,
                                    gtl::ArraySlice<string> original_ops) {
      const auto& functions_list =
          choose_fastest_node.attr().at("branches").list();

      // Branch 0: vectorized
      const FunctionDef* branch_0 =
          GetFunction(output, functions_list.func(0).name());
      ASSERT_NE(branch_0, nullptr);
      CheckBranch(*branch_0, {kBatchV2Op, kParallelMapOp});

      // Branch 1: original
      const FunctionDef* branch_1 =
          GetFunction(output, functions_list.func(1).name());
      ASSERT_NE(branch_1, nullptr);
      CheckBranch(*branch_1, original_ops);
    };

    check_branches(choose_fastest_0,
                   fuse_0 ? fused_sequence : unfused_sequence);
    check_branches(choose_fastest_1,
                   fuse_1 ? fused_sequence : unfused_sequence);
  } else {
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
}

INSTANTIATE_TEST_SUITE_P(ChainedMapAndBatchTest, ChainedMapAndBatchTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool(),
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
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, true));
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
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, true));
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
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, true));
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
  GraphDef output;
  TF_ASSERT_OK(OptimizeWithMapVectorization(item, &output, true));
  CheckVectorizedWithChooseFastest(
      output, /*expected_vectorized_branch=*/{batch_node->op(), map_node->op()},
      /*expected_original_branch=*/{map_node->op(), batch_node->op()},
      input_node->name());
}

// TODO(rachelim): Add test that has a polymorphic function.

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
