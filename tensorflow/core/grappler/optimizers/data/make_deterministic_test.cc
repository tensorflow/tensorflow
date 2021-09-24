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

#include "tensorflow/core/grappler/optimizers/data/make_deterministic.h"

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/determinism.h"

namespace tensorflow {
namespace grappler {
namespace {

class MakeDeterministicTest
    : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

TEST_P(MakeDeterministicTest, NoRewriteInterleave) {
  using test::function::NDef;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name = nest ? "OuterXTimesTwo" : "XTimesTwo";

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelInterleaveV2Node(
           "interleave", "range", "cycle_length", "block_length",
           "num_parallel_calls", func_name, /*sloppy=*/!deterministic)},
      // FunctionLib
      {test::function::XTimesTwo(),
       FunctionDefHelper::Define(
           // Name
           "OuterXTimesTwo",
           // Args
           {"x: float"},
           // Return values
           {"y: float"},
           // Attr def
           {},
           {{{"y"},
             "PartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_FLOAT}},
              {"f", FunctionDefHelper::FunctionRef("XTimesTwo",
                                                   {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("interleave", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "ParallelInterleaveDatasetV2");
  ASSERT_EQ(node_def.attr().at("sloppy").b(), false);
}

TEST_P(MakeDeterministicTest, NoRewriteMap) {
  using test::function::NDef;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name = nest ? "OuterXTimesTwo" : "XTimesTwo";
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       graph_tests_utils::MakeParallelMapV2Node(
           "map", "range", "num_parallel_calls", func_name,
           deterministic ? "true" : "false")},
      // FunctionLib
      {test::function::XTimesTwo(),
       FunctionDefHelper::Define(
           // Name
           "OuterXTimesTwo",
           // Args
           {"x: float"},
           // Return values
           {"y: float"},
           // Attr def
           {},
           {{{"y"},
             "PartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_FLOAT}},
              {"f", FunctionDefHelper::FunctionRef("XTimesTwo",
                                                   {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("map", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "ParallelMapDatasetV2");
  ASSERT_EQ(node_def.attr().at("deterministic").s(), "true");
}

TEST_P(MakeDeterministicTest, NoRewriteBatch) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name = nest ? "OuterRandomUniform" : "RandomUniformFn";
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       graph_tests_utils::MakeMapNode("map", "range", func_name),
       graph_tests_utils::MakeParallelBatchNode(
           "batch", "map", "batch_size", "num_parallel_calls", "drop_remainder",
           deterministic ? "true" : "false")},
      // FunctionLib
      {test::function::RandomUniform(),
       FDH::Define(
           // Name
           "OuterRandomUniform",
           // Args
           {"x: float"},
           // Return values
           {"random_uniform: int64"},
           // Attr def
           {},
           {{{"random_uniform"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_INT64}},
              {"f",
               FDH::FunctionRef("RandomUniformFn", {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("batch", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "ParallelBatchDataset");
  ASSERT_EQ(node_def.attr().at("deterministic").s(), "true");
}

TEST_P(MakeDeterministicTest, NoRewritePrefetch) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name = nest ? "OuterRandomUniform" : "RandomUniformFn";
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("buffer_size", "Const", {},
            {{"value", Tensor(int64_t{1})}, {"dtype", DT_INT64}}),
       graph_tests_utils::MakeParallelMapV2Node(
           "map", "range", "num_parallel_calls", func_name,
           deterministic ? "true" : "false"),
       graph_tests_utils::MakePrefetchNode("prefetch", "map", "buffer_size")},
      // FunctionLib
      {test::function::RandomUniform(),
       FDH::Define(
           // Name
           "OuterRandomUniform",
           // Args
           {"x: float"},
           // Return values
           {"random_uniform: int64"},
           // Attr def
           {},
           {{{"random_uniform"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_INT64}},
              {"f",
               FDH::FunctionRef("RandomUniformFn", {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("prefetch", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "PrefetchDataset");
  ASSERT_EQ(node_def.input_size(), 2);
  ASSERT_EQ(node_def.input(0), "map");
  ASSERT_EQ(node_def.input(1), "buffer_size");
  NodeDef buffer_size =
      output.node(graph_utils::FindGraphNodeWithName("buffer_size", output));
  EXPECT_EQ(buffer_size.attr().at("value").tensor().int64_val(0), 1);
}

TEST_P(MakeDeterministicTest, RewriteInterleave) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name = nest ? "OuterRandomUniform" : "RandomUniformFn";

  NodeDef interleave_node_def = graph_tests_utils::MakeParallelInterleaveV2Node(
      "interleave", "range", "cycle_length", "block_length",
      "num_parallel_calls", func_name, /*sloppy=*/!deterministic);
  interleave_node_def.add_input("^start");

  // Rewrite occurs due to parallelism in interleave function
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("cycle_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("block_length", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       interleave_node_def},
      // FunctionLib
      {test::function::RandomUniform(),
       FDH::Define(
           // Name
           "OuterRandomUniform",
           // Args
           {"x: float"},
           // Return values
           {"random_uniform: int64"},
           // Attr def
           {},
           {{{"random_uniform"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_INT64}},
              {"f",
               FDH::FunctionRef("RandomUniformFn", {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithOp("InterleaveDataset", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.input_size(), 5);
  ASSERT_EQ(node_def.input(0), "range");
  ASSERT_EQ(node_def.input(1), "cycle_length");
  ASSERT_EQ(node_def.input(2), "block_length");
  ASSERT_EQ(node_def.input(3), "^num_parallel_calls");
  ASSERT_EQ(node_def.input(4), "^start");
}

TEST_P(MakeDeterministicTest, RewriteMap) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name = nest ? "OuterRandomUniform" : "RandomUniformFn";

  NodeDef map_node_def = graph_tests_utils::MakeParallelMapV2Node(
      "map", "range", "num_parallel_calls", func_name,
      deterministic ? "true" : "false");
  map_node_def.add_input("^start");

  // Rewrite occurs due to parallelism in map function
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       map_node_def},
      // FunctionLib
      {test::function::RandomUniform(),
       FDH::Define(
           // Name
           "OuterRandomUniform",
           // Args
           {"x: float"},
           // Return values
           {"random_uniform: int64"},
           // Attr def
           {},
           {{{"random_uniform"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_FLOAT}},
              {"Tout", DataTypeSlice{DT_INT64}},
              {"f",
               FDH::FunctionRef("RandomUniformFn", {{"T", DT_FLOAT}})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithOp("MapDataset", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.input_size(), 3);
  ASSERT_EQ(node_def.input(0), "range");
  ASSERT_EQ(node_def.input(1), "^num_parallel_calls");
  ASSERT_EQ(node_def.input(2), "^start");
}

TEST_P(MakeDeterministicTest, RewriteBatch) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name =
      nest ? "OuterReadResourceVariable" : "ReadResourceVariable";

  NodeDef batch_node_def = graph_tests_utils::MakeParallelBatchNode(
      "batch", "map", "batch_size", "num_parallel_calls", "drop_remainder",
      deterministic ? "true" : "false");
  batch_node_def.add_input("^start");

  // Rewrite occurs due to the asynchronicity of the stateful map function
  // in the pipeline
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       graph_tests_utils::MakeMapNode("map", "range", func_name),
       batch_node_def},
      // FunctionLib
      {test::function::ReadResourceVariable(),
       FDH::Define(
           // Name
           "OuterReadResourceVariable",
           // Args
           {"x: resource"},
           // Return values
           {"y: float"},
           // Attr def
           {},
           {{{"y"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_RESOURCE}},
              {"Tout", DataTypeSlice{DT_FLOAT}},
              {"f", FDH::FunctionRef("ReadResourceVariable", {})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithOp("BatchDatasetV2", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.input_size(), 5);
  ASSERT_EQ(node_def.input(0), "map");
  ASSERT_EQ(node_def.input(1), "batch_size");
  ASSERT_EQ(node_def.input(2), "drop_remainder");
  ASSERT_EQ(node_def.input(3), "^num_parallel_calls");
  ASSERT_EQ(node_def.input(4), "^start");
  ASSERT_EQ(node_def.attr().count("deterministic"), 0);
}

TEST_P(MakeDeterministicTest, RewritePrefetch) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  std::tie(nest, deterministic) = GetParam();
  std::string func_name =
      nest ? "OuterReadResourceVariable" : "ReadResourceVariable";

  // Rewrite occurs due to the asynchronicity of the stateful map function
  // in the pipeline
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("buffer_size", "Const", {},
            {{"value", Tensor(int64_t{1})}, {"dtype", DT_INT64}}),
       graph_tests_utils::MakeParallelMapV2Node(
           "map", "range", "num_parallel_calls", func_name,
           deterministic ? "true" : "false"),
       graph_tests_utils::MakePrefetchNode("prefetch", "map", "buffer_size")},
      // FunctionLib
      {test::function::ReadResourceVariable(),
       FDH::Define(
           // Name
           "OuterReadResourceVariable",
           // Args
           {"x: resource"},
           // Return values
           {"y: float"},
           // Attr def
           {},
           {{{"y"},
             "StatefulPartitionedCall",
             {"x"},
             {{"Tin", DataTypeSlice{DT_RESOURCE}},
              {"Tout", DataTypeSlice{DT_FLOAT}},
              {"f", FDH::FunctionRef("ReadResourceVariable", {})}}}})});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("prefetch", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "PrefetchDataset");
  ASSERT_EQ(node_def.input_size(), 3);
  ASSERT_EQ(node_def.input(0), "map");
  ASSERT_EQ(node_def.input(2), "^buffer_size");
  NodeDef buffer_size = output.node(
      graph_utils::FindGraphNodeWithName(node_def.input(1), output));
  EXPECT_EQ(buffer_size.attr().at("value").tensor().int64_val(0), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, MakeDeterministicTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

TEST(MakeDeterministicMapAndBatchTest, NoRewriteMapAndBatch) {
  using test::function::NDef;
  GrapplerItem item;
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT64}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT64}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       graph_tests_utils::MakeMapAndBatchNode(
           "map_and_batch", "range", "batch_size", "num_parallel_calls",
           "drop_remainder", "XTimesTwo")},
      // FunctionLib
      {test::function::XTimesTwo()});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("map_and_batch", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.input_size(), 4);
  ASSERT_EQ(node_def.input(0), "range");
  ASSERT_EQ(node_def.input(1), "batch_size");
  ASSERT_EQ(node_def.input(2), "num_parallel_calls");
  ASSERT_EQ(node_def.input(3), "drop_remainder");
}

TEST(MakeDeterministicMapAndBatchTest, RewriteMapAndBatch) {
  using test::function::NDef;
  GrapplerItem item;

  NodeDef map_and_batch_node_def = graph_tests_utils::MakeMapAndBatchNode(
      "map_and_batch", "range", "batch_size", "num_parallel_calls",
      "drop_remainder", "RandomUniformFn");
  SetAttrValue(
      gtl::ArraySlice<PartialTensorShape>{
          {2}, {-1, 3, -1}, PartialTensorShape()},
      &(*map_and_batch_node_def.mutable_attr())["output_shapes"]);

  // Rewrite occurs due to parallelism in map function
  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT64}}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT64}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       map_and_batch_node_def},
      // FunctionLib
      {test::function::RandomUniform()});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  ASSERT_FALSE(graph_utils::ContainsNodeWithOp("MapAndBatchDataset", output));
  int index = graph_utils::FindGraphNodeWithOp("MapDataset", output);
  ASSERT_GE(index, 0);
  NodeDef map_node_def = output.node(index);
  ASSERT_EQ(map_node_def.input_size(), 4);
  ASSERT_EQ(map_node_def.input(0), "range");
  ASSERT_EQ(map_node_def.input(1), "^batch_size");
  ASSERT_EQ(map_node_def.input(2), "^num_parallel_calls");
  ASSERT_EQ(map_node_def.input(3), "^drop_remainder");
  ASSERT_TRUE(AreAttrValuesEqual(map_and_batch_node_def.attr().at("f"),
                                 map_node_def.attr().at("f")));
  ASSERT_TRUE(AreAttrValuesEqual(map_and_batch_node_def.attr().at("Targuments"),
                                 map_node_def.attr().at("Targuments")));
  ASSERT_TRUE(
      AreAttrValuesEqual(map_and_batch_node_def.attr().at("output_types"),
                         map_node_def.attr().at("output_types")));
  ASSERT_EQ(map_node_def.attr().at("output_shapes").list().shape_size(), 3);
  ASSERT_TRUE(PartialTensorShape({}).IsIdenticalTo(
      map_node_def.attr().at("output_shapes").list().shape(0)));
  ASSERT_TRUE(PartialTensorShape({3, -1}).IsIdenticalTo(
      map_node_def.attr().at("output_shapes").list().shape(1)));
  ASSERT_TRUE(PartialTensorShape().IsIdenticalTo(
      map_node_def.attr().at("output_shapes").list().shape(2)));
  index = graph_utils::FindGraphNodeWithOp("BatchDatasetV2", output);
  ASSERT_GE(index, 0);
  NodeDef batch_node_def = output.node(index);
  ASSERT_EQ(batch_node_def.input_size(), 3);
  ASSERT_EQ(batch_node_def.input(0), map_node_def.name());
  ASSERT_EQ(batch_node_def.input(1), "batch_size");
  ASSERT_EQ(batch_node_def.input(2), "drop_remainder");
  ASSERT_TRUE(
      AreAttrValuesEqual(map_and_batch_node_def.attr().at("output_types"),
                         batch_node_def.attr().at("output_types")));
  ASSERT_TRUE(
      AreAttrValuesEqual(map_and_batch_node_def.attr().at("output_shapes"),
                         batch_node_def.attr().at("output_shapes")));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
