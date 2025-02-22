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

#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/common_shape_fns.h"
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

std::vector<string> GetNodeNames(const FunctionDef& func) {
  std::vector<string> node_names;
  for (const NodeDef& node : func.node_def()) {
    node_names.push_back(node.name());
  }
  return node_names;
}

class SplitMapTest : public ::testing::TestWithParam<std::tuple<bool, bool>> {};

// Test case where the function for ParallelMap or MapAndBatch is split into
// two.
TEST_P(SplitMapTest, SplitMapFunction) {
  using test::function::NDef;
  GrapplerItem item;
  bool deterministic, rewrite_map_and_batch;
  std::tie(deterministic, rewrite_map_and_batch) = GetParam();
  if (deterministic && rewrite_map_and_batch) {
    LOG(INFO) << "Skipping test because MapAndBatch does not have "
                 "'deterministic' attribute";
    return;
  }

  FunctionDef orig_func_def = FunctionDefHelper::Create(
      // function_name
      "MyFunction",
      // in_def
      {"a1: float", "a2: float", "a3: double"},
      // out_def
      {"o1: float", "o2: double"},
      // attr_def
      {},
      // node_def
      {
          {{"i1"}, "Identity", {"a2"}, {{"T", DT_FLOAT}}},
          {{"i2"}, "Identity", {"i1:output"}, {{"T", DT_FLOAT}}},
          // Use SampleDistortedBoundingBox because it is a random op with two
          // inputs. The details of the op are unimportant.
          {{"stateful"},
           "SampleDistortedBoundingBox",
           {"a1", "i2:output"},
           {{"T", DT_FLOAT}}},
          {{"i3"}, "Identity", {"stateful:bboxes:0"}, {{"T", DT_FLOAT}}},
          {{"i4"}, "Identity", {"a3"}, {{"T", DT_DOUBLE}}},
      },
      // ret_def
      {{"o1", "i3:output"}, {"o2", "i4:output"}});

  NodeDef orig_map_node_def;
  if (rewrite_map_and_batch) {
    orig_map_node_def = graph_tests_utils::MakeMapAndBatchNode(
        "map", "range", "batch_size", "num_parallel_calls", "drop_remainder",
        "MyFunction");
  } else {
    orig_map_node_def = graph_tests_utils::MakeParallelMapV2Node(
        "map", "range", "num_parallel_calls", "MyFunction",
        deterministic ? "true" : "false", /*use_unbounded_threadpool=*/false);
  }
  orig_map_node_def.add_input("^start");
  AttrValue* attr_val = &(*orig_map_node_def.mutable_attr())["Targuments"];
  SetAttrValue(std::vector<DataType>{DT_DOUBLE}, attr_val);
  (*orig_map_node_def.mutable_attr())["preserve_cardinality"].set_b(true);
  attr_val = &(*orig_map_node_def.mutable_attr())["output_types"];
  SetAttrValue(std::vector<DataType>{DT_FLOAT, DT_DOUBLE}, attr_val);
  attr_val = &(*orig_map_node_def.mutable_attr())["output_shapes"];
  SetAttrValue(std::vector<TensorShape>{{1}, {1}}, attr_val);

  item.graph = test::function::GDef(
      {NDef("start", "Const", {}, {{"value", 0}, {"dtype", DT_INT32}}),
       NDef("stop", "Const", {}, {{"value", 10}, {"dtype", DT_INT32}}),
       NDef("step", "Const", {}, {{"value", 1}, {"dtype", DT_INT32}}),
       NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("range", "RangeDataset", {"start", "stop", "step"}, {}),
       NDef("num_parallel_calls", "Const", {},
            {{"value", 2}, {"dtype", DT_INT32}}),
       orig_map_node_def},
      // FunctionLib
      {orig_func_def});

  MakeDeterministic optimizer;
  GraphDef output;
  VLOG(1) << "GraphDef before optimization:\n"
          << item.graph.DebugString() << "\n\n";
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  VLOG(1) << "GraphDef after optimization:\n" << output.DebugString() << "\n\n";

  // Get first_map_node_def and assert inputs are correct
  int index = graph_utils::FindGraphNodeWithOp("MapDataset", output);
  ASSERT_GE(index, 0);
  NodeDef first_map_node_def = output.node(index);
  if (rewrite_map_and_batch) {
    ASSERT_THAT(
        first_map_node_def.input(),
        ::testing::ElementsAre("range", "^batch_size", "^num_parallel_calls",
                               "^drop_remainder", "^start"));
  } else {
    ASSERT_THAT(
        first_map_node_def.input(),
        ::testing::ElementsAre("range", "^num_parallel_calls", "^start"));
  }

  // Assert attributes of first_map_node_def
  std::vector<DataType> t_arguments;
  TF_ASSERT_OK(GetNodeAttr(first_map_node_def, "Targuments", &t_arguments));
  ASSERT_THAT(t_arguments, ::testing::ElementsAre(DT_DOUBLE));
  std::vector<DataType> output_types;
  TF_ASSERT_OK(GetNodeAttr(first_map_node_def, "output_types", &output_types));
  ASSERT_THAT(output_types, ::testing::ElementsAre(DT_FLOAT));
  std::vector<TensorShapeProto> output_shapes;
  TF_ASSERT_OK(
      GetNodeAttr(first_map_node_def, "output_shapes", &output_shapes));
  for (const TensorShapeProto& shape : output_shapes) {
    ASSERT_TRUE(shape.unknown_rank());
  }
  bool preserve_cardinality;
  TF_ASSERT_OK(GetNodeAttr(first_map_node_def, "preserve_cardinality",
                           &preserve_cardinality));
  ASSERT_TRUE(preserve_cardinality);

  // Assert function of first_map_node_def is correct
  NameAttrList f;
  TF_ASSERT_OK(GetNodeAttr(first_map_node_def, "f", &f));
  ASSERT_EQ(f.attr_size(), 0);
  index = graph_utils::FindGraphFunctionWithName(f.name(), output.library());
  CHECK_GE(index, 0);
  FunctionDef first_func = output.library().function(index);
  ASSERT_TRUE(first_func.signature().is_stateful());
  ASSERT_THAT(GetNodeNames(first_func),
              ::testing::UnorderedElementsAre("i1", "i2", "stateful"));

  // Get second_map_node_def and assert inputs are correct
  NodeDef second_map_node_def;
  if (rewrite_map_and_batch) {
    index = graph_utils::FindGraphNodeWithOp("MapAndBatchDataset", output);
    CHECK_GE(index, 0);
    second_map_node_def = output.node(index);
    ASSERT_THAT(second_map_node_def.input(),
                ::testing::ElementsAre(first_map_node_def.name(), "batch_size",
                                       "num_parallel_calls", "drop_remainder",
                                       "^start"));
  } else {
    index = graph_utils::FindGraphNodeWithOp("ParallelMapDatasetV2", output);
    CHECK_GE(index, 0);
    second_map_node_def = output.node(index);
    ASSERT_THAT(second_map_node_def.input(),
                ::testing::ElementsAre(first_map_node_def.name(),
                                       "num_parallel_calls", "^start"));
    ASSERT_EQ(second_map_node_def.attr().at("deterministic").s(), "true");
  }

  // Assert attributes of second_map_node_def
  t_arguments.clear();
  TF_ASSERT_OK(GetNodeAttr(second_map_node_def, "Targuments", &t_arguments));
  ASSERT_THAT(t_arguments, ::testing::ElementsAre(DT_DOUBLE));
  output_types.clear();
  TF_ASSERT_OK(GetNodeAttr(second_map_node_def, "output_types", &output_types));
  ASSERT_THAT(output_types, ::testing::ElementsAre(DT_FLOAT, DT_DOUBLE));
  output_shapes.clear();
  TF_ASSERT_OK(
      GetNodeAttr(first_map_node_def, "output_shapes", &output_shapes));
  for (const TensorShapeProto& shape : output_shapes) {
    ASSERT_EQ(shape.dim_size(), 0);
  }
  TF_ASSERT_OK(GetNodeAttr(first_map_node_def, "preserve_cardinality",
                           &preserve_cardinality));
  ASSERT_TRUE(preserve_cardinality);

  // Assert function of second_map_node_def is correct
  TF_ASSERT_OK(GetNodeAttr(second_map_node_def, "f", &f));
  ASSERT_EQ(f.attr_size(), 0);
  index = graph_utils::FindGraphFunctionWithName(f.name(), output.library());
  CHECK_GE(index, 0);
  FunctionDef second_func = output.library().function(index);
  ASSERT_THAT(GetNodeNames(second_func),
              ::testing::UnorderedElementsAre("i3", "i4"));
}

INSTANTIATE_TEST_SUITE_P(Test, SplitMapTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

// Returns a function that calls test::function::XTimesTwo.
FunctionDef OuterXTimesTwo() {
  return FunctionDefHelper::Define(
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
         {"f",
          FunctionDefHelper::FunctionRef("XTimesTwo", {{"T", DT_FLOAT}})}}}});
}

// Returns a function that calls test::function::RandomUniform.
FunctionDef OuterRandomUniform() {
  return FunctionDefHelper::Define(
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
         {"f", FunctionDefHelper::FunctionRef("RandomUniformFn",
                                              {{"T", DT_FLOAT}})}}}});
}

// Returns a function that calls test::function::ReadResourceVariable.
FunctionDef OuterReadResourceVariable() {
  return FunctionDefHelper::Define(
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
         {"f", FunctionDefHelper::FunctionRef("ReadResourceVariable", {})}}}});
}

// FunctionDef OuterRandomUn

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
      {test::function::XTimesTwo(), OuterXTimesTwo()});

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
           deterministic ? "true" : "false",
           /*use_unbounded_threadpool=*/false)},
      // FunctionLib
      {test::function::XTimesTwo(), OuterXTimesTwo()});

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
      {test::function::RandomUniform(), OuterRandomUniform()});

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
           deterministic ? "true" : "false",
           /*use_unbounded_threadpool=*/false),
       graph_tests_utils::MakePrefetchNode("prefetch", "map", "buffer_size")},
      // FunctionLib
      {test::function::RandomUniform(), OuterRandomUniform()});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("prefetch", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "PrefetchDataset");
  ASSERT_EQ(node_def.input_size(), 2);
  // The "map" node may be rewritten, so just assert the input ends with "map".
  ASSERT_THAT(node_def.input(0), ::testing::EndsWith("map"));
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
      {test::function::RandomUniform(), OuterRandomUniform()});

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

// Reason why a map function cannot be split. FUNC_HAS_ATTR means the
// function has an attribute. ASYNC_NONDETERMINISM means there is nondeterminism
// due to asynchrony (splitting ParallelMap only fixes parallel nondeterminism,
// not asynchrony nondeterminism).
enum CannotSplitReason { FUNC_HAS_ATTR, ASYNC_NONDETERMINISM };

class RewriteMapWithoutSplitTest
    : public ::testing::TestWithParam<
          std::tuple<bool, bool, CannotSplitReason>> {};

// Test case where the function for ParallelMap cannot be split, so instead
// ParallelMap is rewritten to Map
TEST_P(RewriteMapWithoutSplitTest, RewriteMapWithoutSplit) {
  using test::function::NDef;
  typedef FunctionDefHelper FDH;
  GrapplerItem item;
  bool nest, deterministic;
  CannotSplitReason reason;
  std::tie(nest, deterministic, reason) = GetParam();
  FunctionDef func;
  FunctionDef outer_func;
  if (reason == FUNC_HAS_ATTR) {
    func = test::function::RandomUniform();
    (*func.mutable_attr())["test_attr"].set_s("test_value");
    outer_func = OuterRandomUniform();
    (*outer_func.mutable_attr())["test_attr"].set_s("test_value");
  } else {
    func = test::function::ReadResourceVariable();
    outer_func = OuterReadResourceVariable();
  }

  std::string func_name =
      nest ? outer_func.signature().name() : func.signature().name();

  NodeDef map_node_def = graph_tests_utils::MakeParallelMapV2Node(
      "map", "range", "num_parallel_calls", func_name,
      deterministic ? "true" : "false", /*use_unbounded_threadpool=*/false);
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
      {func, outer_func});

  VLOG(1) << "Orig graph: \n" << item.graph.DebugString() << "\n\n";

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

  NameAttrList f;
  TF_ASSERT_OK(GetNodeAttr(node_def, "f", &f));
  ASSERT_EQ(f.name(), func_name);
  ASSERT_FALSE(graph_utils::ContainsNodeWithOp("ParallelMapDatasetV2", output));
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
      {test::function::ReadResourceVariable(), OuterReadResourceVariable()});

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
           deterministic ? "true" : "false",
           /*use_unbounded_threadpool=*/false),
       graph_tests_utils::MakePrefetchNode("prefetch", "map", "buffer_size")},
      // FunctionLib
      {test::function::ReadResourceVariable(), OuterReadResourceVariable()});

  MakeDeterministic optimizer;
  GraphDef output;
  TF_ASSERT_OK(optimizer.Optimize(nullptr, item, &output));
  int index = graph_utils::FindGraphNodeWithName("prefetch", output);
  ASSERT_GE(index, 0);
  NodeDef node_def = output.node(index);
  ASSERT_EQ(node_def.op(), "PrefetchDataset");
  ASSERT_EQ(node_def.input_size(), 3);
  ASSERT_THAT(node_def.input(0), ::testing::EndsWith("map"));
  ASSERT_EQ(node_def.input(2), "^buffer_size");
  NodeDef buffer_size = output.node(
      graph_utils::FindGraphNodeWithName(node_def.input(1), output));
  EXPECT_EQ(buffer_size.attr().at("value").tensor().int64_val(0), 0);
}

INSTANTIATE_TEST_SUITE_P(Test, MakeDeterministicTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Test, RewriteMapWithoutSplitTest,
    ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                       ::testing::Values(FUNC_HAS_ATTR, ASYNC_NONDETERMINISM)));

TEST(NoRewriteMapAndBatchTest, NoRewriteMapAndBatch) {
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

class RewriteMapAndBatchWithoutSplitTest
    : public ::testing::TestWithParam<std::tuple<bool, CannotSplitReason>> {};

// Test case where the function for MapAndBatch cannot be split, so instead
// MapAndBatch is rewritten to separate Map and Batch ops
TEST_P(RewriteMapAndBatchWithoutSplitTest, RewriteMapAndBatchWithoutSplit) {
  using test::function::NDef;
  GrapplerItem item;
  bool nest;
  CannotSplitReason reason;
  std::tie(nest, reason) = GetParam();

  FunctionDef func;
  if (reason == FUNC_HAS_ATTR) {
    func = test::function::RandomUniform();
    (*func.mutable_attr())["test_attr"].set_s("test_value");
  } else {
    func = test::function::ReadResourceVariable();
  }

  NodeDef map_and_batch_node_def = graph_tests_utils::MakeMapAndBatchNode(
      "map_and_batch", "range", "batch_size", "num_parallel_calls",
      "drop_remainder", func.signature().name());
  SetAttrValue(
      absl::Span<const PartialTensorShape>{
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
            {{"value", 2}, {"dtype", DT_INT32}}),
       NDef("drop_remainder", "Const", {},
            {{"value", false}, {"dtype", DT_BOOL}}),
       map_and_batch_node_def},
      // FunctionLib
      {func});

  VLOG(1) << "Orig graph: \n" << item.graph.DebugString() << "\n\n";

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

INSTANTIATE_TEST_SUITE_P(
    Test, RewriteMapAndBatchWithoutSplitTest,
    ::testing::Combine(::testing::Bool(),
                       ::testing::Values(FUNC_HAS_ATTR, ASYNC_NONDETERMINISM)));

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
